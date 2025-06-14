import atexit
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa.model import Model


from mesa_llm.recording.simulation_recorder import SimulationRecorder


def _attach_recorder_to_agents(model: "Model", recorder: SimulationRecorder):
    """Utility that iterates over all agents and attaches the recorder."""
    for agent in list(model.agents):
        # Only set if the attribute exists to avoid leaking recorder to non-LLM agents
        if hasattr(agent, "recorder"):
            agent.recorder = recorder


def record_model(
    cls: type["Model"] | None = None, *, recorder_kwargs: dict[str, Any] | None = None
) -> Callable[[type["Model"]], type["Model"]] | type["Model"]:
    """Class decorator that instruments a Mesa `Model` subclass with a :class:`SimulationRecorder`.

    Usage::

        from mesa_llm.recording.integration_hooks import record_model

        @record_model
        class MyModel(Model):
            ...

    The decorator will:
    1. Instantiate a ``SimulationRecorder`` after the model's original ``__init__`` completes and assign it to ``self.recorder``.
    2. Attach the same recorder to every agent that exposes a ``recorder`` attribute (e.g., subclasses of ``LLMAgent``).
    3. Wrap the model's ``step`` method to automatically record ``step_start`` and ``step_end`` events and ensure late-added agents also receive the recorder.
    4. Provide a convenience ``save_recording`` method on the model for persisting the captured simulation events.

    Parameters
    ----------
    recorder_kwargs : dict | None
        Extra keyword arguments forwarded to :class:`SimulationRecorder` when it is created.  This allows callers to
        customise output directory or disable certain event types::

            @record_model(recorder_kwargs={"output_dir": "my_runs", "auto_save_interval": 100})
            class MyModel(Model):
                ...
    """

    if cls is None:
        # Decorator was called with optional kwargs -> return wrapper awaiting the class
        return lambda actual_cls: record_model(
            actual_cls, recorder_kwargs=recorder_kwargs
        )  # type: ignore[misc]

    recorder_kwargs = recorder_kwargs or {}

    original_init = cls.__init__
    original_step = getattr(cls, "step", None)

    @wraps(original_init)
    def init_wrapper(self: "Model", *args, **kwargs):  # type: ignore[override]
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]
        # Create and attach recorder
        self.recorder = SimulationRecorder(model=self, **recorder_kwargs)  # type: ignore[attr-defined]
        _attach_recorder_to_agents(self, self.recorder)

        # Use a closure to capture a reference to `self`
        def _auto_save():
            try:
                # Avoid creating multiple identical files if already saved manually
                if hasattr(self, "recorder") and self.recorder.events:
                    self.save_recording()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[SimulationRecorder] Auto-save failed: {exc}")

        atexit.register(_auto_save)

    cls.__init__ = init_wrapper  # type: ignore[assignment]

    if original_step is not None:

        @wraps(original_step)
        def step_wrapper(self: "Model", *args, **kwargs):  # type: ignore[override]
            # Record beginning of step
            if hasattr(self, "recorder"):
                self.recorder.record_model_event("step_start", {"step": self.steps})  # type: ignore[attr-defined]

            # Execute the original step logic
            result = original_step(self, *args, **kwargs)  # type: ignore[misc]

            # Make sure any new agents created during the step also receive the recorder
            if hasattr(self, "recorder"):
                _attach_recorder_to_agents(self, self.recorder)  # type: ignore[attr-defined]
                # Record end of step after agents have acted
                self.recorder.record_model_event("step_end", {"step": self.steps})  # type: ignore[attr-defined]

            return result

        cls.step = step_wrapper  # type: ignore[assignment]

    # Convenience helper so that user code can save the recording easily
    def save_recording(
        self: "Model", filename: str | None = None, format: str = "json"
    ):
        if not hasattr(self, "recorder"):
            raise AttributeError(
                "Recorder not initialised - did you forget to decorate the model with @record_model?"
            )
        return self.recorder.save(filename=filename, format=format)  # type: ignore[attr-defined]

    cls.save_recording = save_recording  # type: ignore[attr-defined]

    return cls
