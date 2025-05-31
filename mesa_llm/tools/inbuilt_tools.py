import inspect
import sys
from typing import TYPE_CHECKING

from .tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool
def move_to_location(agent: "LLMAgent", target_coordinates: tuple[float, float]) -> str:
    """
    Move to a given location in a discrete grid.

    Args:
        agent: The agent to move.
        target_coordinates: The target coordinates to move to, specified as a tuple of (x, y) floats.

    Returns:
        A string indicating the agent's new position.
    """

    agent.position = target_coordinates

    return f"This agent moved to {target_coordinates}."


@tool
def speak_to(
    speaker_agent: "LLMAgent", listener_agents: list["LLMAgent"], message: str
) -> str:
    """
    Send a message to the recipients and commits it to their memory.
    Args:
        speaker_agent: The agent sending the message
        listener_agents: The agents receiving the message
        message: The message to send
    Returns:
        A string indicating the message sent, the sender and the recipients
    """
    for recipient in [*listener_agents, speaker_agent]:
        recipient.memory.add_to_memory(
            type="Message",
            content=message,
            step=speaker_agent.model.steps,
            metadata={
                "sender": speaker_agent,
                "recipients": listener_agents,
            },
        )
    return f"{speaker_agent} → {listener_agents} : {message}"


# Get all the functions in the module that are not private into a list
inbuilt_tools = [
    obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    if not name.startswith("_")
]
