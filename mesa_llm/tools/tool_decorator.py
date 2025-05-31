from __future__ import annotations

import inspect
import json
import re
import textwrap
import warnings
from collections.abc import Callable
from typing import Any, get_type_hints

_GLOBAL_TOOL_REGISTRY: dict[str, Callable] = {}


# ---------- helper functions ----------------------------------------------------


def _python_to_json_type(py_type: Any) -> str:
    json_type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        bytes: "string",
    }
    return json_type_map.get(py_type, "array" if py_type in (list, tuple) else "object")


class DocstringParsingError(Exception):
    """Raised when a Google-style docstring cannot be parsed."""


_ARG_HEADER_RE = re.compile(r"^\s*Args?:\s*$", re.IGNORECASE)
_RET_HEADER_RE = re.compile(r"^\s*Returns?:\s*$", re.IGNORECASE)
_PARAM_LINE_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


def _parse_docstring(
    func: callable,
) -> tuple[str, dict[str, str], str | None]:
    """
    Parse a function's Google-style docstring.
    Args:
        func: The function to parse the docstring of.

    Returns:
        summary: One-line/high-level description that appears before the *Args* section.
        param_desc: Mapping *param name → description* (text only, no types).
        return_desc: Description of the value in the *Returns* section, or *None* if that section is absent.
    """
    # ---------- fetch & pre-process -------------------------------------------------
    raw = inspect.getdoc(func) or ""
    if not raw:
        raise DocstringParsingError(f"{func.__name__} has no docstring.")

    # Normalise indentation & line endings
    lines = textwrap.dedent(raw).strip().splitlines()

    # ---------- locate block boundaries -------------------------------------------
    try:
        args_idx = next(i for i, ln in enumerate(lines) if _ARG_HEADER_RE.match(ln))
    except StopIteration:
        args_idx = None

    try:
        ret_idx = next(i for i, ln in enumerate(lines) if _RET_HEADER_RE.match(ln))
    except StopIteration:
        ret_idx = None

    # Short description = from top up to first blank line or Args:
    cut = (
        args_idx
        if args_idx is not None
        else ret_idx
        if ret_idx is not None
        else len(lines)
    )
    for i, ln in enumerate(lines[:cut]):
        if ln.strip() == "":
            cut = i
            break
    summary = " ".join(ln.strip() for ln in lines[:cut]).strip()

    # ---------- parse *Args* -------------------------------------------------------
    param_desc: dict[str, str] = {}
    if args_idx is not None:
        i = args_idx + 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1  # skip blank lines

        while i < len(lines) and (ret_idx is None or i < ret_idx):
            m = _PARAM_LINE_RE.match(lines[i])
            if not m:
                raise DocstringParsingError(
                    f"Malformed parameter line in {func.__name__}: '{lines[i]}'"
                )
            name, desc = m.groups()
            desc_lines = [desc.rstrip()]
            i += 1
            # grab any following indented continuation lines
            while (
                i < len(lines)
                and (ret_idx is None or i < ret_idx)
                and (lines[i].startswith(" ") or lines[i].startswith("\t"))
                and not _PARAM_LINE_RE.match(
                    lines[i]
                )  # Don't treat other parameters as continuation
            ):
                desc_lines.append(lines[i].strip())
                i += 1
            param_desc[name] = " ".join(desc_lines).strip()
            # skip possible extra blank lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1

    # ---------- parse *Returns* ----------------------------------------------------
    return_desc: str | None = None
    if ret_idx is not None:
        ret_body = [ln.strip() for ln in lines[ret_idx + 1 :] if ln.strip()]
        return_desc = " ".join(ret_body) if ret_body else None

    # ---------- validation ---------------------------------------------------------
    sig_params: list[str] = [
        p.name for p in inspect.signature(func).parameters.values()
    ]
    missing = [p for p in sig_params if p not in param_desc]
    if missing:
        raise DocstringParsingError(
            f"Docstring for {func.__name__} is missing descriptions for: {missing}"
        )

    return summary, param_desc, return_desc


# ---------- decorator ----------------------------------------------------


def tool(fn: Callable):
    """
    Decorate a function so it becomes an LLM-callable tool and is auto-registered.
    Return (description, {arg: description}, returns).
    """
    name = fn.__name__
    description, arg_docs, return_docs = _parse_docstring(fn)

    sig = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except NameError:
        # Fallback to using annotations directly if type_hints evaluation fails
        type_hints = getattr(fn, "__annotations__", {})

    properties = {}
    for param_name, _param in sig.parameters.items():
        raw_type = type_hints.get(param_name, Any)
        properties[param_name] = {
            "type": _python_to_json_type(raw_type),
            "description": arg_docs.get(param_name, ""),
        }
        if not arg_docs.get(param_name):
            warnings.warn(
                f'Missing docstring for argument "{param_name}" in tool "{name}"',
                stacklevel=2,
            )

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description + " returns: " + (return_docs or ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(sig.parameters),
            },
        },
    }

    fn.__tool_schema__ = schema  #  <-- expose to ToolManager
    _GLOBAL_TOOL_REGISTRY[name] = fn  #  <-- auto-register
    return fn


if __name__ == "__main__":
    # CL to execute this file: python -m mesa_llm.tools.tool_decorator
    @tool
    def dummy_function(location: str):
        """
        Move to a location.
        Args:
            location: The location to move to.
        Returns:
            The location moved to.
        """
        return f"Moved to {location}"

    print(json.dumps(dummy_function.__tool_schema__, indent=2))
