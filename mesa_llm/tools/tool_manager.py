import inspect
from collections.abc import Callable

"""
    The user can use insntaces of ToolManager to regsiter functions as tools through the decorator.
    The user can also use the ToolManager instance to get the schema of the tools, call a tool with validated arguments, and check if a tool is registered.
    Moreover, the user can group like tools together by creating a new ToolManager instance and registering the tools to it.
    So if agent A requires tools A1, A2, and A3, and agent B requires tools B1, B2, and B3, the user can create two ToolManager instances: tool_manager_A and tool_manager_B.
"""

class ToolManager:
    def __init__(self):
        self.tools: dict[str, Callable] = {}

    def register(self, fn: Callable):
        """Register a tool function by name"""
        name = fn.__name__
        self.tools[name] = fn #storing the name & function pair as a dicitonary

    def get_schema(self) -> list[dict]:
        """Return schema in the liteLLM format"""
        #we need to convert the function signature from python to a JSON schema
        py_to_json_type = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        schema = []
        for name, fn in self.tools.items():
            sig = inspect.signature(fn)
            properties = {}
            required = []

            for param in sig.parameters.values():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    # skip *args and **kwargs
                    continue
                param_schema = {
                    "description": f"{param.name} parameter"
                }

                # If type annotation is available
                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation

                    json_type = py_to_json_type.get(annotation)

                    if json_type:
                        param_schema["type"] = json_type
                    else:
                        # fallback: allow any type
                        param_schema["type"] = ["string", "number", "boolean", "object", "array", "null"]
                else:
                    # No annotation so fallback
                    param_schema["type"] = ["string", "number", "boolean", "object", "array", "null"]

                properties[param.name] = param_schema

                if param.default == inspect.Parameter.empty:
                    required.append(param.name)

            schema.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schema #incase the user wants to change the something like say the parameter description, they will have to get the schema and edit it manually

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool with validated args"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name](**arguments)

    def has_tool(self, name: str) -> bool:
        return name in self.tools



