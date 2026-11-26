"""Tool registry: the capabilities agents can use, each with a typed (pydantic) input.

Rule-based agents call tools through the registry. Because `describe()` exports every tool's
JSON schema, an LLM-driven agent can call the very same tools (the dispatcher does).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Unknown tool or invalid arguments. The message is readable by people and by an LLM."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable[[Any], Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }


@dataclass
class ToolCall:
    """One entry of the registry's call log."""

    name: str
    args: dict[str, Any]
    ok: bool
    error: str = ""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.calls: list[ToolCall] = []

    def register(
        self, name: str, description: str, input_model: type[BaseModel]
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Decorator that registers a function taking one `input_model` argument."""

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            if name in self._tools:
                raise ValueError(f"tool {name!r} is already registered")
            self._tools[name] = Tool(name, description, input_model, func)
            return func

        return decorator

    def call(self, name: str, args: BaseModel | dict[str, Any]) -> Any:
        """Validate `args` against the tool's input model, then run the tool."""
        raw_args = args.model_dump(mode="json") if isinstance(args, BaseModel) else dict(args)
        tool = self._tools.get(name)
        if tool is None:
            error = f"unknown tool {name!r}; available tools: {', '.join(self.names())}"
            self.calls.append(ToolCall(name, raw_args, ok=False, error=error))
            raise ToolError(error)
        try:
            parsed = args if isinstance(args, tool.input_model) else tool.input_model.model_validate(raw_args)
        except ValidationError as exc:
            error = f"invalid arguments for {name!r}: {exc}"
            self.calls.append(ToolCall(name, raw_args, ok=False, error=error))
            raise ToolError(error) from exc
        result = tool.func(parsed)
        self.calls.append(ToolCall(name, raw_args, ok=True))
        return result

    def describe(self) -> list[dict[str, Any]]:
        """JSON schemas of all tools (for documentation or an LLM)."""
        return [self._tools[name].schema() for name in self.names()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
