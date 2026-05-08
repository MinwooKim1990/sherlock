"""Builtin tools + a `@sherlock.tool` decorator (SPEC §5.6)."""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import httpx


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    func: Callable[..., Any]


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def call(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name].func(**kwargs)

    def names(self) -> list[str]:
        return sorted(self.tools.keys())


def sherlock_tool(*, name: str | None = None, description: str = ""):
    """Decorator: register a Python function as a Sherlock tool."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        sig = inspect.signature(fn)
        properties: dict = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            ann = param.annotation if param.annotation is not inspect._empty else str
            json_type = _python_to_json(ann)
            properties[pname] = {"type": json_type}
            if param.default is inspect._empty:
                required.append(pname)
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        builtin_registry.register(
            Tool(name=tool_name, description=description, schema=schema, func=fn)
        )
        return fn

    return deco


def _python_to_json(t: Any) -> str:
    if t in (str, "str"):
        return "string"
    if t in (int, "int"):
        return "integer"
    if t in (float, "float"):
        return "number"
    if t in (bool, "bool"):
        return "boolean"
    if t in (list, "list"):
        return "array"
    if t in (dict, "dict"):
        return "object"
    return "string"


# ---- builtin implementations ----

builtin_registry = ToolRegistry()


def _current_time(tz_offset_hours: float = 0.0) -> dict:
    """Return ISO-8601 timestamp with optional UTC offset hours."""
    now = datetime.now(timezone.utc)
    return {
        "utc_iso": now.isoformat(),
        "epoch": now.timestamp(),
        "offset_hours_requested": tz_offset_hours,
    }


def _calculator(expression: str) -> dict:
    """Safe arithmetic eval."""
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": math.pow, "log": math.log,
        "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "expression": expression}
    return {"result": result, "expression": expression}


def _url_fetch(url: str, timeout: float = 10.0) -> dict:
    """Fetch a URL and return its body (truncated)."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            text = r.text[:8000]
            return {"url": url, "status": r.status_code, "text": text}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "url": url}


def _file_read(path: str, max_bytes: int = 100_000) -> dict:
    try:
        with open(path, "rb") as fp:
            data = fp.read(max_bytes)
        return {"path": path, "bytes_read": len(data), "text": data.decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": path}


# Public dataclass wrappers (per spec mention of CurrentTime, Calculator, UrlFetch).

@dataclass
class CurrentTime:
    @staticmethod
    def now() -> dict:
        return _current_time()


@dataclass
class Calculator:
    @staticmethod
    def eval(expression: str) -> dict:
        return _calculator(expression)


@dataclass
class UrlFetch:
    @staticmethod
    def get(url: str) -> dict:
        return _url_fetch(url)


# Register builtins.
builtin_registry.register(
    Tool(
        name="current_time",
        description="Return current UTC ISO timestamp + epoch.",
        schema={"type": "object", "properties": {"tz_offset_hours": {"type": "number"}}, "required": []},
        func=_current_time,
    )
)
builtin_registry.register(
    Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression.",
        schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        func=_calculator,
    )
)
builtin_registry.register(
    Tool(
        name="url_fetch",
        description="Fetch a URL and return its body (max 8K chars).",
        schema={
            "type": "object",
            "properties": {"url": {"type": "string"}, "timeout": {"type": "number"}},
            "required": ["url"],
        },
        func=_url_fetch,
    )
)
builtin_registry.register(
    Tool(
        name="file_read",
        description="Read a file from disk. Returns text (UTF-8, replacement on error).",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
            "required": ["path"],
        },
        func=_file_read,
    )
)
