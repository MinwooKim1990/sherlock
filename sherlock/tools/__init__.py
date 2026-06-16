"""Tool layer (M7-light): builtin tools + custom decorator + web search.

Public surface for v0.3.0:

  * Builtin tools (Calculator / CurrentTime / UrlFetch / decorator).
  * Multi-provider web search via :mod:`sherlock.tools.web_search`.
  * Pre-built callables (`web_search_fn`, `fetch_url_fn`) and tool-schema
    generators (`make_openai_tools`, `make_anthropic_tools`,
    `dispatch_tool_call`) for *native* tool-calling integrations — i.e.
    the user lets their own LLM library handle tool_call loops and just
    points it at the Sherlock helpers.

The agentic, tag-driven path (`<<sherlock-tool: search "..." >>`) is
handled inside :mod:`sherlock.agent`; that path uses the same engine
instances exposed here.
"""

from __future__ import annotations

import json
from typing import Optional

from sherlock.tools.builtin import (
    Calculator,
    CurrentTime,
    Tool,
    ToolRegistry,
    UrlFetch,
    builtin_registry,
    sherlock_tool,
)
from sherlock.tools.memory_tool import (
    dispatch_memory,
    make_anthropic_memory_tool,
    make_openai_memory_tool,
    memory_entity,
    memory_lookup,
    memory_pinned,
    memory_timeline,
    parse_memory_payload,
)
from sherlock.tools.web_search import (
    BraveSearch,
    DuckDuckGoSearch,
    SearchEngine,
    StubSearch,
    TavilySearch,
    ValyuSearch,
    build_role_engines,
    build_search_engine,
    create_search,
)


def _resolve_engine(engine: Optional[SearchEngine | str]) -> SearchEngine:
    """Coerce a name / instance / None into a usable :class:`SearchEngine`.

    Defaults to DuckDuckGo (no API key needed).
    """
    if engine is None:
        return DuckDuckGoSearch()
    if isinstance(engine, SearchEngine):
        return engine
    if isinstance(engine, str):
        return create_search(engine)
    raise TypeError(
        f"engine must be SearchEngine, name string, or None; got {type(engine).__name__}"
    )


# ---------------------------------------------------------------------------
# Pre-built callables — handy for users who want to expose search/fetch via
# native tool-calling on their own LLM library.
# ---------------------------------------------------------------------------


def web_search_fn(
    query: str,
    *,
    engine: Optional[SearchEngine | str] = None,
    max_results: int = 5,
) -> list[dict]:
    """Run a web search and return a list of result dicts.

    Args:
        query: The natural-language search query.
        engine: A :class:`SearchEngine` instance, an engine name
            (``"duckduckgo"`` / ``"tavily"`` / ``"brave"`` / ``"valyu"``),
            or ``None`` (default → DuckDuckGo, no key required).
        max_results: Upper bound on results returned.

    Returns:
        ``[{"title": ..., "url": ..., "content": ..., "source": ...}, ...]``
        — each engine populates the same shape. On error the list has a
        single ``{"error": ...}`` entry so callers can keep going.
    """
    eng = _resolve_engine(engine)
    return eng.search(query, max_results=max_results)


def fetch_url_fn(
    url: str,
    *,
    engine: Optional[SearchEngine | str] = None,
    raw: bool = False,
    timeout: float = 15.0,
) -> dict:
    """Fetch a URL and return either readable text or raw HTML.

    Args:
        url: The page to fetch.
        engine: Engine to use (some providers like Tavily ship a custom
            extraction endpoint that beats trafilatura). ``None`` →
            DuckDuckGo, which falls back to the shared httpx + trafilatura
            implementation.
        raw: If True, return raw HTML in ``html`` (truncated). Otherwise
            return trafilatura-extracted readable text in ``text``.
        timeout: HTTP timeout in seconds.

    Returns:
        ``{"url": ..., "status": ..., "text": ...}`` (text mode) or
        ``{"url": ..., "status": ..., "html": ...}`` (raw mode) on
        success; ``{"error": ..., "url": ...}`` on failure.
    """
    eng = _resolve_engine(engine)
    return eng.fetch(url, raw=raw, timeout=timeout)


# ---------------------------------------------------------------------------
# Tool-schema generators — for native tool-calling integrations.
# ---------------------------------------------------------------------------

_SEARCH_DESCRIPTION = (
    "Run a fresh web search and return up to N results. Use this when the "
    "user asks about something time-sensitive (today's news, current "
    "prices, recent product releases, upcoming events) or about facts you "
    "don't already know. Results are not authoritative — cross-check at "
    "least two before quoting them."
)
_FETCH_DESCRIPTION = (
    "Fetch a single URL and return its contents. Default returns "
    "trafilatura-extracted readable text. Pass raw=true if you need the "
    "underlying HTML (rare — usually text is what you want)."
)


def make_openai_tools(
    engine: Optional[
        SearchEngine | str
    ] = None,  # noqa: ARG001 — engine is captured by dispatch, not schema
) -> list[dict]:
    """Return tool schemas in OpenAI Chat-Completions ``tools=`` format.

    Pair with :func:`dispatch_tool_call` in your tool_call loop.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": _SEARCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in natural language.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max number of results to return (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": _FETCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute URL to fetch.",
                        },
                        "raw": {
                            "type": "boolean",
                            "description": "If true, return raw HTML instead of text.",
                            "default": False,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def make_anthropic_tools(
    engine: Optional[SearchEngine | str] = None,  # noqa: ARG001 — engine is captured by dispatch
) -> list[dict]:
    """Return tool schemas in Anthropic Messages ``tools=`` format.

    Pair with :func:`dispatch_tool_call` in your tool_use handling loop.
    """
    return [
        {
            "name": "web_search",
            "description": _SEARCH_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_url",
            "description": _FETCH_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL to fetch.",
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "If true, return raw HTML instead of text.",
                    },
                },
                "required": ["url"],
            },
        },
    ]


def dispatch_tool_call(
    name: str,
    arguments: dict,
    *,
    engine: Optional[SearchEngine | str] = None,
) -> str:
    """Run a Sherlock tool by name and return a JSON string.

    Designed for use inside a native tool-call loop. ``arguments`` is the
    dict your LLM library hands you (already JSON-decoded). Returns a
    JSON string that you forward back as the tool-result message.

    Errors do NOT raise — they are encoded as ``{"error": "..."}`` JSON
    so the LLM can self-correct without crashing the host process.
    """
    try:
        if name in {"web_search", "search"}:
            results = web_search_fn(
                query=str(arguments.get("query", "")),
                engine=engine,
                max_results=int(arguments.get("max_results") or 5),
            )
            return json.dumps({"results": results}, ensure_ascii=False)
        if name in {"fetch_url", "fetch"}:
            res = fetch_url_fn(
                url=str(arguments.get("url", "")),
                engine=engine,
                raw=bool(arguments.get("raw") or False),
                timeout=float(arguments.get("timeout") or 15.0),
            )
            return json.dumps(res, ensure_ascii=False)
        return json.dumps({"error": f"unknown tool: {name}"})
    except Exception as exc:  # pragma: no cover — defensive guard
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


__all__ = [
    # Builtin layer
    "Calculator",
    "CurrentTime",
    "Tool",
    "ToolRegistry",
    "UrlFetch",
    "builtin_registry",
    "sherlock_tool",
    # Web search engines
    "BraveSearch",
    "DuckDuckGoSearch",
    "SearchEngine",
    "StubSearch",
    "TavilySearch",
    "ValyuSearch",
    "build_role_engines",
    "build_search_engine",
    "create_search",
    # Callables + schemas for native tool-calling
    "dispatch_tool_call",
    "fetch_url_fn",
    "make_anthropic_tools",
    "make_openai_tools",
    "web_search_fn",
    # Memory tool (v0.4.0)
    "dispatch_memory",
    "make_anthropic_memory_tool",
    "make_openai_memory_tool",
    "memory_entity",
    "memory_lookup",
    "memory_pinned",
    "memory_timeline",
    "parse_memory_payload",
]
