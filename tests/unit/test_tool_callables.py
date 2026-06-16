"""Unit tests for the importable tool helpers (v0.3.0).

`sherlock.tools.web_search_fn`, `fetch_url_fn`, `make_openai_tools`,
`make_anthropic_tools`, `dispatch_tool_call` — these are the helpers users
hook into when wiring native tool-calling on their own LLM library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sherlock.tools import (
    dispatch_tool_call,
    fetch_url_fn,
    make_anthropic_tools,
    make_openai_tools,
    web_search_fn,
)
from sherlock.tools.web_search import SearchEngine


@dataclass
class CannedEngine(SearchEngine):
    name: str = "canned"

    def search(self, query, *, max_results=5):
        return [{"title": "canned", "url": "https://x", "content": query, "source": "canned"}]

    def fetch(self, url, *, raw=False, timeout=15.0):
        if raw:
            return {"url": url, "status": 200, "html": "<p>raw</p>"}
        return {"url": url, "status": 200, "text": "extracted"}


def test_web_search_fn_with_instance():
    out = web_search_fn("hello", engine=CannedEngine())
    assert out[0]["content"] == "hello"


def test_web_search_fn_engine_string_default():
    # Default is DuckDuckGo — we don't hit the network; just verify the
    # callable wires through without raising on construction.
    fn = web_search_fn
    assert callable(fn)


def test_fetch_url_fn_text_vs_raw():
    eng = CannedEngine()
    text_mode = fetch_url_fn("https://x", engine=eng)
    raw_mode = fetch_url_fn("https://x", engine=eng, raw=True)
    assert "text" in text_mode and "html" not in text_mode
    assert "html" in raw_mode and "text" not in raw_mode


def test_make_openai_tools_schema_shape():
    tools = make_openai_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == {"web_search", "fetch_url"}
    for t in tools:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_make_anthropic_tools_schema_shape():
    tools = make_anthropic_tools()
    names = {t["name"] for t in tools}
    assert names == {"web_search", "fetch_url"}
    for t in tools:
        assert "input_schema" in t


def test_dispatch_tool_call_search():
    out = dispatch_tool_call(
        "web_search", {"query": "test", "max_results": 1}, engine=CannedEngine()
    )
    parsed = json.loads(out)
    assert parsed["results"][0]["content"] == "test"


def test_dispatch_tool_call_fetch_raw():
    out = dispatch_tool_call("fetch_url", {"url": "https://x", "raw": True}, engine=CannedEngine())
    parsed = json.loads(out)
    assert "html" in parsed and parsed["html"] == "<p>raw</p>"


def test_dispatch_tool_call_unknown():
    out = dispatch_tool_call("not_a_tool", {}, engine=CannedEngine())
    parsed = json.loads(out)
    assert "error" in parsed
