"""Integration: <<sherlock-tool: ...>> tag dispatch + 3-round cap.

Verifies that:
  * When LLM-1 emits a `<<sherlock-tool: search ...>>` tag, Sherlock
    executes it, injects results, and re-calls LLM-1 for a final reply.
  * `<<sherlock-tool: fetch URL>>` works the same way.
  * The 3-round-per-turn cap stops infinite tool loops.
  * Tag text is stripped from the user-visible reply.
"""

from __future__ import annotations

from dataclasses import dataclass

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine


@dataclass
class CannedEngine(SearchEngine):
    name: str = "canned"

    def search(self, query, *, max_results=5):
        return [
            {
                "title": f"result for {query}",
                "url": "https://example.com/1",
                "content": "snippet body",
                "source": "canned",
            }
        ]

    def fetch(self, url, *, raw=False, timeout=15.0):
        if raw:
            return {"url": url, "status": 200, "html": "<p>raw</p>"}
        return {"url": url, "status": 200, "text": "fetched body"}


def test_tool_tag_search_then_final(tmp_path):
    """LLM emits a search tag round 1, then a clean final reply round 2."""
    state = {"calls": 0}

    def my_llm(messages):
        state["calls"] += 1
        if state["calls"] == 1:
            return 'thinking aloud.\n<<sherlock-tool: search "python">>'
        return "Here is what I found: pythons exist."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="You are a research helper.",
        storage_dir=tmp_path,
        main_search_engine=CannedEngine(),
        inference_search_engine=None,
    )
    reply = agent.chat("tell me about pythons")
    # Tag is stripped:
    assert "<<sherlock-tool" not in reply
    assert "pythons exist" in reply
    # Round-trip happened (>= 2 LLM calls):
    assert state["calls"] >= 2


def test_tool_tag_fetch(tmp_path):
    state = {"calls": 0}

    def my_llm(messages):
        state["calls"] += 1
        if state["calls"] == 1:
            return "let me check.\n<<sherlock-tool: fetch https://example.com>>"
        return "fetched and ready."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="x",
        storage_dir=tmp_path,
        main_search_engine=CannedEngine(),
    )
    reply = agent.chat("look at the page")
    assert "fetched and ready" in reply
    assert state["calls"] >= 2


def test_tool_tag_three_round_cap(tmp_path):
    """LLM keeps emitting tags every turn — Sherlock must cap at 3 rounds
    and force-strip residual tags from the final visible response.
    """
    state = {"calls": 0}

    def greedy_llm(messages):
        state["calls"] += 1
        # Always request another search tag — agent must cap us off.
        return f"round {state['calls']}.\n<<sherlock-tool: search \"x{state['calls']}\">>"

    agent = Sherlock.with_callable(
        main_chat=greedy_llm,
        system_prompt="x",
        storage_dir=tmp_path,
        main_search_engine=CannedEngine(),
    )
    reply = agent.chat("go")
    # Tag is force-stripped even when the cap fired:
    assert "<<sherlock-tool" not in reply
    # The agent should have called the LLM at most cap+1 times
    # (initial round + up to 3 follow-ups).
    assert state["calls"] <= 4


def test_tool_tag_no_engine_returns_error_field(tmp_path):
    """If no search engine is wired, tool dispatch still works but the
    result block surfaces an error so the LLM can recover."""
    state = {"calls": 0}
    captured: list = []

    def my_llm(messages):
        state["calls"] += 1
        # Round 2: read the injected tool-results message.
        # CallableProvider hands the user a list of plain dicts.
        if state["calls"] == 2 and messages:
            captured.append(messages[-1]["content"])
        if state["calls"] == 1:
            return '<<sherlock-tool: search "anything">>'
        return "ok"

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="x",
        storage_dir=tmp_path,
        main_search_engine=None,
        inference_search_engine=None,
    )
    reply = agent.chat("...")
    assert "ok" in reply
    # The tool-results block surfaced an "error" line:
    assert captured and "error" in captured[0].lower()
