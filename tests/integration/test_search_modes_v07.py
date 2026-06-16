"""v0.7 — three search modes: param'd LLM-1 search (Phase 1) and the
LLM-3 background self-evaluating inference-search loop (Phase 2).

These OPERATE the features (assert the engine is actually driven with the
chosen counts / round counts), not just that nothing raised.
"""

from __future__ import annotations

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine


class CountingEngine(SearchEngine):
    """Fake search engine that records every (query, max_results) call and
    returns canned hits so the loop has something to persist."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        self.calls.append((query, max_results))
        return [
            {
                "title": f"{query} result {i}",
                "url": f"https://example.com/{i}",
                "content": f"snippet about {query} number {i}",
            }
            for i in range(max_results)
        ]


# --------------------------------------------------------------------------
# Phase 1 — LLM-1 may set k=N (clamped to config.search.max_results_cap)
# --------------------------------------------------------------------------


def _main_search_then_answer(tag: str):
    """Build a main callable that emits one search tool tag, then answers."""
    state = {"n": 0}

    def main(messages):
        state["n"] += 1
        if state["n"] == 1:
            return f"let me look that up.\n<<sherlock-tool: {tag}>>"
        return "here is the answer."

    return main


def test_llm1_search_honours_k(tmp_path):
    eng = CountingEngine()
    agent = Sherlock.with_callable(
        main_chat=_main_search_then_answer('search "samsung stock" k=8'),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    agent.chat("how is samsung doing")
    assert eng.calls, "search engine was never called"
    query, k = eng.calls[0]
    assert k == 8, f"expected k=8 to reach the engine, got {k}"
    assert "samsung" in query.lower()
    assert "k=" not in query, "the k= token leaked into the query string"


def test_llm1_search_clamps_k_to_cap(tmp_path):
    eng = CountingEngine()
    agent = Sherlock.with_callable(
        main_chat=_main_search_then_answer('search "nvidia" k=99'),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    # default cap is 10
    assert agent.config.search.max_results_cap == 10
    agent.chat("nvidia news")
    _, k = eng.calls[0]
    assert k == 10, f"k=99 should clamp to the cap (10), got {k}"


def test_llm1_search_default_is_five(tmp_path):
    eng = CountingEngine()
    agent = Sherlock.with_callable(
        main_chat=_main_search_then_answer('search "weather today"'),
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    agent.chat("weather?")
    _, k = eng.calls[0]
    assert k == 5, f"no k= should default to 5, got {k}"


# --------------------------------------------------------------------------
# Phase 2 — LLM-3 background self-evaluating inference-search loop (≤10)
# --------------------------------------------------------------------------

_INFER_JSON = (
    '{"hypotheses": ['
    '{"intent": "track samsung earnings", "probability": 0.7, "evidence": [],'
    ' "search_keywords": ["samsung"], "reasoning_type": "abduction"}'
    '], "tools_recommended": [], "context_to_expand": [], "context_to_exclude": [],'
    ' "freshness_required": ["samsung q3 earnings"], "confidence_overall": 0.7,'
    ' "evolution_signals": {}}'
)


def _review_json(need_more: bool, nxt: list[str]) -> str:
    import json

    return json.dumps(
        {
            "recent": True,
            "fleshes_out": True,
            "right_query": True,
            "worth_saving": True,
            "need_more": need_more,
            "next_queries": nxt,
            "note": "ok",
        }
    )


def _make_llm3(review_script: list[str]):
    """LLM-3 callable: first call (infer) returns hypotheses+freshness; each
    subsequent call (review_search) pops the next scripted review verdict."""
    state = {"reviews": list(review_script)}

    def llm3(messages):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = last.get("content", "")
        if "reviewing ONE round" in content:
            return state["reviews"].pop(0) if state["reviews"] else _review_json(False, [])
        return _INFER_JSON

    return llm3


def test_llm3_loop_stops_when_satisfied(tmp_path):
    """need_more twice, then stop → exactly 3 search rounds + 3 round events."""
    eng = CountingEngine()
    events: list[tuple[str, dict]] = []

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=_make_llm3(
            [
                _review_json(True, ["samsung q3 revenue detail"]),
                _review_json(True, ["samsung memory chip outlook"]),
                _review_json(False, []),
            ]
        ),
        system_prompt="…",
        storage_dir=tmp_path,
        inference_search_engine=eng,
        main_search_engine="disabled",
    )
    agent._turn_index = 10  # bypass cold-start
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat("hey")

    rounds = [d for (t, d) in events if t == "infer.search.round"]
    assert len(eng.calls) == 3, f"expected 3 search rounds, got {len(eng.calls)}: {eng.calls}"
    assert len(rounds) == 3, f"expected 3 infer.search.round events, got {len(rounds)}"
    # queries advanced via next_queries each round
    assert eng.calls[0][0] == "samsung q3 earnings"
    assert eng.calls[1][0] == "samsung q3 revenue detail"
    assert eng.calls[2][0] == "samsung memory chip outlook"
    # last round said stop
    assert rounds[-1]["need_more"] is False
    # results-per-round honoured (default 4)
    assert all(c[1] == agent.config.inference.search_results_per_round for c in eng.calls)


def test_llm3_loop_hard_caps_at_ten(tmp_path):
    """If LLM-3 always wants more, the loop is still capped at 10 rounds."""
    eng = CountingEngine()
    events: list[tuple[str, dict]] = []

    # Always need_more with a fresh (unique) query so dedup never short-circuits.
    def _make_greedy_llm3():
        state = {"i": 0}

        def llm3(messages):
            last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
            if "reviewing ONE round" in last.get("content", ""):
                state["i"] += 1
                return _review_json(True, [f"follow-up query {state['i']}"])
            return _INFER_JSON

        return llm3

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=_make_greedy_llm3(),
        system_prompt="…",
        storage_dir=tmp_path,
        inference_search_engine=eng,
        main_search_engine="disabled",
    )
    agent._turn_index = 10
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat("hey")

    rounds = [d for (t, d) in events if t == "infer.search.round"]
    assert len(eng.calls) == 10, f"hard ceiling is 10 rounds, got {len(eng.calls)}"
    assert len(rounds) == 10


def test_llm3_loop_persists_hits_as_memory(tmp_path):
    """Worthwhile hits are written as SEARCH_RESULT memories (carry-forward)."""
    eng = CountingEngine()
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=_make_llm3([_review_json(False, [])]),
        system_prompt="…",
        storage_dir=tmp_path,
        inference_search_engine=eng,
        main_search_engine="disabled",
    )
    agent._turn_index = 10
    agent.chat("hey")

    # One round ran and its hits were persisted.
    assert len(eng.calls) == 1
    conv_id = agent.conversation_id
    mems = agent._memory.list(conversation_id=conv_id)
    fresh = [m for m in mems if "freshness" in (m.tags or "")]
    assert fresh, "no freshness SEARCH_RESULT memories were persisted"
