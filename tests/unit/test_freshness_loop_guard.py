"""Waste guards on LLM-3's background freshness search loop.

A tiny / weak LLM-3 used to keep answering ``need_more=true`` round after
round, so a DEAD or EMPTY search engine (every round returns nothing, or only
``{"error": ...}`` payloads) made ``_run_inference_search_loop`` spin all the
way to the round ceiling — spamming ``infer.search.round`` events for no gain.

Two guards stop that, both pure waste-elimination (no result caps — a single
productive round resets the loop and it keeps going):

1. ``review_search`` forces ``need_more=false`` when a round produced no usable
   results (empty, or only engine error-payloads).
2. ``_run_inference_search_loop`` filters error-payloads out of ``hits`` at the
   source and breaks after two consecutive barren rounds even if LLM-3 insists.
"""

from __future__ import annotations

from sherlock import Sherlock


def _agent(tmp_path, name, *, infer=None):
    return Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        inference_chat=infer or (lambda m: "{}"),
        summary_chat=lambda m: "{}",
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        main_search_engine="stub",
        inference_search_engine=None,
        companions_mode="turbo",
        background=False,
    )


class _DeadEngine:
    """A search engine that always fails — returns an error-payload, never a hit."""

    def __init__(self):
        self.calls = 0

    def search(self, topic, max_results=4):
        self.calls += 1
        return [{"error": "engine down"}]


def test_freshness_loop_stops_fast_on_dead_engine(tmp_path):
    events: list[dict] = []
    a = _agent(tmp_path, "dead")
    a.set_event_sink(lambda ev: events.append(ev))

    eng = _DeadEngine()
    a._inference_search_engine = eng
    # A greedy / weak LLM-3 that ALWAYS wants another round with a fresh query —
    # left unchecked this would run to the 10-round ceiling on a dead engine.
    a._inferer.review_search = lambda **kw: {
        "need_more": True,
        "worth_saving": True,
        "next_queries": [f"q{kw['round_index']}"],
        "note": "",
    }

    search_results: list = []
    a._run_inference_search_loop(
        conv_id="c1",
        turn_index=1,
        hypotheses=[{"intent": "x", "probability": 0.9}],
        initial_queries=["latest world cup news"],
        search_results=search_results,
    )

    rounds = [e for e in events if e.get("type") == "infer.search.round"]
    assert len(rounds) <= 2, f"dead engine must stop fast, spun {len(rounds)} rounds"
    assert eng.calls <= 2, f"engine queried {eng.calls} times on a dead engine"
    assert search_results == []  # error-payloads never accumulate as results


def test_review_search_forces_stop_on_empty_or_error_results(tmp_path):
    # LLM-3 greedily asks for more even though there is nothing to build on.
    greedy = (
        '{"recent": false, "fleshes_out": false, "right_query": true, '
        '"worth_saving": true, "need_more": true, '
        '"next_queries": ["dig deeper", "more"], "note": "want more"}'
    )
    a = _agent(tmp_path, "rs", infer=lambda m: greedy)

    # Empty results → forced stop regardless of the model's greed.
    r_empty = a._inferer.review_search(
        topic="t", hypotheses=[], results=[], round_index=1, max_rounds=10
    )
    assert r_empty["need_more"] is False
    assert r_empty["next_queries"] == []

    # Only error-payloads → also a forced stop (no usable result this round).
    r_err = a._inferer.review_search(
        topic="t", hypotheses=[], results=[{"error": "boom"}], round_index=1, max_rounds=10
    )
    assert r_err["need_more"] is False
    assert r_err["next_queries"] == []

    # Real results present → the guard is surgical: the model's need_more stands.
    r_hit = a._inferer.review_search(
        topic="t",
        hypotheses=[],
        results=[{"title": "WC", "url": "u", "content": "a real finding"}],
        round_index=1,
        max_rounds=10,
    )
    assert r_hit["need_more"] is True
