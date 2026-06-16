"""v1.3 deep-research coverage gate.

A small LLM-1 tends to declare the WHOLE run "sufficient" after covering only
the FIRST part of a multi-part request (live: a 5-city Japan-events query →
answered Sapporo, stopped at round 2 with 4 facts, leaving Aomori/Akita/Morioka
as "no events"). The fix: gate the model's `sufficient` on every strategy
sub-topic having a supporting fact — an early `sufficient` with uncovered
sub-topics STEERS the next round at the gaps instead of stopping. The
convergence / stall stops stay the honest escape hatch when the missing pieces
aren't out there. These tests pin the gate.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from tests.integration.test_research_strategy_v10 import MiniEngine

SUB_TOPICS = ["삿포로 일루미네이션 행사", "도쿄 신년 행사"]
STRATEGY = json.dumps(
    {
        "objective": "도시별 겨울 행사 조사",
        "sub_topics": SUB_TOPICS,
        "scope": {"include": [], "exclude": []},
        "clarifying_questions": [],
    }
)


def _make_agent(main, tmp_path):
    return Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",  # lexical coverage path (no real embedder)
        background=False,
        main_search_engine=MiniEngine(),
        inference_search_engine="disabled",
    )


def test_early_sufficient_is_gated_until_subtopics_covered(tmp_path):
    """Round 1 covers only Sapporo and says sufficient=true. The gate must NOT
    stop — it steers round 2 at the uncovered Tokyo sub-topic; once both are
    covered it stops."""
    events: list[tuple] = []
    n = {"round": 0}

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return STRATEGY
        if "Answer these meta-questions" in c:
            n["round"] += 1
            if n["round"] == 1:
                # covers ONLY 삿포로 — but claims the whole run is done.
                fact = "삿포로 일루미네이션 행사는 12월 25일까지 열린다"
            else:
                # round 2+ covers 도쿄.
                fact = "도쿄 신년 행사로 메이지 신궁 하츠모데가 열린다"
            return json.dumps(
                {
                    "facts": [{"fact": fact, "sources": [f"https://ex.com/r{n['round']}"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,  # the small-model over-eager "done"
                    "next_queries": [],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL"
        return "plain."

    agent = _make_agent(main, tmp_path)
    agent.config.search.deep_research_max_rounds = 6
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._run_deep_research(agent._ensure_conversation().id, "겨울 행사", 1, "drCov")

    rounds = [d for (t, d) in events if t == "deep_research.round"]
    steers = [d for (t, d) in events if t == "deep_research.coverage_steer"]
    docs = next(d for (t, d) in events if t == "deep_research.documents")

    # The premature round-1 "sufficient" did NOT stop the run.
    assert len(rounds) >= 2, f"coverage gate must keep the loop past round 1: {len(rounds)} rounds"
    # The gate fired and named the uncovered Tokyo sub-topic.
    assert steers, "expected a deep_research.coverage_steer event"
    assert any("도쿄 신년 행사" in (s.get("uncovered") or []) for s in steers)
    assert steers[0]["total"] == 2 and steers[0]["covered"] == 1
    # Once BOTH sub-topics are covered, it stops honestly (model-sufficient).
    assert docs["stop_reason"] == "model_sufficient"
    # Both cities ended up in the fact base (nothing dropped).
    assert rounds[-1]["facts_total"] >= 2


def test_absence_fact_does_not_cover_its_subtopic(tmp_path):
    """A "no events reported for Tokyo" finding names the city but carries no
    real info — it must NOT mark that sub-topic covered, so the gate keeps
    steering at it until a REAL fact arrives."""
    events: list[tuple] = []
    n = {"round": 0}

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return STRATEGY
        if "Answer these meta-questions" in c:
            n["round"] += 1
            if n["round"] == 1:
                facts = [
                    {
                        "fact": "삿포로 일루미네이션 행사는 12월에 열린다",
                        "sources": ["https://ex.com/s"],
                    },
                    # absence finding for Tokyo — must NOT count as coverage
                    {
                        "fact": "도쿄 신년 행사에 대한 보고된 자료가 없습니다",
                        "sources": ["https://ex.com/t0"],
                    },
                ]
            else:
                facts = [
                    {"fact": "도쿄 신년 행사로 하츠모데가 열린다", "sources": ["https://ex.com/t1"]}
                ]
            return json.dumps(
                {
                    "facts": facts,
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL"
        return "plain."

    agent = _make_agent(main, tmp_path)
    agent.config.search.deep_research_max_rounds = 6
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._run_deep_research(agent._ensure_conversation().id, "겨울 행사", 1, "drAbs")

    steers = [d for (t, d) in events if t == "deep_research.coverage_steer"]
    rounds = [d for (t, d) in events if t == "deep_research.round"]
    # The Tokyo absence-fact in round 1 did NOT cover Tokyo → the gate steers at it.
    assert steers, "absence fact must not satisfy coverage; gate should steer"
    assert any("도쿄 신년 행사" in (s.get("uncovered") or []) for s in steers)
    assert len(rounds) >= 2


def test_no_subtopics_keeps_legacy_immediate_stop(tmp_path):
    """With no strategy sub-topics, an early `sufficient` stops immediately — the
    gate is inert (back-compat for every existing run/test)."""
    events: list[tuple] = []

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return "{}"  # no sub_topics → gate inert
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "facts": [{"fact": "the one fact", "sources": ["https://ex.com/1"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL"
        return "plain."

    agent = _make_agent(main, tmp_path)
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._run_deep_research(agent._ensure_conversation().id, "topic", 1, "drNo")

    rounds = [d for (t, d) in events if t == "deep_research.round"]
    steers = [d for (t, d) in events if t == "deep_research.coverage_steer"]
    docs = next(d for (t, d) in events if t == "deep_research.documents")
    assert len(rounds) == 1, "no sub-topics → first sufficient stops at round 1"
    assert not steers
    assert docs["stop_reason"] == "model_sufficient"
