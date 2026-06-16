"""v1.4 keystone: raw-fragment store + reconstruct-from-raw.

The live failure: Sherlock SEARCHED + FETCHED the real "Aomori Winter Wonderland"
page, but that round's terse extraction missed it, the raw fragment was DISCARDED,
and the facts-only synthesis could never recover it → final report said "no events
for Aomori". The fix keeps every round's raw fragments per sub-topic and re-reads
them at synthesis so the miss is recovered. These tests pin that, deterministically
(scripted engine + fake provider — no network, no real model).
"""

from __future__ import annotations

import json
import re

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine

AOMORI_EVENT = "Aomori Winter Wonderland — December 2026 at Aoiumi Park"
STRATEGY = json.dumps(
    {
        "objective": "도시별 겨울 행사",
        "sub_topics": ["Sapporo events", "Aomori events"],
        "scope": {"include": [], "exclude": []},
        "clarifying_questions": [],
    }
)


class _EventEngine(SearchEngine):
    """Every search surfaces the Aomori event snippet + a Sapporo snippet."""

    def search(self, query, *, max_results=5):
        return [
            {
                "title": "Aomori winter",
                "url": "https://ex.com/aomori",
                "content": AOMORI_EVENT + ", a winter light festival.",
            },
            {
                "title": "Sapporo info",
                "url": "https://ex.com/sapporo",
                "content": "Sapporo has a snow festival in February.",
            },
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": "page about regional events"}


def _make_main(n):
    """LLM-1 that UNDER-EXTRACTS: it never turns the Aomori snippet into a fact
    (only a Sapporo fact), and writes sections that mention the Aomori event ONLY
    when the prompt actually contains the raw fragment."""

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return STRATEGY
        if "Answer these meta-questions" in c:
            n["r"] += 1
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "Sapporo Snow Festival is in February",
                            "sources": ["https://ex.com/sapporo"],
                        }
                    ],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "ONE SECTION" in c:  # sectioned/raw-reconstruction synthesis
            if AOMORI_EVENT in c:
                return "## Aomori\n" + AOMORI_EVENT + " is the main winter event."
            return "## Section\nSapporo Snow Festival is in February."
        if "RESEARCH DOCUMENTS:" in c:  # facts-only single-call synthesis
            return "## Report\nSapporo Snow Festival is in February."
        return "plain."

    return main


def _agent(main, tmp_path):
    a = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_EventEngine(),
        inference_search_engine="disabled",
    )
    a.config.search.deep_research_max_rounds = 4
    return a


def test_round_missed_fact_recovered_at_synthesis(tmp_path):
    n = {"r": 0}
    events: list[tuple] = []
    agent = _agent(_make_main(n), tmp_path)
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    answer = agent._run_deep_research(agent._ensure_conversation().id, "winter events", 1, "drR")

    # The Aomori snippet was STORED as raw under its sub-topic (never discarded) ...
    state_round = next(d for (t, d) in events if t == "deep_research.round")
    assert state_round["raw_fragments_stored"] >= 1
    # ... and although NO round extracted it as a fact (only the Sapporo fact exists),
    # the final report RECOVERS it by re-reading the raw fragments.
    assert "Aomori Winter Wonderland" in answer, answer


def test_kill_switch_reverts_to_facts_only(tmp_path):
    """With reconstruct-from-raw OFF, the store is absent and the round-missed
    Aomori event does NOT appear — proving the recovery (not luck) surfaces it."""
    n = {"r": 0}
    events: list[tuple] = []
    agent = _agent(_make_main(n), tmp_path)
    agent.config.search.deep_research_reconstruct_from_raw = False
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    answer = agent._run_deep_research(agent._ensure_conversation().id, "winter events", 1, "drK")

    round_ev = next(d for (t, d) in events if t == "deep_research.round")
    assert round_ev["raw_fragments_stored"] == 0  # store disabled
    assert "Aomori Winter Wonderland" not in answer


class _SapporoOnlyEngine(SearchEngine):
    """Only ever surfaces Sapporo material — nothing routes to other cities."""

    def search(self, query, *, max_results=5):
        return [
            {
                "title": "Sapporo",
                "url": "https://ex.com/s",
                "content": "Sapporo has a snow festival in February.",
            }
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": "sapporo page"}


def test_requested_subtopic_never_silently_dropped(tmp_path):
    """A requested sub-topic with no facts AND no raw fragments still gets an
    honest section — it is never silently dropped from the report."""
    strat = json.dumps(
        {
            "objective": "events",
            "sub_topics": ["Sapporo events", "Akita events"],
            "scope": {"include": [], "exclude": []},
            "clarifying_questions": [],
        }
    )
    n = {"r": 0}

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return strat
        if "Answer these meta-questions" in c:
            n["r"] += 1
            return json.dumps(
                {
                    "facts": [
                        {
                            "fact": "Sapporo snow festival is in February",
                            "sources": ["https://ex.com/s"],
                        }
                    ],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "ONE SECTION" in c:  # echo the section name so we can assert presence
            m = re.search(r"Section: «([^»]+)»", c)
            return f"## {m.group(1) if m else '?'}\nsection body."
        if "RESEARCH DOCUMENTS:" in c:
            return "## Report\nbody."
        return "plain."

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_SapporoOnlyEngine(),
        inference_search_engine="disabled",
    )
    agent.config.search.deep_research_max_rounds = 3
    answer = agent._run_deep_research(agent._ensure_conversation().id, "events", 1, "drE")
    # Akita had no facts and no raw routed to it, yet its section is still present.
    assert "Akita events" in answer, answer
    assert "Sapporo events" in answer
