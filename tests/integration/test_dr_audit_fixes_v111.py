"""v1.11 audit-fix regressions.

Two confirmed bugs from the v1.10 audit:
  * coverage-steer queries were appended AFTER the model's next_queries and then
    sliced off by the round's queries[:3] window — the steer event fired but the
    gap query never actually ran. Fix: steer leads.
  * the deep_research.tokens telemetry only fired per-round (before synthesis),
    so the editor + v1.10 verify chain (faithfulness/consistency/web_recheck)
    were never accounted. Fix: a FINAL tokens emit after the verify chain, and
    on_usage/_dr_account on each of those stages.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine


class _RoutingEngine(SearchEngine):
    """Records queries and returns content that carries the sub-topic tokens so a
    fragment routes into raw_fragments_by_subtopic (needed for faithfulness)."""

    def __init__(self, topics):
        self.calls: list[str] = []
        self._topics = topics

    def search(self, query, *, max_results=5):
        self.calls.append(query)
        # embed every sub-topic phrase so routing always finds a bucket
        blob = "; ".join(self._topics)
        return [
            {
                "title": query,
                "url": f"https://e/{abs(hash(query)) % 97}",
                "content": f"{blob} — {query}",
            }
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": f"page {url}"}


def _make_agent(main, eng, tmp_path):
    return Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )


def test_coverage_steer_actually_runs_next_round(tmp_path):
    """Round 1 covers only 'alpha carnival' but returns 3 next_queries and says
    sufficient. The uncovered 'betazulu parade' must be STEERED into — and, because
    steer now LEADS, it survives the round-2 queries[:3] window and is really
    searched (old code appended it at index 3+, so it was sliced off, never ran)."""
    topics = ["alpha carnival", "betazulu parade"]  # non-overlapping tokens
    eng = _RoutingEngine(topics)
    strategy = json.dumps(
        {
            "objective": "o",
            "sub_topics": topics,
            "scope": {"include": [], "exclude": []},
            "clarifying_questions": [],
        }
    )
    events: list[tuple] = []
    boundary: dict[str, int] = {}
    n = {"round": 0}

    def sink(ev):
        events.append((ev.get("type"), ev.get("data", {})))
        if ev.get("type") == "deep_research.round" and "r1" not in boundary:
            boundary["r1"] = len(eng.calls)  # calls up to and incl. round 1

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return strategy
        if "Answer these meta-questions" in c:
            n["round"] += 1
            if n["round"] == 1:
                return json.dumps(
                    {
                        "facts": [
                            {"fact": "alpha carnival runs in May", "sources": ["https://e/1"]}
                        ],
                        "key_finding": "k",
                        "summary": "s",
                        "gaps": [],
                        "sufficient": True,
                        "next_queries": ["gamma alpha q", "delta alpha q", "epsilon alpha q"],
                    }
                )
            return json.dumps(
                {
                    "facts": [{"fact": "betazulu parade runs in June", "sources": ["https://e/2"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL"
        return "## R\nbody"

    agent = _make_agent(main, eng, tmp_path)
    agent.config.search.deep_research_max_rounds = 6
    agent.set_event_sink(sink)
    agent._run_deep_research(agent._ensure_conversation().id, "festivals", 1, "steer")

    steers = [d for (t, d) in events if t == "deep_research.coverage_steer"]
    assert steers and any("betazulu parade" in (s.get("uncovered") or []) for s in steers)
    round2_calls = " ".join(eng.calls[boundary.get("r1", 0) :]).lower()
    assert "betazulu" in round2_calls, (
        "the steered gap query must actually run next round, not be sliced off by "
        f"queries[:3]; round2 searched: {eng.calls[boundary.get('r1', 0):]}"
    )


def test_verify_chain_tokens_are_accounted(tmp_path):
    """editor + faithfulness + consistency each add an LLM call after synthesis; a
    FINAL deep_research.tokens event must surface them in by_stage (they used to run
    entirely off the token books)."""
    topics = ["alpha carnival"]
    eng = _RoutingEngine(topics)
    events: list[tuple] = []

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "RESEARCH STRATEGY" in c:
            return json.dumps(
                {
                    "objective": "o",
                    "sub_topics": topics,
                    "scope": {"include": [], "exclude": []},
                    "clarifying_questions": [],
                }
            )
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "facts": [{"fact": "alpha carnival runs in May", "sources": ["https://e/1"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "FAITHFULNESS-checking" in c or "CONSISTENCY checker" in c:
            return json.dumps({"fixes": []})
        return "## Report\nalpha carnival body https://e/1"

    agent = _make_agent(main, eng, tmp_path)
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent._run_deep_research(agent._ensure_conversation().id, "festivals", 1, "tok")

    tok = [d for (t, d) in events if t == "deep_research.tokens"]
    assert tok, "no deep_research.tokens event"
    final = [d for d in tok if d.get("final")]
    assert final, "expected a FINAL deep_research.tokens emit after the verify chain"
    stages = final[-1].get("by_stage", {})
    for st in ("editor", "faithfulness", "consistency"):
        assert st in stages, f"{st} not accounted in final tokens by_stage: {sorted(stages)}"
