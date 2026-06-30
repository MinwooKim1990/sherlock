"""Phase 2b (v1.10): FINAL whole-report consistency sweep (LLM-2 librarian).

The per-group faithfulness pass sees ONE sub-topic's raw at a time, so a fact stated
two ways in two SECTIONS (a date as Sep 4-5 here / Sep 4-6 there, a tour name two ways,
an event in the summary table but not the detail) survives it. This single whole-report
pass reconciles each INTERNAL contradiction to one best-supported value — factual
consistency (사실의 통일성) ONLY, never format/length/structure. Non-destructive (verbatim
span rewrite, never delete), shrink-guarded. Runs whenever verify != "off"; OFF = skip.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine

_STRATEGY = json.dumps(
    {
        "objective": "o",
        "sub_topics": ["Alpha events"],
        "scope": {"include": [], "exclude": []},
        "clarifying_questions": [],
    }
)


class _E(SearchEngine):
    def search(self, q, *, max_results=5):
        return [{"title": "Alpha events", "url": "https://e/1", "content": "Alpha events detail"}]


def _agent(main, tmp_path):
    # no summary_chat → _summary_provider is None → consistency falls back to _provider (main)
    return Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_E(),
        inference_search_engine="disabled",
    )


def _fake(fixes):
    def m(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "CONSISTENCY checker" in c:
            return json.dumps({"fixes": fixes})
        return "x"

    return m


# -------------------------------------------------------------------- direct
def test_reconciles_internal_contradiction(tmp_path):
    report = "Summary: IVE plays Hong Kong Sep 4-5.\nDetail: the Hong Kong show runs Sep 4-6."
    a = _agent(
        _fake([{"issue": "date disagrees", "wrong": "Sep 4-5", "right": "Sep 4-6"}]), tmp_path
    )
    out = a._reconcile_report_consistency(report, "t", "r")
    assert "Sep 4-5" not in out and out.count("Sep 4-6") == 2


def test_phantom_span_rejected(tmp_path):
    report = "IVE plays Hong Kong on Sep 4-6."
    a = _agent(_fake([{"issue": "x", "wrong": "NONEXISTENT", "right": "Y"}]), tmp_path)
    out = a._reconcile_report_consistency(report, "t", "r")
    assert out == report, "phantom-span reconciliation must not be applied"


def test_noop_fix_ignored(tmp_path):
    report = "IVE plays Hong Kong on Sep 4-6."
    a = _agent(_fake([{"issue": "x", "wrong": "Sep 4-6", "right": "Sep 4-6"}]), tmp_path)
    out = a._reconcile_report_consistency(report, "t", "r")
    assert out == report


def test_shrink_guard_reverts(tmp_path):
    report = "KEEPER " + ("Z" * 100)
    # a runaway "reconciliation" that would gut the report must trip the 0.3 guard
    a = _agent(_fake([{"issue": "x", "wrong": "Z" * 100, "right": "z"}]), tmp_path)
    out = a._reconcile_report_consistency(report, "t", "r")
    assert out == report, "gutting the report should trip the 0.3 shrink guard"


def test_blank_report_is_noop(tmp_path):
    a = _agent(_fake([{"issue": "x", "wrong": "a", "right": "b"}]), tmp_path)
    assert a._reconcile_report_consistency("   ", "t", "r") == "   "


# -------------------------------------------------------------------- e2e gate
def _emain(prompts):
    def m(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
        if "RESEARCH STRATEGY" in c:
            return _STRATEGY
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "facts": [{"fact": "f", "sources": ["https://e/1"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "FAITHFULNESS-checking" in c or "CONSISTENCY checker" in c:
            return json.dumps({"fixes": []})
        return "## R\nAlpha events body https://e/1"

    return m


def test_consistency_runs_by_default_and_skips_when_off(tmp_path):
    p1: list[str] = []
    a1 = _agent(_emain(p1), tmp_path)
    a1._run_deep_research(a1._ensure_conversation().id, "topic", 1, "on")
    assert any("CONSISTENCY checker" in x for x in p1), "consistency sweep did not run by default"

    p2: list[str] = []
    a2 = _agent(_emain(p2), tmp_path)
    a2.config.search.deep_research_verify = "off"
    a2._run_deep_research(a2._ensure_conversation().id, "topic", 1, "off")
    assert not any("CONSISTENCY checker" in x for x in p2), "consistency sweep ran when OFF"
