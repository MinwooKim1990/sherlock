"""Phase 2 (v1.10): LLM-2 faithfulness verify — the accuracy core (riskiest phase).

A SEPARATE cross-model pass re-reads the report against the gathered RAW (per
sub-topic) and fixes mis-extractions (report says X, raw says Y) + contradictions
the same-model v3 editor misses. Fences against corrupting a correct report:
verbatim-span match only (phantom spans rejected), capped fixes, 0.3 shrink guard.
OFF = skip (byte-identical). Checks RAW, never the facts (that would be circular).
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
    # no summary_chat → _summary_provider is None → faithfulness falls back to _provider (main)
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
        if "FAITHFULNESS-checking" in c:
            return json.dumps({"fixes": fixes})
        return "x"

    return m


_RAW = {"raw_fragments_by_subtopic": {"HK": [{"url": "https://e/1", "text": "raw", "date": ""}]}}


# -------------------------------------------------------------------- direct
def test_misextraction_fix_applied(tmp_path):
    a = _agent(
        _fake(
            [{"claim": "Sep 11-13", "issue": "misextraction", "fix": "Sep 4-6", "needs_web": False}]
        ),
        tmp_path,
    )
    out, _flag = a._verify_report_faithfulness("IVE plays Hong Kong on Sep 11-13.", _RAW, "t", "r")
    assert "Sep 4-6" in out and "Sep 11-13" not in out


def test_phantom_span_rejected(tmp_path):
    report = "IVE plays Hong Kong on Sep 11-13."
    a = _agent(_fake([{"claim": "NONEXISTENT", "issue": "misextraction", "fix": "X"}]), tmp_path)
    out, _flag = a._verify_report_faithfulness(report, _RAW, "t", "r")
    assert out == report, "phantom-span fix must not be applied"


def test_empty_raw_is_noop(tmp_path):
    report = "something specific"
    a = _agent(_fake([{"claim": "something", "fix": "else"}]), tmp_path)
    out, flag = a._verify_report_faithfulness(report, {"raw_fragments_by_subtopic": {}}, "t", "r")
    assert out == report and flag == []


def test_shrink_guard_reverts(tmp_path):
    report = "KEEPER " + ("Z" * 100)
    a = _agent(_fake([{"claim": "Z" * 100, "issue": "unsupported", "fix": "remove"}]), tmp_path)
    out, _flag = a._verify_report_faithfulness(report, _RAW, "t", "r")
    assert out == report, "removing most of the report should trip the 0.3 shrink guard"


def test_needs_web_collected(tmp_path):
    a = _agent(
        _fake([{"claim": "Sep 11-13", "issue": "unsupported", "fix": "", "needs_web": True}]),
        tmp_path,
    )
    _out, flag = a._verify_report_faithfulness("on Sep 11-13.", _RAW, "t", "r")
    assert len(flag) == 1 and flag[0]["claim"] == "Sep 11-13"


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
        if "FAITHFULNESS-checking" in c:
            return json.dumps({"fixes": []})
        return "## R\nAlpha events body https://e/1"

    return m


def test_faithfulness_runs_by_default_and_skips_when_off(tmp_path):
    p1: list[str] = []
    a1 = _agent(_emain(p1), tmp_path)
    a1._run_deep_research(a1._ensure_conversation().id, "topic", 1, "on")
    assert any("FAITHFULNESS-checking" in x for x in p1), "faithfulness did not run by default"

    p2: list[str] = []
    a2 = _agent(_emain(p2), tmp_path)
    a2.config.search.deep_research_verify = "off"
    a2._run_deep_research(a2._ensure_conversation().id, "topic", 1, "off")
    assert not any("FAITHFULNESS-checking" in x for x in p2), "faithfulness ran when OFF"
