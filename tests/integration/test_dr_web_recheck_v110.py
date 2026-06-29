"""Phase 3 (v1.10): LLM-3 web re-check of flagged claims (opt-in "faithfulness+web").

Re-verifies ONLY the few claims the LLM-2 pass flagged as needing a fresh lookup:
one web search per claim (capped), then LLM-3 judges → confirmed | corrected |
unverifiable. corrected → verbatim replace; unverifiable → tag [unverified] (never
delete). Default is opt-in (deep_research_verify="faithfulness", not "+web").
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.config import SearchConfig
from sherlock.tools.web_search import SearchEngine


class _RecheckEngine(SearchEngine):
    def __init__(self):
        self.searches: list[str] = []

    def search(self, q, *, max_results=5):
        self.searches.append(q)
        return [
            {"title": "t", "url": "https://e/1", "content": "IVE Hong Kong Sep 4-6", "date": "2026"}
        ]


class _EmptyEngine(SearchEngine):
    def search(self, q, *, max_results=5):
        return []


def _inf(verdict, corrected=""):
    def m(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if "Re-verify ONE claim" in c:
            return json.dumps(
                {"verdict": verdict, "corrected_text": corrected, "source": "https://e/1"}
            )
        return '{"hypotheses": [], "freshness_required": [], "confidence_overall": 0.5}'

    return m


def _agent(inf_engine, inf_main, tmp_path):
    return Sherlock.with_callable(
        main_chat=lambda messages: "x",
        inference_chat=inf_main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_EmptyEngine(),
        inference_search_engine=inf_engine,
    )


def test_corrected_replaces_claim(tmp_path):
    a = _agent(_RecheckEngine(), _inf("corrected", "Sep 4-6"), tmp_path)
    out = a._web_recheck_flagged("IVE Hong Kong on Sep 11-13.", [{"claim": "Sep 11-13"}], "t", "r")
    assert "Sep 4-6" in out and "Sep 11-13" not in out


def test_unverifiable_tags_span(tmp_path):
    a = _agent(_RecheckEngine(), _inf("unverifiable"), tmp_path)
    out = a._web_recheck_flagged("X on Sep 11-13.", [{"claim": "Sep 11-13"}], "t", "r")
    assert "Sep 11-13 [unverified]" in out


def test_confirmed_leaves_claim(tmp_path):
    a = _agent(_RecheckEngine(), _inf("confirmed"), tmp_path)
    report = "IVE Hong Kong on Sep 11-13."
    out = a._web_recheck_flagged(report, [{"claim": "Sep 11-13"}], "t", "r")
    assert out == report


def test_cap_honored(tmp_path):
    eng = _RecheckEngine()
    a = _agent(eng, _inf("confirmed"), tmp_path)
    a.config.search.deep_research_web_recheck_max = 2
    flagged = [{"claim": f"c{i}"} for i in range(5)]
    a._web_recheck_flagged("c0 c1 c2 c3 c4", flagged, "t", "r")
    assert len(eng.searches) == 2, "web re-check cap not honored"


def test_no_hits_leaves_report(tmp_path):
    a = _agent(_EmptyEngine(), _inf("corrected", "Y"), tmp_path)
    report = "claim X here"
    out = a._web_recheck_flagged(report, [{"claim": "claim X"}], "t", "r")
    assert out == report


def test_web_recheck_is_opt_in_by_default():
    assert SearchConfig().deep_research_verify == "faithfulness"  # NOT "faithfulness+web"
