"""v0.8 — internationalized search + token hygiene (operated end-to-end).

Scripted LLM-1/LLM-3 callables + fake engines drive the deep-research loop and
assert: a multilingual keyword PLAN drives the engine; per-call token
accounting is emitted; no snippet/page is fed to an LLM twice (cross-round
dedup); LLM-3's question-gen prompt never sees raw pages/URLs; every distinct
fact reaches the synthesis (no result loss); corroborated facts are tagged.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine


class RepeatEngine(SearchEngine):
    """Returns the SAME two URLs on every search (forces convergence + dedup)."""

    def __init__(self):
        self.calls = []

    def search(self, query, *, max_results=5):
        self.calls.append((query, max_results))
        return [
            {"title": "t1", "url": "https://ex.com/1", "content": "c1"},
            {"title": "t2", "url": "https://ex.com/2", "content": "c2"},
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "text": "page body"}


class ByQueryEngine(SearchEngine):
    """Returns a distinct URL per query (so new sources keep appearing)."""

    def __init__(self):
        self.calls = []

    def search(self, query, *, max_results=5):
        self.calls.append((query, max_results))
        h = abs(hash(query)) % 100000
        return [{"title": f"t{h}", "url": f"https://src{h}.com/a", "content": f"about {query}"}]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "text": "page body"}


def _make_main(log: list, *, sufficient_at=1, fact_text=None):
    state = {"n": 0}

    def main(messages):
        c = next((m for m in reversed(messages) if m.get("role") == "user"), {}).get("content", "")
        log.append(c)
        if "Answer these meta-questions" in c:
            state["n"] += 1
            n = state["n"]
            ft = fact_text or f"fact-{n}"
            return json.dumps(
                {
                    "facts": [{"fact": ft, "sources": [f"https://d{n}.com/a"]}],
                    "key_finding": f"k{n}",
                    "summary": "s",
                    "gaps": ["g"],
                    "sufficient": n >= sufficient_at,
                    "next_queries": [] if n >= sufficient_at else [f"more {n}"],
                }
            )
        if "RESEARCH DOCUMENTS:" in c:
            return "FINAL: synthesised."
        if "RESEARCHME" in c:
            return 'ok.\n<<sherlock-tool: deep_research "the topic">>'
        return "plain."

    return main


def _make_llm3(log: list):
    def llm3(messages):
        c = next((m for m in reversed(messages) if m.get("role") == "user"), {}).get("content", "")
        log.append(c)
        if "MULTILINGUAL web-search sweep" in c:
            return json.dumps(
                [
                    {"lang": "ja", "keywords": "日本 観光 穴場"},
                    {"lang": "ko", "keywords": "일본 여행 명소"},
                    {"lang": "en", "keywords": "japan hidden gems"},
                ]
            )
        if "META-COGNITION QUESTIONS" in c:
            return json.dumps(["deeper angle?", "broader angle?"])
        return '{"hypotheses": []}'

    return llm3


def _run(tmp_path, eng, main, llm3, msg="RESEARCHME the topic", **cfg):
    events = []
    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=llm3,
        system_prompt="…",
        storage_dir=tmp_path,
        main_search_engine=eng,
        inference_search_engine="disabled",
        deep_research_approver=lambda *_: True,
    )
    for k, v in cfg.items():
        setattr(agent.config.search, k, v)
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    agent.chat(msg)
    agent.wait_for_background(timeout=20)
    return agent, events


def test_multilingual_plan_drives_engine(tmp_path):
    eng = ByQueryEngine()
    agent, events = _run(tmp_path, eng, _make_main([], sufficient_at=1), _make_llm3([]))
    queries = [q for (q, _k) in eng.calls]
    # round-1 sweep issued the planned queries in 3 languages
    assert any("日本" in q for q in queries), queries
    assert any("일본" in q or "여행" in q for q in queries), queries
    assert any("japan" in q.lower() for q in queries), queries
    plan_ev = [d for (t, d) in events if t == "deep_research.plan"]
    assert plan_ev and len(plan_ev[0]["languages"]) >= 2


def test_token_measurement_emitted(tmp_path):
    agent, events = _run(tmp_path, ByQueryEngine(), _make_main([], sufficient_at=1), _make_llm3([]))
    tok = [d for (t, d) in events if t == "deep_research.tokens"]
    assert tok and tok[-1]["calls"] > 0 and tok[-1]["in"] > 0
    docs = [d for (t, d) in events if t == "deep_research.documents"]
    assert docs and docs[-1]["tokens"]["calls"] > 0
    # plan + meta_a stages are accounted separately
    assert "by_stage" in tok[-1]


def test_no_double_feed_and_llm3_sees_no_pages(tmp_path):
    main_log, llm3_log = [], []
    # never satisfied → runs until convergence (same URLs every round)
    agent, events = _run(
        tmp_path, RepeatEngine(), _make_main(main_log, sufficient_at=99), _make_llm3(llm3_log)
    )
    round_prompts = [p for p in main_log if "Answer these meta-questions" in p]
    assert len(round_prompts) >= 2
    # B4: each source URL is fed to LLM-1 in AT MOST ONE round (no re-feeding).
    for url in ("https://ex.com/1", "https://ex.com/2"):
        assert sum(url in p for p in round_prompts) <= 1, url
    # B1: LLM-3's question-gen prompt carries the compact state, never raw pages/URLs.
    meta_prompts = [p for p in llm3_log if "META-COGNITION QUESTIONS" in p]
    assert meta_prompts, "round ≥3 (LLM-3) never reached"
    assert all("http" not in p for p in meta_prompts)


def test_all_facts_reach_synthesis(tmp_path):
    main_log = []
    agent, events = _run(
        tmp_path, ByQueryEngine(), _make_main(main_log, sufficient_at=3), _make_llm3([])
    )
    synth = [p for p in main_log if "RESEARCH DOCUMENTS:" in p]
    assert synth, "synthesis never ran"
    # every distinct fact gathered (fact-1..fact-3) survives into the synthesis input
    for i in (1, 2, 3):
        assert f"fact-{i}" in synth[-1], f"fact-{i} lost before synthesis"


def test_triangulation_tags_corroborated(tmp_path):
    main_log = []
    # same fact text every round, different-domain source each round → corroborated
    agent, events = _run(
        tmp_path,
        ByQueryEngine(),
        _make_main(main_log, sufficient_at=3, fact_text="the shared fact"),
        _make_llm3([]),
    )
    synth = [p for p in main_log if "RESEARCH DOCUMENTS:" in p]
    assert synth and "[corroborated" in synth[-1], "corroborated fact not tagged in synthesis"
