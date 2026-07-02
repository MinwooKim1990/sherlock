"""v1.11: deep-research per-round search parallelism.

A round's queries are independent, so they run concurrently on a dedicated pool
with results collected in QUERY ORDER — the answer, facts, and searched-query set
are byte-identical to the serial path; only wall-clock changes. OFF = exact serial
loop. These tests pin (a) result parity parallel-vs-serial and (b) that searches
really do overlap (not just claim to).
"""

from __future__ import annotations

import json
import threading
import time

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine

TOPICS = ["alpha carnival", "betazulu parade", "gamma gala"]
STRATEGY = json.dumps(
    {
        "objective": "o",
        "sub_topics": TOPICS,
        "scope": {"include": [], "exclude": []},
        "clarifying_questions": [],
    }
)


class _ProofEngine(SearchEngine):
    """Deterministic per-query results + records peak concurrency so a test can
    prove searches overlapped (max_concurrent >= 2) or ran serially (== 1)."""

    def __init__(self):
        self.calls: list[str] = []
        self._live = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def search(self, query, *, max_results=5):
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        time.sleep(0.05)  # widen the overlap window so peak concurrency is observable
        with self._lock:
            self._live -= 1
            self.calls.append(query)
        blob = "; ".join(TOPICS)
        return [
            {
                "title": query,
                "url": f"https://e/{abs(hash(query)) % 97}",
                "content": f"{blob} — {query}",
            }
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": f"page {url}"}


def _main(messages):
    c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    if "RESEARCH STRATEGY" in c:
        return STRATEGY
    if "Answer these meta-questions" in c:
        return json.dumps(
            {
                "facts": [
                    {"fact": "alpha carnival runs in May", "sources": ["https://e/1"]},
                    {"fact": "betazulu parade runs in June", "sources": ["https://e/2"]},
                    {"fact": "gamma gala runs in July", "sources": ["https://e/3"]},
                ],
                "key_finding": "k",
                "summary": "s",
                "gaps": [],
                "sufficient": True,
                "next_queries": [],
            }
        )
    if "FAITHFULNESS-checking" in c or "CONSISTENCY checker" in c:
        return json.dumps({"fixes": []})
    if "RESEARCH DOCUMENTS:" in c:
        return "FINAL"
    return "## Report\nalpha carnival; betazulu parade; gamma gala https://e/1"


def _build(tmp_path, parallel):
    eng = _ProofEngine()
    agent = Sherlock.with_callable(
        main_chat=_main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    agent.config.search.deep_research_parallel_search = parallel
    return agent, eng


def _run(tmp_path, parallel):
    agent, eng = _build(tmp_path, parallel)
    ans = agent._run_deep_research(agent._ensure_conversation().id, "festivals", 1, "p")
    return ans, eng


def test_parallel_and_serial_give_identical_results(tmp_path):
    ans_p, eng_p = _run(tmp_path / "par", True)
    ans_s, eng_s = _run(tmp_path / "ser", False)
    assert ans_p == ans_s, "parallel search must not change the produced report"
    # same set of queries searched (order within a round may differ under threads)
    assert sorted(eng_p.calls) == sorted(eng_s.calls) and eng_p.calls


def test_search_batch_overlaps_and_preserves_order(tmp_path):
    # directly exercise the batch helper (a full run's round-1 query count depends
    # on the planner, so prove concurrency at the primitive instead).
    agent, eng = _build(tmp_path, True)
    qs = ["qa", "qb", "qc", "qd"]
    out = agent._search_batch(eng.search, 5.0, qs, 4)
    assert eng.max_concurrent >= 2, f"searches should overlap, peak={eng.max_concurrent}"
    assert len(out) == len(qs) and all(ok for ok, _ in out)
    # results are aligned to INPUT order: each result echoes its own query's title
    for q, (_ok, res) in zip(qs, out):
        assert res and res[0]["title"] == q


def test_serial_path_runs_one_at_a_time(tmp_path):
    _ans, eng = _run(tmp_path, False)
    assert eng.max_concurrent == 1, f"OFF must be strictly serial, peak={eng.max_concurrent}"
