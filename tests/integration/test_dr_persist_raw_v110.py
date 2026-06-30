"""Phase 6 (v1.10): opt-in SQL raw persistence for post-hoc recall.

With deep_research_persist_raw ON, a run's raw fragments are written to SQLite as a
pinned MemoryType.DEEP_RESEARCH_RAW doc (tagged by research_id) so they can be
retrieved/queried later. Default OFF (storage growth, not an accuracy feature) — no
such memory is written. Not used by the verify pass (raw is in-memory there).
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.memory.entry import MemoryType
from sherlock.tools.web_search import SearchEngine


class _E(SearchEngine):
    def search(self, q, *, max_results=5):
        return [{"title": "t", "url": "https://e/1", "content": "snippet detail here"}]


def _main(messages):
    c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
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
    return "## R\nbody https://e/1"


def _agent(tmp_path):
    a = Sherlock.with_callable(
        main_chat=_main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_E(),
        inference_search_engine="disabled",
    )
    a.config.search.deep_research_max_rounds = 2
    return a


def _count_raw(agent, conv):
    return sum(
        1
        for m in agent._memory.list(conversation_id=conv)
        if m.type == MemoryType.DEEP_RESEARCH_RAW
    )


def test_persist_raw_when_on(tmp_path):
    a = _agent(tmp_path)
    a.config.search.deep_research_persist_raw = True
    conv = a._ensure_conversation().id
    a._run_deep_research(conv, "topic", 1, "on")
    assert _count_raw(a, conv) >= 1, "raw not persisted when flag ON"


def test_no_persist_by_default(tmp_path):
    a = _agent(tmp_path)  # default OFF
    conv = a._ensure_conversation().id
    a._run_deep_research(conv, "topic", 1, "off")
    assert _count_raw(a, conv) == 0, "raw persisted despite flag OFF (default)"
