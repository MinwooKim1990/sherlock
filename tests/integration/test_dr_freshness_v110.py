"""Phase 4 (v1.10): freshness. Source dates are captured by every engine and, when
`deep_research_freshness` is ON (default), surfaced in round snippets + the synthesis
raw block so the model can prefer the freshest source and flag stale-as-current.
OFF = dates captured but never surfaced (prompts byte-identical). Dates are OPAQUE
strings, never parsed/compared in code; nothing is filtered on date.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine, _extract_date


def test_extract_date_helper():
    assert _extract_date(
        '<meta property="article:published_time" content="2026-06-01T00:00Z">'
    ).startswith("2026-06-01")
    assert _extract_date("<html>no date here</html>") == ""
    assert _extract_date('<script>{"datePublished":"2025-12-09"}</script>') == "2025-12-09"


class _DateEngine(SearchEngine):
    def search(self, q, *, max_results=5):
        return [{"title": "t", "url": "https://e/1", "content": "snippet", "date": "2026-06-20"}]


def _main(prompts):
    def m(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
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

    return m


def _agent(main, tmp_path, fresh=True):
    a = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_DateEngine(),
        inference_search_engine="disabled",
    )
    a.config.search.deep_research_freshness = fresh
    a.config.search.deep_research_max_rounds = 2
    return a


def test_dates_surfaced_when_on(tmp_path):
    p: list[str] = []
    a = _agent(_main(p), tmp_path, fresh=True)
    a._run_deep_research(a._ensure_conversation().id, "topic", 1, "drF")
    assert any("[date: 2026-06-20]" in x for x in p), "source date not surfaced when ON"


def test_dates_hidden_when_off(tmp_path):
    p: list[str] = []
    a = _agent(_main(p), tmp_path, fresh=False)
    a._run_deep_research(a._ensure_conversation().id, "topic", 1, "drF")
    assert not any("[date:" in x for x in p), "date leaked into prompt when freshness OFF"
