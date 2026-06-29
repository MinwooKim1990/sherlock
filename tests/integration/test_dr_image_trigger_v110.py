"""Phase 5 (v1.10): image trigger on rich queries. og:image is captured only on a
page fetch, and fetches otherwise fire only on THIN rounds — so info-rich queries
never got an image. The harvest branch fetches the top hit ONCE per round (only when
nothing else was fetched) to grab its og:image. OFF = no harvest (byte-identical).
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine


class _RichEngine(SearchEngine):
    """4 long snippets → the round is RICH (not thin) → the thin-fetch never fires,
    so any fetch must be the image-harvest branch."""

    def __init__(self):
        self.fetches: list[str] = []

    def search(self, q, *, max_results=5):
        return [{"title": f"r{i}", "url": f"https://e/{i}", "content": "x" * 400} for i in range(4)]

    def fetch(self, url, *, raw=False, timeout=10.0):
        self.fetches.append(url)
        return {"url": url, "status": 200, "text": "page text", "image": "https://img/x.jpg"}


def _main(messages):
    c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    if "Answer these meta-questions" in c:
        return json.dumps(
            {
                "facts": [{"fact": "f", "sources": ["https://e/0"]}],
                "key_finding": "k",
                "summary": "s",
                "gaps": [],
                "sufficient": True,
                "next_queries": [],
            }
        )
    return "## R\nbody"


def _agent(eng, tmp_path, fetch_image=True):
    a = Sherlock.with_callable(
        main_chat=_main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=eng,
        inference_search_engine="disabled",
    )
    a.config.search.deep_research_fetch_image = fetch_image
    a.config.search.deep_research_max_rounds = 4
    return a


def test_image_harvest_on_rich_round_when_on(tmp_path):
    eng = _RichEngine()
    a = _agent(eng, tmp_path, fetch_image=True)
    a._run_deep_research(a._ensure_conversation().id, "topic", 1, "drON")
    assert len(eng.fetches) >= 1, "rich round did not harvest an image"


def test_no_harvest_when_off(tmp_path):
    eng = _RichEngine()
    a = _agent(eng, tmp_path, fetch_image=False)
    a._run_deep_research(a._ensure_conversation().id, "topic", 1, "drOFF")
    assert eng.fetches == [], "fetched on a rich round despite flag OFF"
