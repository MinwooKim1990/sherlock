"""Phase 1 (v1.10): structured per-entity extraction.

When `deep_research_structured_extraction` is ON (default), each fact may carry an
`entity` + `attrs` so a bound attribute (a date) stays welded to ITS subject — the
direct fix for small-model entity-binding swaps (e.g. a city↔date mixup). OFF =
byte-identical legacy {"fact","sources"} schema (no `entity` in the prompt or facts).
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine

_STRUCT_FACT = json.dumps(
    {
        "facts": [
            {
                "fact": "IVE Hong Kong concert is Sep 4-6",
                "entity": "Hong Kong",
                "attrs": {"date": "Sep 4-6"},
                "sources": ["https://e/1"],
            }
        ],
        "key_finding": "k",
        "summary": "s",
        "gaps": [],
        "sufficient": True,
        "next_queries": [],
    }
)


class _E(SearchEngine):
    def search(self, q, *, max_results=5):
        return [{"title": "t", "url": "https://e/1", "content": "snippet"}]


def _main(prompts):
    def m(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
        if "Answer these meta-questions" in c:
            return _STRUCT_FACT
        return "x"

    return m


def _agent(main, tmp_path, structured=True):
    a = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_E(),
        inference_search_engine="disabled",
    )
    a.config.search.deep_research_structured_extraction = structured
    return a


def _call(agent):
    return agent._answer_research_round(
        "topic",
        {"confirmed_facts": [], "open_gaps": []},
        [{"title": "t", "url": "https://e/1", "content": "snippet"}],
        [],
        ["q1?"],
        [],
        1,
        4,
        "",
    )


def test_structured_schema_in_prompt_when_on(tmp_path):
    p: list[str] = []
    _call(_agent(_main(p), tmp_path, structured=True))
    assert any('"entity"' in x for x in p), "structured schema not offered to the model"


def test_legacy_schema_when_off(tmp_path):
    p: list[str] = []
    _call(_agent(_main(p), tmp_path, structured=False))
    assert not any('"entity"' in x for x in p), "entity schema leaked when flag OFF"


def test_entity_attrs_preserved_when_on(tmp_path):
    qa = _call(_agent(_main([]), tmp_path, structured=True))
    f = qa["facts"][0]
    assert f.get("entity") == "Hong Kong"
    assert f.get("attrs", {}).get("date") == "Sep 4-6"


def test_entity_attrs_dropped_when_off(tmp_path):
    qa = _call(_agent(_main([]), tmp_path, structured=False))
    f = qa["facts"][0]
    assert "entity" not in f and "attrs" not in f, "structured keys kept when flag OFF"
