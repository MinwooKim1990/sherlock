"""v1.11 — deep-research output stays in the USER'S language end-to-end.

Bug (pre-v1.11, surfaced testing a Korean query): synthesis wrote the report in
Korean (it carries an explicit lang lock), but the v3 EDITOR pass then rewrote the
whole report in ENGLISH — its all-English prompt had NO language-preservation
instruction, so a small model translated it. The faithfulness/consistency/web
passes had the same gap in their replacement text. Fix: every post-synthesis LLM
pass is now told to keep the report's original language (+ a LANGUAGE line in the
shared _PRESENTATION_GUIDE). These tests assert the instruction reaches each pass.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.agent import _PRESENTATION_GUIDE
from sherlock.tools.web_search import SearchEngine


def test_presentation_guide_pins_language():
    # the shared guide (used by synthesis AND the editor) must carry a language rule
    assert "LANGUAGE" in _PRESENTATION_GUIDE
    assert "NEVER translate" in _PRESENTATION_GUIDE or "never translate" in _PRESENTATION_GUIDE


class _E(SearchEngine):
    def search(self, q, *, max_results=5):
        return [{"title": "제품", "url": "https://e/1", "content": "제품 상세 정보"}]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": f"page {url}"}


def test_every_post_synthesis_pass_gets_a_language_instruction(tmp_path):
    prompts: list[str] = []

    def main(messages):
        c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        prompts.append(c)
        if "RESEARCH STRATEGY" in c:
            return json.dumps(
                {
                    "objective": "o",
                    "sub_topics": ["제품"],
                    "scope": {"include": [], "exclude": []},
                    "clarifying_questions": [],
                }
            )
        if "Answer these meta-questions" in c:
            return json.dumps(
                {
                    "facts": [{"fact": "제품 상세 정보", "sources": ["https://e/1"]}],
                    "key_finding": "k",
                    "summary": "s",
                    "gaps": [],
                    "sufficient": True,
                    "next_queries": [],
                }
            )
        if "FAITHFULNESS-checking" in c or "CONSISTENCY checker" in c:
            return json.dumps({"fixes": []})
        return "## 보고서\n제품 상세 정보 https://e/1"

    agent = Sherlock.with_callable(
        main_chat=main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_E(),
        inference_search_engine="disabled",
    )
    # no summary provider → faithfulness/consistency fall back to main, so their
    # prompts are captured here too.
    agent._run_deep_research(agent._ensure_conversation().id, "제품 비교", 1, "lang")

    editor = [p for p in prompts if "fact-checking a research report" in p]
    faith = [p for p in prompts if "FAITHFULNESS-checking" in p]
    consist = [p for p in prompts if "CONSISTENCY checker" in p]
    synth = [p for p in prompts if "SAME language as" in p and "fact-checking" not in p]

    assert editor and any(
        "ORIGINAL LANGUAGE" in p for p in editor
    ), "editor pass must be told to keep the report's language"
    assert faith and any(
        "SAME LANGUAGE as the report" in p for p in faith
    ), "faithfulness `fix` must stay in the report's language"
    assert consist and any(
        "SAME LANGUAGE as the report" in p for p in consist
    ), "consistency `right` must stay in the report's language"
    assert synth, "synthesis retains its explicit lang lock"
