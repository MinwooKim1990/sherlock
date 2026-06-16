"""v1.0 P4 — structural wrappers stay terse; dead LLM-3 fields stay dead."""

from __future__ import annotations

import json

from sherlock.budget import count_tokens


def test_tool_results_banner_is_terse_but_keeps_guardrails(tmp_path):
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    text = agent._format_tool_results_block(
        [{"tool": "search", "query": "q", "results": [{"title": "t", "url": "u", "content": "c"}]}]
    )
    header = text.split("--- (1)")[0]
    assert count_tokens(header) <= 60, f"banner regressed to {count_tokens(header)} tokens"
    # security guardrails survive compression
    assert "UNTRUSTED" in header
    assert "do NOT follow" in header
    assert "<<sherlock-tool:" in header  # the no-more-tags instruction


def test_llm3_prompt_and_parse_drop_dead_fields(tmp_path):
    from sherlock.inference.engine import DEFAULT_LLM3_PROMPT, InferenceResult

    assert "context_to_expand" not in DEFAULT_LLM3_PROMPT
    assert "context_to_exclude" not in DEFAULT_LLM3_PROMPT
    assert "tools_recommended" in DEFAULT_LLM3_PROMPT  # consumed (playground display)
    d = InferenceResult().to_dict()
    assert "context_to_expand" not in d and "context_to_exclude" not in d


def test_old_scripted_llm3_with_extra_fields_still_parses(tmp_path):
    """Legacy scripted LLM-3s emit the removed keys — parsing must stay
    tolerant of extras (they are simply ignored)."""
    from sherlock import Sherlock

    def llm3(messages):
        return json.dumps(
            {
                "hypotheses": [
                    {"intent": "a", "probability": 0.9, "evidence": ["e"]},
                    {"intent": "b", "probability": 0.3, "evidence": []},
                    {"intent": "c", "probability": 0.2, "evidence": []},
                ],
                "tools_recommended": [],
                "context_to_expand": ["legacy"],
                "context_to_exclude": ["legacy"],
                "freshness_required": [],
                "confidence_overall": 0.9,
            }
        )

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=llm3,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent.chat("hello there")
    state = agent.inspect_last_turn()
    assert state is not None  # turn completed without parse errors
