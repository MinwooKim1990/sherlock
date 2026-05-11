"""Sherlock.with_callable() — bring-your-own-LLM end-to-end smoke."""
from __future__ import annotations

import pytest

from sherlock import Sherlock


def test_with_callable_sync_echo(tmp_path):
    call_log: list[list[dict]] = []

    def my_llm(messages: list[dict]) -> str:
        call_log.append(messages)
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        return f"echo: {(last_user or {}).get('content', '')}"

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="You are a helpful test assistant.",
        storage_dir=tmp_path,
    )
    reply_1 = agent.chat("hi")
    reply_2 = agent.chat("again")

    assert "echo: hi" in reply_1
    assert "echo: again" in reply_2
    # Persistence
    msgs = agent.messages()
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    # LLM called at least once per user turn (companions may call additional)
    assert len(call_log) >= 2


def test_with_callable_strips_companions_tag(tmp_path):
    def my_llm(messages: list[dict]) -> str:
        return "the real answer.\n<<sherlock-companions: compact>>"

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="…",
        storage_dir=tmp_path,
    )
    reply = agent.chat("anything")
    assert reply.strip() == "the real answer."
    assert "<<sherlock-companions" not in reply


def test_with_callable_separate_companions(tmp_path):
    main_calls = []
    summary_calls = []
    inference_calls = []

    def main(messages):
        main_calls.append(messages)
        # Force LLM-1 to request both companions so each callable runs:
        return "ok.\n<<sherlock-companions: compact, infer>>"

    def summary(messages):
        summary_calls.append(messages)
        return '{"summary": "test", "facts": [], "topic_label": "t", "topic_changed_from_previous": false, "retrieval_keywords": []}'

    def inference(messages):
        inference_calls.append(messages)
        return (
            '{"hypotheses": ['
            '{"intent": "test1", "probability": 0.6, "evidence": [], "search_keywords": [], "reasoning_type": "deduction"},'
            '{"intent": "test2", "probability": 0.3, "evidence": [], "search_keywords": [], "reasoning_type": "abduction"},'
            '{"intent": "test3", "probability": 0.1, "evidence": [], "search_keywords": [], "reasoning_type": "pragmatic"}'
            '], "tools_recommended": [], "context_to_expand": [], "context_to_exclude": [],'
            ' "freshness_required": [], "confidence_overall": 0.6, "evolution_signals": {}}'
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        inference_chat=inference,
        system_prompt="…",
        storage_dir=tmp_path,
    )
    # Bypass cold-start by jumping to a later turn
    agent._turn_index = 10
    agent.chat("hi")

    assert len(main_calls) >= 1
    # Inferer fires when LLM-1 requests it AND cold-start clears (turn>=10):
    assert len(inference_calls) >= 1


@pytest.mark.asyncio
async def test_with_callable_async(tmp_path):
    """Async chat function should work both via sync .chat() and via .achat()."""

    async def my_llm(messages):
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        return f"async-echo: {(last_user or {}).get('content', '')}"

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="…",
        storage_dir=tmp_path,
    )
    reply = await agent.achat("hi")
    assert "async-echo: hi" in reply
