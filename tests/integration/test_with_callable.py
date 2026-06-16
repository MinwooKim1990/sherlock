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

    # v1.4: the user's words now ride at the end of the final user message, under
    # the "THE USER'S ACTUAL MESSAGE" boundary, so the echo contains them (no
    # longer immediately after "echo: ").
    assert "echo: " in reply_1 and reply_1.rstrip().endswith("hi")
    assert reply_2.rstrip().endswith("again")
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


def test_system_prompt_layering_default_appends(tmp_path):
    """User prompt stays primary; Sherlock extension appended after it."""
    seen: list[str] = []

    def my_llm(messages):
        # First message is the composed system prompt.
        if messages and messages[0].get("role") == "system":
            seen.append(messages[0]["content"])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="ROLE: pirate-themed help desk.",
        storage_dir=tmp_path,
    )
    agent.chat("hi")
    assert seen, "system prompt never reached the callable"
    composed = seen[0]
    # User prompt appears verbatim:
    assert "ROLE: pirate-themed help desk." in composed
    # Sherlock extension rides alongside:
    assert "SHERLOCK SYSTEM" in composed
    # And the user portion comes first (default position = "after"):
    assert composed.index("pirate") < composed.index("SHERLOCK SYSTEM")
    # Inspector still records the user-only and extension halves:
    assert "pirate" in agent._user_system_prompt
    assert "SHERLOCK SYSTEM" in agent._sherlock_extension


def test_system_prompt_layering_before_position(tmp_path):
    seen: list[str] = []

    def my_llm(messages):
        if messages and messages[0].get("role") == "system":
            seen.append(messages[0]["content"])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="USER: be terse.",
        storage_dir=tmp_path,
        extension_position="before",
    )
    agent.chat("hi")
    composed = seen[0]
    assert composed.index("SHERLOCK SYSTEM") < composed.index("USER: be terse.")


def test_system_prompt_layering_optout(tmp_path):
    seen: list[str] = []

    def my_llm(messages):
        if messages and messages[0].get("role") == "system":
            seen.append(messages[0]["content"])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="ONLY MINE.",
        storage_dir=tmp_path,
        sherlock_extension="",
    )
    agent.chat("hi")
    composed = seen[0]
    assert "ONLY MINE." in composed
    assert "SHERLOCK SYSTEM" not in composed


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
    assert "async-echo: " in reply and reply.rstrip().endswith("hi")
