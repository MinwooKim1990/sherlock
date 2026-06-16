"""Regression tests for the v0.4.0 post-audit fixes (P0-1..P0-4, P1-3)."""

from __future__ import annotations

import json

from sherlock import Sherlock

# ── P0-1: timestamp must NOT sit in the stable TIER-1 prefix ─────────────


def test_timestamp_not_in_tier1_prefix(tmp_path):
    """v1.4: the system message is now FULLY stable (cacheable) — the volatile
    timestamp lives in the FINAL user message, never in the system prefix, and
    the whole system message is byte-identical across turns so caching survives.
    """
    systems: list[str] = []
    finals: list[str] = []

    def llm(messages):
        if messages and messages[0]["role"] == "system":
            systems.append(messages[0]["content"])
            finals.append(messages[-1]["content"])
        return "ok."

    agent = Sherlock.with_callable(main_chat=llm, system_prompt="ROLE: x", storage_dir=tmp_path)
    agent.chat("a")
    agent.chat("b")
    # Timestamp rides the volatile final user message, NOT the stable system msg.
    assert "CURRENT TIME" in finals[-1]
    assert "CURRENT TIME" not in systems[-1], "timestamp leaked into the stable system prefix"
    # With no pinned/persona yet the whole system message must be byte-identical
    # across turns — the cacheable prefix is stable.
    assert systems[0] == systems[1], "stable system prefix changed between turns"


# ── P0-2: a giant system prompt cannot overflow the context window ───────


def test_giant_system_prompt_does_not_overflow(tmp_path):
    from sherlock.budget import count_tokens

    sent: list[list[dict]] = []

    def llm(messages):
        sent.append([dict(m) for m in messages])
        return "ok."

    # 250k-word persona — far bigger than any real window.
    huge = "word " * 250_000
    agent = Sherlock.with_callable(main_chat=llm, system_prompt=huge, storage_dir=tmp_path)
    # Force a known small window so we can assert the ceiling.
    from sherlock.budget import select_profile_for_window, apply_overrides

    agent._ctx_window = 200_000
    agent._slot_budget = apply_overrides(select_profile_for_window(200_000), {})
    agent.chat("hello")
    # Total prompt tokens must stay under the window minus output reserve.
    last = sent[-1]
    total = sum(count_tokens(m["content"]) for m in last)
    assert total <= 200_000, f"prompt overflowed: {total} tokens"


# ── P0-3: tag-driven infer bypasses cold-start ───────────────────────────


def test_infer_tag_bypasses_cold_start(tmp_path):
    infer_calls = {"n": 0}

    def main(messages):
        return "ok.\n<<sherlock-companions: infer>>"

    def inference(messages):
        infer_calls["n"] += 1
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "intent": "h1",
                        "probability": 0.5,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "abduction",
                    },
                    {
                        "intent": "h2",
                        "probability": 0.3,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "deduction",
                    },
                    {
                        "intent": "h3",
                        "probability": 0.2,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "pragmatic",
                    },
                ],
                "tools_recommended": [],
                "context_to_expand": [],
                "context_to_exclude": [],
                "freshness_required": [],
                "confidence_overall": 0.5,
                "evolution_signals": {},
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    # turn_index stays at 2 (< cold_start_turns=10). Pre-fix this swallowed infer.
    agent._turn_index = 1  # so the next chat() is turn 2
    agent.chat("trigger infer on an early turn")
    assert infer_calls["n"] >= 1, "tag-driven infer was swallowed by cold-start"
    state = agent.inspect_last_turn()
    assert state.hypotheses, "no hypotheses produced despite infer tag"


# ── P0-4: provider-error responses are not persisted / not summarized ────


def test_error_response_not_persisted(tmp_path):
    def main(messages):
        return "[provider error 500: upstream exploded]"

    agent = Sherlock.with_callable(main_chat=main, system_prompt="x", storage_dir=tmp_path)
    reply = agent.chat("hello")
    # User still sees the error:
    assert "provider error" in reply
    # But it is NOT persisted as an assistant turn.
    roles = [m.role for m in agent.messages()]
    # Only system + user — no assistant error turn.
    assert "assistant" not in roles, f"error turn was persisted: {roles}"


def test_error_response_does_not_fire_companions(tmp_path):
    summary_calls = {"n": 0}

    def main(messages):
        # Even if (somehow) a tag rode along, an error must not trigger companions.
        return "[timeout — provider did not respond]\n<<sherlock-companions: compact>>"

    def summary(messages):
        summary_calls["n"] += 1
        return json.dumps(
            {
                "summary": "x",
                "facts": [],
                "topic_label": "t",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10
    agent.chat("hello")
    assert summary_calls["n"] == 0, "companion fired on an error turn"


# ── P1-3: persona summary replaces, doesn't accumulate ───────────────────


def test_persona_summary_does_not_accumulate(tmp_path):
    counter = {"n": 0}

    def main(messages):
        return "ok.\n<<sherlock-companions: compact>>"

    def summary(messages):
        counter["n"] += 1
        return json.dumps(
            {
                "summary": f"summary {counter['n']}",
                "facts": [],
                "topic_label": "t",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
                "persona_summary": f"persona version {counter['n']}",
                "predicted_directions": [],
                "worth_digging": [],
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10
    for _ in range(4):
        agent.chat("go")

    conv_id = agent.conversation_id
    entries = agent.memory.list(conversation_id=conv_id, pinned=True)
    personas = [e for e in entries if "persona_summary" in (e.tags or "")]
    assert len(personas) == 1, f"persona summaries accumulated: {len(personas)}"
    # And it's the LATEST version.
    assert "version 4" in personas[0].content
