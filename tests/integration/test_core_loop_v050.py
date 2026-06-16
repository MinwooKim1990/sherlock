"""v0.5.0 Phase 1 — core curation loop regression tests."""

from __future__ import annotations

import json

import pytest

from sherlock import Sherlock


def _infer_json(intent="user really wants reassurance"):
    return json.dumps(
        {
            "hypotheses": [
                {
                    "intent": intent,
                    "probability": 0.72,
                    "evidence": ["tone"],
                    "search_keywords": [],
                    "reasoning_type": "abduction",
                },
                {
                    "intent": "surface ask",
                    "probability": 0.2,
                    "evidence": [],
                    "search_keywords": [],
                    "reasoning_type": "deduction",
                },
                {
                    "intent": "alt",
                    "probability": 0.08,
                    "evidence": [],
                    "search_keywords": [],
                    "reasoning_type": "pragmatic",
                },
            ],
            "tools_recommended": [],
            "context_to_expand": [],
            "context_to_exclude": [],
            "freshness_required": [],
            "confidence_overall": 0.72,
            "evolution_signals": {},
        }
    )


def test_hypotheses_carry_forward_to_next_turn_slot(tmp_path):
    """LLM-3 runs post-response on turn N; its top hypothesis must appear in
    the TIER-3 active-intent slot on turn N+1.
    """
    systems: list[str] = []
    calls = {"n": 0}

    def main(messages):
        calls["n"] += 1
        # Only capture LLM-1 slots (they carry the TIER header). The
        # summary companion also routes through main_chat here.
        if messages and messages[0]["role"] == "system" and "TIER 1" in messages[0]["content"]:
            # v1.4: inference/active-intent now rides the FINAL user message, not
            # the system message — capture that (the system TIER-1 gate identifies
            # an LLM-1 call).
            systems.append(messages[-1]["content"])
        if calls["n"] == 1:
            return "first reply.\n<<sherlock-companions: infer>>"
        return "second reply."

    def inference(messages):
        return _infer_json(intent="UNIQUE_INTENT_MARKER")

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10  # clear cold-start so infer runs
    agent.chat("turn one")  # LLM-3 fires post-response → pending set
    agent.chat("turn two")  # slot should now carry the hypothesis

    # The 2nd turn's system prompt must contain the carried hypothesis.
    assert any(
        "UNIQUE_INTENT_MARKER" in s for s in systems[1:]
    ), "prior-turn hypothesis did not carry into the next slot"
    assert any("INFERENCE HYPOTHESES" in s for s in systems[1:])


def test_pending_consumed_once(tmp_path):
    """Pending context is consumed on the next turn and not re-shown twice."""
    systems: list[str] = []
    calls = {"n": 0}

    def main(messages):
        calls["n"] += 1
        # Capture only LLM-1 slots (TIER header); summary routes via main too.
        is_llm1 = bool(
            messages and messages[0]["role"] == "system" and "TIER 1" in messages[0]["content"]
        )
        if is_llm1:
            # v1.4: inference/active-intent now rides the FINAL user message, not
            # the system message — capture that (the system TIER-1 gate identifies
            # an LLM-1 call).
            systems.append(messages[-1]["content"])
        # Only the FIRST LLM-1 turn emits infer; later LLM-1 turns are plain.
        if is_llm1 and len([s for s in systems]) == 1:
            return "ok.\n<<sherlock-companions: infer>>"
        return "ok."

    def inference(messages):
        return _infer_json(intent="CARRY_ONCE_MARKER")

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10
    agent.chat("a")  # infer fires → pending
    agent.chat("b")  # consumes pending (active-intent block appears)
    agent.chat("c")  # pending already cleared (no active-intent block)

    def active_intent_block(sys_msg: str) -> str:
        # The dedicated active-intent block lives under this header.
        if "INFERENCE HYPOTHESES" not in sys_msg:
            return ""
        return sys_msg.split("INFERENCE HYPOTHESES", 1)[1].split("═══", 1)[0]

    # Turn 2: the carried hypothesis populates the active-intent block.
    assert "CARRY_ONCE_MARKER" in active_intent_block(
        systems[1]
    ), "turn 2 active-intent block should carry the hypothesis"
    # Turn 3: pending was consumed → no active-intent block carrying it.
    # (It may still surface via RAG retrieval, which is a separate block.)
    assert "CARRY_ONCE_MARKER" not in active_intent_block(
        systems[2]
    ), "turn 3 must not re-carry the consumed hypothesis in the active-intent slot"


def test_self_retrieval_excluded(tmp_path):
    """The current user input must not be retrieved as its own RAG match,
    and recent USER_UTTERANCEs (in the tail) shouldn't flood RAG.
    """

    def main(messages):
        return "ok."

    agent = Sherlock.with_callable(main_chat=main, system_prompt="x", storage_dir=tmp_path)
    agent.chat("my daughter has a peanut allergy")
    agent.chat("my daughter has a peanut allergy")  # same text again
    state = agent.inspect_last_turn()
    # The current-turn USER_UTTERANCE must not be among retrieved memories.
    for entry, _ in state.retrieved_memories:
        # No retrieved USER_UTTERANCE should be from the current turn.
        from sherlock.memory.entry import MemoryType

        if entry.type == MemoryType.USER_UTTERANCE:
            assert entry.last_used_turn_index < agent._turn_index - 2


@pytest.mark.asyncio
async def test_achat_strips_tags(tmp_path):
    """achat() must return tag-stripped text (no <<sherlock-companions>> leak)."""

    async def main(messages):
        return "the answer.\n<<sherlock-companions: compact, infer>>"

    agent = Sherlock.with_callable(main_chat=main, system_prompt="x", storage_dir=tmp_path)
    reply = await agent.achat("hi")
    assert "<<sherlock-companions" not in reply
    assert reply.strip() == "the answer."


def test_wrapper_error_not_persisted(tmp_path):
    def main(messages):
        return "[wrapper-error: RuntimeError: boom]"

    agent = Sherlock.with_callable(main_chat=main, system_prompt="x", storage_dir=tmp_path)
    reply = agent.chat("hello")
    assert "wrapper-error" in reply
    assert "assistant" not in [m.role for m in agent.messages()]


def test_current_user_input_not_duplicated_in_slot(tmp_path):
    """Regression (v0.6): the current turn's user message is persisted before
    slot assembly (crash-safety) but appended separately — without excluding it
    from the K-turn tail it appeared TWICE in every LLM-1 prompt.
    """
    captured = []

    def main(messages):
        captured.append([m["content"] for m in messages if m["role"] == "user"])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=main, system_prompt="x", storage_dir=tmp_path, background=False
    )
    agent.chat("alpha unique phrase")
    assert sum(c.count("alpha unique phrase") for c in captured[-1]) == 1, captured[-1]
    agent.chat("beta unique phrase")
    # prior turn (alpha) is legit history; current (beta) must appear once.
    assert sum(c.count("beta unique phrase") for c in captured[-1]) == 1, captured[-1]
    assert sum(c.count("alpha unique phrase") for c in captured[-1]) == 1, captured[-1]


async def test_achat_current_input_not_duplicated(tmp_path):
    """Same no-duplication guarantee on the async path (achat parity)."""
    captured = []

    async def main(messages):
        captured.append([m["content"] for m in messages if m["role"] == "user"])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=main, system_prompt="x", storage_dir=tmp_path, background=False
    )
    await agent.achat("gamma unique phrase")
    assert sum(c.count("gamma unique phrase") for c in captured[-1]) == 1, captured[-1]


def test_auto_infer_fires_without_tag(tmp_path, monkeypatch):
    """Regression (v0.6): with auto_infer='smart', LLM-3 fires on a topic shift
    even when LLM-1 never emits the <<...: infer>> tag (no longer dormant).
    """
    monkeypatch.setenv("SHERLOCK_AUTO_INFER", "smart")  # opt in (suite default is off)
    events = []

    def main(messages):
        return "plain reply, no tag whatsoever."

    def inference(messages):
        return _infer_json(intent="AUTO_INFER_MARKER")

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
        background=False,
    )
    agent.set_event_sink(lambda e: events.append(e["type"]))
    agent.chat("hi there")  # turn 1 → cold-start auto-infer
    agent.chat("a completely different subject")  # topic shift → auto-infer
    assert events.count("infer.done") >= 1, "auto-infer should fire LLM-3 without the tag"

    # And with auto_infer='off' it must NOT fire (pure tag-driven).
    monkeypatch.setenv("SHERLOCK_AUTO_INFER", "off")
    ev2 = []
    agent2 = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path / "b",
        background=False,
    )
    agent2.set_event_sink(lambda e: ev2.append(e["type"]))
    agent2.chat("hello")
    agent2.chat("unrelated thing")
    assert ev2.count("infer.done") == 0, "auto_infer=off must stay purely tag-driven"
