"""TIER-labelled slot layout + dynamic K-turn walk-backward (v0.4.0)."""

from __future__ import annotations

from sherlock import Sherlock


def test_tier_labels_appear_in_order(tmp_path):
    """The composed system prompt must contain TIER 1/2/4 headers in
    increasing order. TIER 3 only appears when speculative content is
    present.
    """
    captured: list[str] = []

    def my_llm(messages):
        if messages and messages[0].get("role") == "system":
            captured.append(messages[0]["content"])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
    )
    agent.chat("hi")
    assert captured, "system message never reached the callable"
    sys_msg = captured[0]
    assert "TIER 1: GROUND TRUTH" in sys_msg
    assert "TIER 4: ACTIVE CONTEXT" in sys_msg
    # Ordering check:
    assert sys_msg.index("TIER 1") < sys_msg.index("TIER 4")


def test_k_turn_dynamic_walks_backward(tmp_path):
    """Dynamic K-turn picks as many *whole* turns as budget allows.

    With default (Haiku-200K-equivalent) budget and short turns, all 4
    user turns should make it into the tail.
    """
    sent: list[list[dict]] = []

    def my_llm(messages):
        sent.append([dict(m) for m in messages])
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    for u in ["alpha", "beta", "gamma", "delta"]:
        agent.chat(u)
    # On the 4th turn, the K-turn tail should include the prior 3 user
    # turns + 3 assistant replies → 6 non-system messages before the
    # current input. Default profile + Haiku window has huge budget.
    last_call = sent[-1]
    non_sys_count = sum(1 for m in last_call if m["role"] != "system")
    # ≥ 4 non-system messages (we expect 7: 3 prior pairs + current input).
    assert non_sys_count >= 4


def test_inspect_last_turn_exposes_budget(tmp_path):
    def my_llm(messages):
        return "ok."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("test")
    state = agent.inspect_last_turn()
    assert state is not None
    assert state.slot_budget  # dict populated
    assert "compacted_memory_max" in state.slot_budget
    assert state.k_turn_turns_used >= 0
    assert state.k_turn_tokens_used >= 0


def test_anticipated_block_surfaces_predictions(tmp_path):
    """v0.6: RAG-matched LLM-2 predictions / worth-digging threads are split out
    of the generic retrieval block into a dedicated ANTICIPATED DIRECTIONS block
    (proactive topic-shift adaptation)."""
    from sherlock.memory.entry import MemorySource, MemoryType

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok", system_prompt="x", storage_dir=tmp_path
    )
    conv = agent._ensure_conversation().id
    pred = agent.memory.add(
        conversation_id=conv,
        content="user will likely ask about hotels next",
        type=MemoryType.INFERENCE,
        source=MemorySource.LLM_2_PREDICTION,
        confidence=0.7,
        tags="prediction",
    )
    fact = agent.memory.add(
        conversation_id=conv,
        content="user prefers Vue",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
    )
    block, regular = agent._format_anticipated_block([(pred, 0.91), (fact, 0.80)])
    assert "ANTICIPATED DIRECTIONS" in block and "hotels" in block
    # the prediction is pulled out; only the regular fact remains for RAG block
    assert [e.id for e, _ in regular] == [fact.id]


def test_k_turn_never_splits_mid_message(tmp_path):
    """Walk-backward must take whole messages or none — never partial."""
    sent: list[list[dict]] = []

    def my_llm(messages):
        sent.append([dict(m) for m in messages])
        return "x"

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("hello")
    agent.chat("world")
    # All forwarded user/assistant messages should match a stored msg.
    msgs = {m.content for m in agent.messages()}
    for call in sent:
        for m in call:
            if m["role"] in {"user", "assistant"}:
                # The agent may inject a synthesised tool-result user
                # message — those start with the tool block header.
                if m["content"].startswith("[SHERLOCK TOOL RESULTS"):
                    continue
                assert m["content"] in msgs
