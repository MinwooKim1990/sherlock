"""v1.5 Stage 3 — LLM-2 memory-consistency check (code-first).

Flags pinned facts the new user message contradicts (negation mismatch / number
divergence, gated by topical overlap). OFF by default → slot byte-identical.
"code" = pure-code, inline. "code+llm2" = confirm code candidates with one LLM-2
call. Reuses the module-level `_looks_contradictory` / `_fact_tokens`.
"""

from __future__ import annotations

from sherlock import Sherlock
from sherlock.memory.entry import MemorySource, MemoryType


class _E:
    """Minimal stand-in for a MemoryEntry (the check only reads pinned/content/id)."""

    def __init__(self, content, pinned=True, id="m1"):
        self.content = content
        self.pinned = pinned
        self.id = id


def _agent(tmp_path, name, mode="off", summary_chat=None):
    return Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        summary_chat=summary_chat,
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        memory_consistency_check=mode,
    )


# ---------- OFF (default) ----------------------------------------------------
def test_off_returns_nothing(tmp_path):
    a = _agent(tmp_path, "off")  # default "off"
    out = a._check_memory_consistency(
        "I'm not allergic to shellfish anymore", [_E("I'm allergic to shellfish")]
    )
    assert out == []


# ---------- code mode: contradiction detection -------------------------------
def test_code_flags_negation_contradiction(tmp_path):
    a = _agent(tmp_path, "neg", mode="code")
    out = a._check_memory_consistency(
        "actually I'm not allergic to shellfish at all", [_E("I'm allergic to shellfish")]
    )
    assert len(out) == 1 and "shellfish" in out[0]["fact"]


def test_code_flags_number_divergence(tmp_path):
    a = _agent(tmp_path, "num", mode="code")
    out = a._check_memory_consistency("I have 3 dogs now", [_E("I have 2 dogs")])
    assert len(out) == 1


def test_code_no_false_positive_unrelated_numbers(tmp_path):
    a = _agent(tmp_path, "unrel", mode="code")
    # different topic → no shared content word → never compared
    assert a._check_memory_consistency("the meeting is at 3pm", [_E("I have 2 dogs")]) == []


def test_code_no_false_positive_first_person_pronoun(tmp_path):
    # AUDIT BUG-1: "i" must NOT count as the shared topic word — two unrelated
    # first-person statements with differing numbers must not flag.
    a = _agent(tmp_path, "fp", mode="code")
    assert a._check_memory_consistency("I bought 3 coffees today", [_E("I have 2 kids")]) == []
    assert a._check_memory_consistency("I live at 12 Oak St", [_E("I have 2 cats")]) == []


def test_code_korean_negation_standalone(tmp_path):
    # Standalone Korean negation (못/안) with a substantive shared topic word fires.
    a = _agent(tmp_path, "ko", mode="code")
    out = a._check_memory_consistency(
        "나 매운 거 잘 못 먹어", [_E("나는 매운 음식을 진짜 잘 먹어")]
    )
    assert len(out) == 1


def test_code_no_flag_when_consistent(tmp_path):
    a = _agent(tmp_path, "ok", mode="code")
    assert a._check_memory_consistency("I have 2 dogs and love them", [_E("I have 2 dogs")]) == []


def test_code_ignores_non_pinned(tmp_path):
    a = _agent(tmp_path, "nonpin", mode="code")
    out = a._check_memory_consistency(
        "I'm not allergic to shellfish", [_E("I'm allergic to shellfish", pinned=False)]
    )
    assert out == []


def test_render_block_shape(tmp_path):
    a = _agent(tmp_path, "render", mode="code")
    block = a._render_consistency_block([{"fact": "I'm allergic to shellfish"}])
    assert "MEMORY-CONSISTENCY CHECK" in block
    assert '"I\'m allergic to shellfish"' in block


# ---------- code+llm2 escalation --------------------------------------------
def test_llm2_confirm_keeps_only_confirmed(tmp_path):
    a = _agent(
        tmp_path, "llm2keep", mode="code+llm2", summary_chat=lambda m: '{"contradictions": [0]}'
    )
    cands = [{"fact": "fact A", "fact_id": "1"}, {"fact": "fact B", "fact_id": "2"}]
    kept = a._llm2_confirm_contradictions("new msg", cands)
    assert kept == [cands[0]]


def test_llm2_confirm_drops_all_when_none(tmp_path):
    a = _agent(
        tmp_path, "llm2drop", mode="code+llm2", summary_chat=lambda m: '{"contradictions": []}'
    )
    cands = [{"fact": "fact A", "fact_id": "1"}]
    assert a._llm2_confirm_contradictions("new msg", cands) == []


def test_llm2_confirm_falls_back_on_garbage(tmp_path):
    a = _agent(tmp_path, "llm2bad", mode="code+llm2", summary_chat=lambda m: "not json at all")
    cands = [{"fact": "fact A", "fact_id": "1"}]
    # any parse failure → keep the code candidates (never lose a real conflict)
    assert a._llm2_confirm_contradictions("new msg", cands) == cands


def test_llm2_confirm_missing_key_falls_back(tmp_path):
    # AUDIT: a valid dict WITHOUT "contradictions" is not a verdict → keep
    # candidates (don't silently suppress). An empty list IS a verdict → drop.
    a = _agent(tmp_path, "llm2nokey", mode="code+llm2", summary_chat=lambda m: '{"answer": "none"}')
    cands = [{"fact": "fact A", "fact_id": "1"}]
    assert a._llm2_confirm_contradictions("new msg", cands) == cands


# ---------- slot integration -------------------------------------------------
def _pin(agent, content):
    cid = agent._ensure_conversation().id
    agent.memory.add(
        conversation_id=cid,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        pinned=True,
    )
    return cid


def test_slot_off_no_consistency_block(tmp_path):
    a = _agent(tmp_path, "slotoff")  # off
    _pin(a, "I'm allergic to shellfish")
    a.chat("actually I'm not allergic to shellfish at all")
    final = a.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "MEMORY-CONSISTENCY CHECK" not in final


def test_slot_code_injects_consistency_block(tmp_path):
    a = _agent(tmp_path, "slotcode", mode="code")
    _pin(a, "I'm allergic to shellfish")
    a.chat("actually I'm not allergic to shellfish at all")
    final = a.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "MEMORY-CONSISTENCY CHECK" in final
    assert "shellfish" in final
