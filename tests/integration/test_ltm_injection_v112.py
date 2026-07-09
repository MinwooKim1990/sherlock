"""v1.12 Stage A2 — long-term memory INJECTION (USER PROFILE tier + RAG channel).

Covers the read side of cross-conversation memory:
  - OFF (default) is byte-identical: no USER PROFILE block, no second (sentinel)
    hybrid search, no ``ltm_*`` keys in the ``slot.assembled`` event;
  - ON: the USER PROFILE tier-2 block renders after the persona block, with the
    ALWAYS-category-first ranking and the facts / chars caps enforced;
  - the sentinel RAG channel surfaces a durable fact that is NOT in the profile
    block, dedups the ones that ARE, and stays silent when rag_channel is off;
  - the headline cross-session path: a fact promoted in one conversation reaches
    the LLM-1 prompt of a NEW Sherlock instance in a NEW conversation;
  - achat parity: the same block rides the async path.

Hermetic: fake embeddings (deterministic) + a canned main/summary callable.
Sentinel rows are seeded directly through the store so ranking/caps are exact.
"""

from __future__ import annotations

import json

import pytest

from sherlock import Sherlock
from sherlock.budget import PROFILE_8K, count_tokens
from sherlock.memory.entry import LTM_CONVERSATION_ID, MemorySource, MemoryType

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _capturing_main(captured: list[str], reply: str = "Noted."):
    """Main (LLM-1) callable that records each turn's system message."""

    def main(messages):
        for m in messages:
            role = m["role"] if isinstance(m, dict) else m.role
            content = m["content"] if isinstance(m, dict) else m.content
            if role == "system":
                captured.append(content)
                break
        return reply

    return main


def _agent(tmp_path, *, long_term, main_chat=None, summary_chat=None, companions="off"):
    return Sherlock.with_callable(
        main_chat=main_chat or _capturing_main([]),
        summary_chat=summary_chat,
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
        embedding="fake",  # deterministic; entity/BM25 channel drives recall
        background=False,
        companions_mode=companions,
        long_term=long_term,
    )


def _seed(agent, content, category, confidence, *, origin="conv-origin"):
    """Seed one durable sentinel-scope fact (as the promoter would write it)."""
    return agent.memory.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=confidence,
        pinned=True,
        tags=f"ltm,{category}",
        evidence=json.dumps([{"quote": content.lower()[:24], "turn": 0}]),
        origin_conversation_id=origin,
        dedup=False,
    )


def _spy_search(agent):
    """Record the conversation_id of every hybrid.search call. Returns the list."""
    scopes: list = []
    orig = agent._hybrid.search

    def spy(*a, **k):
        scopes.append(k.get("conversation_id"))
        return orig(*a, **k)

    agent._hybrid.search = spy
    return scopes


# ---------------------------------------------------------------------------
# OFF (default) — byte-identical
# ---------------------------------------------------------------------------


def test_off_default_no_profile_no_sentinel_search(tmp_path):
    captured: list[str] = []
    events: list[dict] = []
    agent = _agent(tmp_path, long_term=False, main_chat=_capturing_main(captured))
    # Sentinel rows EXIST but the feature is off — they must be invisible.
    _seed(agent, "User's name is Kim", "identity_health", 1.0)
    agent.set_event_sink(events.append)
    scopes = _spy_search(agent)

    agent.chat("Hello there")

    assert captured, "main never ran"
    assert all("USER PROFILE" not in c for c in captured)
    # No second hybrid search over the sentinel scope.
    assert LTM_CONVERSATION_ID not in scopes
    # slot.assembled carries no long-term keys when disabled.
    slot = next(e for e in events if e["type"] == "slot.assembled")
    assert "ltm_profile_facts" not in slot["data"]
    assert "ltm_rag_hits" not in slot["data"]


# ---------------------------------------------------------------------------
# ON — the USER PROFILE block: placement, ranking, caps
# ---------------------------------------------------------------------------


def test_profile_block_ranking_always_first(tmp_path):
    agent = _agent(tmp_path, long_term=True)
    _seed(agent, "User prefers dark roast coffee", "stable_preference", 0.95)
    _seed(agent, "User is allergic to peanuts", "identity_health", 0.50)
    _seed(agent, "Always answer in metric units", "user_directive", 0.60)
    _seed(agent, "User's name is Kim", "identity_health", 0.90)

    sel = agent._select_ltm_profile_rows()
    order = [c for _, c in sel]
    assert order == [
        "User's name is Kim",  # identity_health 0.90 (bucket 0, conf desc)
        "Always answer in metric units",  # user_directive 0.60
        "User is allergic to peanuts",  # identity_health 0.50
        "User prefers dark roast coffee",  # stable_preference (bucket 1) last
    ]
    # The block itself: header present, one bullet per fact, category-tagged.
    block = agent._format_user_profile_block()
    assert block.startswith("[USER PROFILE")
    assert "- [identity_health] User's name is Kim" in block
    # identity_health (bucket 0) ranks above stable_preference despite lower conf.
    assert block.index("allergic to peanuts") < block.index("dark roast coffee")


def test_profile_block_after_persona(tmp_path):
    agent = _agent(tmp_path, long_term=True)
    conv = agent._ensure_conversation()
    # Seed a persona summary in the ACTIVE conversation scope.
    agent.memory.add(
        conversation_id=conv.id,
        content="The user is terse and prefers bullet points.",
        type=MemoryType.SUMMARY,
        source=MemorySource.SYSTEM,
        confidence=1.0,
        pinned=True,
        tags="persona_summary",
        dedup=False,
    )
    _seed(agent, "User's name is Kim", "identity_health", 1.0)

    msgs = agent._assemble_messages("hello", [], [], [], topic_changed=False)
    sys = msgs[0].content
    assert "PERSONA SUMMARY" in sys and "USER PROFILE" in sys
    assert sys.index("PERSONA SUMMARY") < sys.index("USER PROFILE")


def test_profile_facts_cap(tmp_path):
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_facts": 2})
    _seed(agent, "User's name is Kim", "identity_health", 0.99)
    _seed(agent, "Always answer in metric units", "user_directive", 0.80)
    _seed(agent, "User is allergic to peanuts", "identity_health", 0.60)
    _seed(agent, "User prefers dark roast coffee", "stable_preference", 0.90)

    sel = agent._select_ltm_profile_rows()
    assert len(sel) == 2
    assert [c for _, c in sel] == [
        "User's name is Kim",
        "Always answer in metric units",
    ]


def test_profile_chars_cap_trims(tmp_path):
    # profile_max_chars leaves ~29 chars for the second fact — above the 16-char
    # F6 fragment floor, so it is trimmed (not dropped) to what's left.
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_chars": 70})
    a = "User's name is Kim and they live in Seoul"  # 41 chars (bucket 0, high)
    b = "User is allergic to peanuts and to tree nuts too"  # long (bucket 0, lower)
    _seed(agent, a, "identity_health", 0.99)
    _seed(agent, b, "identity_health", 0.70)
    _seed(agent, "User likes tea", "stable_preference", 0.90)

    sel = agent._select_ltm_profile_rows()
    total = sum(len(c) for _, c in sel)
    assert total <= 70
    # First fact fits verbatim; the second is trimmed to what's left of the budget.
    assert sel[0][1] == a
    assert len(sel[1][1]) < len(b)


def test_zero_rows_no_block(tmp_path):
    agent = _agent(tmp_path, long_term=True)
    assert agent._select_ltm_profile_rows() == []
    assert agent._format_user_profile_block() == ""


# ---------------------------------------------------------------------------
# sentinel RAG channel
# ---------------------------------------------------------------------------


def test_rag_surfaces_non_profile_fact(tmp_path):
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_facts": 2})
    _seed(agent, "User is allergic to peanuts", "identity_health", 1.0)
    _seed(agent, "Always answer in metric units", "user_directive", 0.9)
    # bucket 1 → outside the top-2 profile, but recallable via RAG.
    zurich = _seed(agent, "User enjoys visiting Zurich", "stable_preference", 0.8)

    profile_ids = {e.id for e, _ in agent._select_ltm_profile_rows()}
    assert zurich.id not in profile_ids  # confirms it's NOT in the profile block

    agent._ensure_conversation()
    retrieved = agent._retrieve_memories("Tell me about Zurich", current_turn_index=1)
    ids = {e.id for e, _ in retrieved}
    assert zurich.id in ids  # surfaced through the sentinel RAG channel


def test_rag_dedups_profile_fact(tmp_path):
    agent = _agent(tmp_path, long_term=True)  # default profile_max_facts=12
    kim = _seed(agent, "User enjoys visiting Zurich often", "identity_health", 1.0)

    profile_ids = {e.id for e, _ in agent._select_ltm_profile_rows()}
    assert kim.id in profile_ids  # it IS in the profile block

    agent._ensure_conversation()
    retrieved = agent._retrieve_memories("Tell me about Zurich", current_turn_index=1)
    # Already live in TIER 2 via the profile block → not re-paid for via RAG.
    assert kim.id not in {e.id for e, _ in retrieved}


def test_rag_channel_disabled_no_second_search(tmp_path):
    agent = _agent(
        tmp_path,
        long_term={"enabled": True, "rag_channel": False, "profile_max_facts": 1},
    )
    _seed(agent, "User is allergic to peanuts", "identity_health", 1.0)
    zurich = _seed(agent, "User enjoys visiting Zurich", "stable_preference", 0.8)

    agent._ensure_conversation()
    scopes = _spy_search(agent)
    retrieved = agent._retrieve_memories("Tell me about Zurich", current_turn_index=1)

    assert LTM_CONVERSATION_ID not in scopes  # sentinel channel never searched
    assert zurich.id not in {e.id for e, _ in retrieved}


def test_sentinel_hits_downweighted(tmp_path):
    # A sentinel RAG hit's reported score is scaled by the tier-4 fallback weight
    # (0.5 by default) so a session hit of equal raw score wins the tie.
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_facts": 1})
    _seed(agent, "User is allergic to peanuts", "identity_health", 1.0)  # fills profile
    _seed(agent, "User enjoys visiting Zurich", "stable_preference", 0.8)  # RAG-only
    agent._ensure_conversation()
    sentinel = agent._ltm_sentinel_hits("Tell me about Zurich", 1)
    assert sentinel, "sentinel channel returned nothing"
    raw = agent._hybrid.search(
        "Tell me about Zurich",
        conversation_id=LTM_CONVERSATION_ID,
        top_k=3,
        current_turn_index=None,  # match the sentinel channel: no cross-conv recency boost
    )
    raw_score = {e.id: s for e, s in raw}
    for e, s in sentinel:
        assert s == pytest.approx(raw_score[e.id] * 0.5)


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


def test_slot_assembled_ltm_keys_when_on(tmp_path):
    events: list[dict] = []
    agent = _agent(tmp_path, long_term=True)
    _seed(agent, "User's name is Kim", "identity_health", 1.0)
    _seed(agent, "Always answer in metric units", "user_directive", 0.9)
    agent.set_event_sink(events.append)

    agent.chat("Hello there")

    slot = next(e for e in events if e["type"] == "slot.assembled")
    assert slot["data"]["ltm_profile_facts"] == 2
    assert "ltm_rag_hits" in slot["data"]


# ---------------------------------------------------------------------------
# headline: cross-session promotion → injection
# ---------------------------------------------------------------------------

_USER_MSG = "Hi, my name is Kim and I am allergic to peanuts."
_IDENTITY_FACT = {
    "content": "User's name is Kim",
    "type": "fact",
    "source": "user",
    "confidence": 1.0,
    "quote": "my name is kim",
    "pin_recommended": True,
    "let_fade": False,
    "long_term": True,
    "category": "identity_health",
}


def _llm2_payload(facts) -> str:
    return json.dumps(
        {
            "summary": "",
            "facts": facts,
            "topic_label": "t",
            "topic_changed_from_previous": False,
            "retrieval_keywords": [],
        }
    )


def test_cross_session_promoted_fact_reaches_new_conversation(tmp_path):
    # Conversation 1: promote a durable identity fact (real promotion path).
    agent1 = Sherlock.with_callable(
        main_chat=lambda _m: "Noted.\n<<sherlock-companions: compact>>",
        summary_chat=lambda _m: _llm2_payload([dict(_IDENTITY_FACT)]),
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        companions_mode="turbo",
        long_term=True,
    )
    agent1.chat(_USER_MSG)
    assert len(agent1.long_term_memory()) == 1

    # A brand-new Sherlock on the SAME storage_dir, chatting in a NEW conversation.
    captured: list[str] = []
    agent2 = Sherlock.with_callable(
        main_chat=_capturing_main(captured),
        summary_chat=lambda _m: _llm2_payload([]),
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        companions_mode="off",
        long_term=True,
    )
    agent2.chat("What's the weather like?")

    assert agent2.conversation_id != agent1.conversation_id  # genuinely new conv
    assert captured, "agent2 main never ran"
    sys = captured[0]
    assert "USER PROFILE" in sys
    assert "User's name is Kim" in sys


# ---------------------------------------------------------------------------
# achat parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_block_achat_parity(tmp_path):
    captured: list[str] = []

    async def main(messages):
        for m in messages:
            role = m["role"] if isinstance(m, dict) else m.role
            content = m["content"] if isinstance(m, dict) else m.content
            if role == "system":
                captured.append(content)
                break
        return "Noted."

    agent = _agent(tmp_path, long_term=True, main_chat=main)
    _seed(agent, "User's name is Kim", "identity_health", 1.0)

    await agent.achat("Hello there")

    assert captured, "achat main never ran"
    assert "USER PROFILE" in captured[0]
    assert "User's name is Kim" in captured[0]


# ---------------------------------------------------------------------------
# v1.12 A2 audit fixes — CJK trim (a), PROFILE_8K starvation (b),
# ltm_rag_hits value (c), F3 single-compute regression (d), F6 fragment floor
# ---------------------------------------------------------------------------


def test_profile_chars_cap_trims_cjk(tmp_path):
    # (a) A Korean fact longer than the remaining char budget is trimmed to a
    # valid, NON-EMPTY prefix — Python slices by code point, so no garbled /
    # split-character output and no empty bullet.
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_chars": 40})
    a = "사용자의 이름은 김민우"  # 12 chars, bucket 0, high conf → rides verbatim
    b = "사용자는 땅콩과 견과류 알레르기가 있으니 반드시 확인할 것"  # long, bucket 0, lower conf
    _seed(agent, a, "identity_health", 0.99)
    _seed(agent, b, "identity_health", 0.70)
    _seed(agent, "차를 좋아함", "stable_preference", 0.90)

    sel = agent._select_ltm_profile_rows()
    total = sum(len(c) for _, c in sel)
    assert total <= 40
    assert sel[0][1] == a  # first fact verbatim
    trimmed = sel[1][1]
    assert 0 < len(trimmed) < len(b)  # trimmed, but non-empty
    assert b.startswith(trimmed)  # a real prefix of the Korean fact, not mojibake

    block = agent._format_user_profile_block()
    assert block.startswith("[USER PROFILE")
    assert trimmed in block


def test_profile_char_budget_skips_tiny_fragment(tmp_path):
    # (F6) With <16 chars of budget left after the first fact, the second fact is
    # dropped entirely rather than injected as a meaningless 3-9 char fragment.
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_chars": 50})
    a = "User's name is Kim and they live in Seoul"  # 41 chars, bucket 0, high
    b = "User is allergic to peanuts and tree nuts"  # bucket 0, lower conf
    _seed(agent, a, "identity_health", 0.99)
    _seed(agent, b, "identity_health", 0.70)

    sel = agent._select_ltm_profile_rows()
    # 50 - 41 = 9 chars left → below the 16-char floor → no fragment of `b`.
    assert [c for _, c in sel] == [a]


def test_profile_8k_caps_block_keeps_highlights(tmp_path):
    # (b) On the 8K budget profile a worst-case (>tier-2-budget) USER PROFILE must
    # be capped to compacted_memory_max // 3 so the session highlights still get a
    # non-zero budget instead of being starved to nothing.
    agent = _agent(
        tmp_path,
        long_term={"enabled": True, "profile_max_facts": 40, "profile_max_chars": 6000},
    )
    for i in range(40):
        _seed(
            agent,
            f"User durable preference number {i:02d}: " + "some detail " * 12,
            "stable_preference",
            0.99 - i * 0.001,
        )
    raw_profile = agent._format_user_profile_block()
    assert count_tokens(raw_profile) > PROFILE_8K.compacted_memory_max  # genuinely worst-case

    conv = agent._ensure_conversation()
    for i in range(2):
        agent.memory.add(
            conversation_id=conv.id,
            content=f"Session highlight number {i}",
            type=MemoryType.SUMMARY,
            source=MemorySource.SYSTEM,
            confidence=1.0,
            tags="",
            dedup=False,
        )

    agent._slot_budget = PROFILE_8K  # force the small-window budget
    msgs = agent._assemble_messages("hello", [], [], [], topic_changed=False)
    sys = msgs[0].content

    assert "USER PROFILE" in sys
    start = sys.index("[USER PROFILE")
    rest = sys[start:]
    nxt = rest.find("\n\n[", 1)
    profile_block = rest if nxt < 0 else rest[:nxt]
    cap = PROFILE_8K.compacted_memory_max // 3
    assert count_tokens(profile_block) <= cap + 16  # capped at a third of tier-2
    assert count_tokens(profile_block) < count_tokens(raw_profile)  # cap actually bit
    # The highlights were NOT starved to a zero budget (the F2 regression).
    assert "COMPACTED MEMORY HIGHLIGHTS" in sys


def test_slot_assembled_ltm_rag_hits_value(tmp_path):
    # (c) Assert the ltm_rag_hits VALUE, not just its presence: exactly the one
    # non-profile sentinel fact surfaces through the RAG channel.
    events: list[dict] = []
    agent = _agent(tmp_path, long_term={"enabled": True, "profile_max_facts": 2})
    _seed(agent, "User is allergic to peanuts", "identity_health", 1.0)  # profile
    _seed(agent, "Always answer in metric units", "user_directive", 0.9)  # profile
    _seed(agent, "User enjoys visiting Zurich", "stable_preference", 0.8)  # RAG-only
    agent.set_event_sink(events.append)

    agent.chat("Tell me about Zurich")

    slot = next(e for e in events if e["type"] == "slot.assembled")
    assert slot["data"]["ltm_profile_facts"] == 2
    assert slot["data"]["ltm_rag_hits"] == 1


def test_ltm_profile_selection_computed_once_per_turn(tmp_path):
    # (d) F3 drift regression: with rag_channel ON the retrieval branch stashes the
    # selection ONCE and both the sentinel dedup and the injected block reuse the
    # snapshot — so _select_ltm_profile_rows runs exactly once per turn (a second,
    # unlocked recompute is what a mid-turn compaction could desync).
    agent = _agent(tmp_path, long_term=True)  # rag_channel default ON
    _seed(agent, "User's name is Kim", "identity_health", 1.0)
    _seed(agent, "User enjoys visiting Zurich", "stable_preference", 0.8)

    calls = {"n": 0}
    orig = agent._select_ltm_profile_rows

    def counting():
        calls["n"] += 1
        return orig()

    agent._select_ltm_profile_rows = counting
    agent.chat("Tell me about Zurich")

    assert calls["n"] == 1
