"""v1.12 Stage A1 — LLM-2 long-term promotion gate (summarizer level).

Covers the pieces that are cleanest to exercise directly against the
SummarizerEngine + MemoryStore (no agent): the prompt suffix kill-switch,
the code-level taxonomy gate, incognito write-blocking, the correction →
long-term supersede propagation (incl. the invalid_at_turn bug fix), and the
long-term cap. Cross-conversation restart + events + decay isolation live in
the integration suite (test_ltm_promotion_v112.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock.config import LongTermMemoryConfig
from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import LTM_CONVERSATION_ID, MemoryState
from sherlock.memory.summarizer import DEFAULT_LLM2_PROMPT, SummarizerEngine
from sherlock.providers.base import ChatMessage
from sherlock.providers.fake import FakeProvider


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


class _RecordingProvider(FakeProvider):
    """FakeProvider that captures the system prompt each chat() call received."""

    def __init__(self, canned_reply: str) -> None:
        super().__init__(canned_reply=canned_reply)
        self.system_prompts: list[str] = []
        self.system_messages: list = []

    def chat(self, messages, **kwargs):
        for m in messages:
            if m.role == "system":
                self.system_prompts.append(m.content)
                self.system_messages.append(m)
                break
        return super().chat(messages, **kwargs)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    from sherlock.storage import Storage

    storage = Storage(tmp_path / "test.db")
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=build_embedding_provider(_FakeEmbedConfig()),
        vector_path=tmp_path / "vectors",
    )


def _payload(facts: list[dict], **extra) -> str:
    body = {
        "summary": "",
        "facts": facts,
        "topic_label": "test",
        "topic_changed_from_previous": False,
        "retrieval_keywords": [],
    }
    body.update(extra)
    return json.dumps(body)


def _run(
    store: MemoryStore,
    payload: str,
    *,
    long_term=None,
    turn_index: int = 3,
    transcript: str = "hi there",
    provider=None,
) -> dict:
    engine = SummarizerEngine(
        provider=provider or FakeProvider(canned_reply=payload),
        store=store,
        long_term=long_term,
    )
    return engine.run(
        conversation_id="c",
        recent_turns=[ChatMessage(role="user", content=transcript)],
        turn_index=turn_index,
    )


def _sentinel(store: MemoryStore) -> list:
    return store.list(conversation_id=LTM_CONVERSATION_ID)


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


# ---------- kill switch: OFF is byte-identical + no sentinel writes ----------


def test_disabled_no_suffix_and_no_sentinel_rows(store):
    rec = _RecordingProvider(_payload([dict(_IDENTITY_FACT)]))
    result = _run(store, "", long_term=None, provider=rec, transcript="my name is kim")
    sys_prompt = rec.system_prompts[0]
    # No long-term instruction leaked into the LLM-2 prompt.
    assert "LONG-TERM MEMORY" not in sys_prompt
    assert '"long_term"' not in sys_prompt
    assert '"category"' not in sys_prompt
    # No promotions, no sentinel rows.
    assert result["long_term_promoted"] == []
    assert _sentinel(store) == []


def test_disabled_config_object_still_off(store):
    """A LongTermMemoryConfig with enabled=False behaves like None."""
    rec = _RecordingProvider(_payload([dict(_IDENTITY_FACT)]))
    cfg = LongTermMemoryConfig(enabled=False)
    result = _run(store, "", long_term=cfg, provider=rec, transcript="my name is kim")
    assert "LONG-TERM MEMORY" not in rec.system_prompts[0]
    assert result["long_term_promoted"] == []
    assert _sentinel(store) == []


def test_enabled_prompt_gains_suffix(store):
    rec = _RecordingProvider(_payload([]))
    cfg = LongTermMemoryConfig(enabled=True)
    _run(store, "", long_term=cfg, provider=rec)
    sys_prompt = rec.system_prompts[0]
    assert "LONG-TERM MEMORY" in sys_prompt
    assert '"long_term"' in sys_prompt
    assert "identity_health" in sys_prompt


# ---------- promotion ----------


def test_promote_identity_health_row_shape(store):
    cfg = LongTermMemoryConfig(enabled=True)
    result = _run(
        store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim"
    )
    rows = _sentinel(store)
    assert len(rows) == 1
    row = rows[0]
    assert row.content == "User's name is Kim"
    assert row.pinned is True
    assert row.origin_conversation_id == "c"
    assert "ltm" in row.tags and "identity_health" in row.tags
    # Evidence carries the grounded quote + origin turn.
    ev = json.loads(row.evidence)
    assert ev and ev[0]["quote"] == "my name is kim" and ev[0]["turn"] == 3
    # Result dict reports it.
    assert result["long_term_promoted"] == [
        {"content": "User's name is Kim", "category": "identity_health", "id": row.id}
    ]


def test_user_directive_always_promotes_even_without_quote(store):
    cfg = LongTermMemoryConfig(enabled=True)
    fact = {
        "content": "Always address the user as Captain",
        "type": "fact",
        "source": "user",
        "confidence": 0.9,
        "pin_recommended": True,
        "long_term": True,
        "category": "user_directive",
    }
    _run(store, _payload([fact]), long_term=cfg)
    rows = _sentinel(store)
    assert len(rows) == 1
    assert "user_directive" in rows[0].tags


# ---------- taxonomy gate ----------


def test_taxonomy_gate_promotes_only_qualifying(store):
    cfg = LongTermMemoryConfig(enabled=True)
    quote = "grounded phrase here"
    transcript = "context: grounded phrase here and more talk"
    facts = [
        # none/unknown → never (even with long_term=true).
        {
            "content": "Booked a 3pm dentist slot today",
            "type": "fact",
            "source": "user",
            "confidence": 0.9,
            "long_term": True,
            "category": "none",
        },
        # conservative, confidence 0.5 → skip.
        {
            "content": "Prefers dark mode maybe",
            "type": "fact",
            "source": "user",
            "confidence": 0.5,
            "quote": quote,
            "long_term": True,
            "category": "stable_preference",
        },
        # conservative, 0.8 but NO quote → skip.
        {
            "content": "Sister is named Mina",
            "type": "fact",
            "source": "user",
            "confidence": 0.8,
            "long_term": True,
            "category": "relationship",
        },
        # conservative, 0.8 + grounded quote → promote.
        {
            "content": "Building a novel-writing app over the next year",
            "type": "fact",
            "source": "user",
            "confidence": 0.8,
            "quote": quote,
            "long_term": True,
            "category": "long_term_project",
        },
        # identity → always promote.
        dict(_IDENTITY_FACT),
    ]
    result = _run(
        store,
        _payload(facts),
        long_term=cfg,
        transcript=transcript + " my name is kim",
    )
    promoted_contents = {p["content"] for p in result["long_term_promoted"]}
    assert promoted_contents == {
        "Building a novel-writing app over the next year",
        "User's name is Kim",
    }
    assert len(_sentinel(store)) == 2


def test_conservative_ungrounded_quote_not_promoted(store):
    """An ungrounded quote caps confidence to 0.5 → conservative gate fails."""
    cfg = LongTermMemoryConfig(enabled=True)
    fact = {
        "content": "Loves hiking every weekend",
        "type": "fact",
        "source": "user",
        "confidence": 0.9,
        "quote": "this exact string is nowhere in the transcript at all",
        "long_term": True,
        "category": "stable_preference",
    }
    result = _run(store, _payload([fact]), long_term=cfg, transcript="totally unrelated talk")
    assert result["long_term_promoted"] == []
    assert _sentinel(store) == []


# ---------- incognito ----------


def test_incognito_no_suffix_and_no_writes(store):
    rec = _RecordingProvider(_payload([dict(_IDENTITY_FACT)]))
    cfg = LongTermMemoryConfig(enabled=True, incognito=True)
    result = _run(store, "", long_term=cfg, provider=rec, transcript="my name is kim")
    # Suffix absent (byte-clean) AND no writes.
    assert "LONG-TERM MEMORY" not in rec.system_prompts[0]
    assert result["long_term_promoted"] == []
    assert _sentinel(store) == []


# ---------- contradiction: correction supersedes the long-term copy ----------


def test_correction_supersedes_long_term_and_sets_invalid_at_turn(store):
    cfg = LongTermMemoryConfig(enabled=True)
    # Turn 3: promote identity fact (also lands as a pinned session fact).
    _run(store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim")
    sentinel_before = _sentinel(store)
    assert len(sentinel_before) == 1
    old_ltm = sentinel_before[0]

    # Turn 6: LLM-2 corrects the fact referenced as [M1] in ALREADY-KNOWN.
    corr_payload = _payload([], corrections=[{"replaces": "M1", "content": "User's name is Lee"}])
    _run(store, corr_payload, long_term=cfg, turn_index=6, transcript="actually my name is lee")

    # Old long-term row is frozen with the bug-fixed invalid_at_turn populated.
    refreshed_old = store.get(old_ltm.id)
    assert refreshed_old.superseded_by is not None
    assert refreshed_old.invalid_at_turn == 6

    # A live corrected long-term row now exists.
    live = [e for e in _sentinel(store) if not e.superseded_by]
    assert len(live) == 1
    assert live[0].content == "User's name is Lee"

    # Bug-fix parity on the session path too (invalid_at_turn populated).
    session_old = [
        e
        for e in store.list(conversation_id="c")
        if e.content == "User's name is Kim" and e.superseded_by
    ]
    assert session_old and session_old[0].invalid_at_turn == 6


# ---------- long-term cap ----------


def test_ltm_cap_evicts_lowest_confidence(store):
    cfg = LongTermMemoryConfig(enabled=True, cap=2)
    facts = [
        {
            "content": "User is named Kim",
            "source": "user",
            "confidence": 0.95,
            "long_term": True,
            "category": "identity_health",
        },
        {
            "content": "User uses she/her pronouns",
            "source": "user",
            "confidence": 0.85,
            "long_term": True,
            "category": "identity_health",
        },
        {
            "content": "User is allergic to shellfish",
            "source": "user",
            "confidence": 0.75,
            "long_term": True,
            "category": "identity_health",
        },
    ]
    _run(store, _payload(facts), long_term=cfg)
    live = [e for e in _sentinel(store) if not e.superseded_by]
    assert len(live) == 2
    kept = {e.content for e in live}
    # Lowest-confidence row (0.75) was evicted.
    assert "User is allergic to shellfish" not in kept
    assert kept == {"User is named Kim", "User uses she/her pronouns"}


# ---------- decay isolation (store-level) ----------


# ---------- v1.12 audit fixes ----------


def test_off_state_prompt_byte_identical_and_cache_hint(store):
    """F10: OFF prompt == DEFAULT_LLM2_PROMPT and the cache hint covers it all."""
    rec = _RecordingProvider(_payload([]))
    _run(store, "", long_term=None, provider=rec)
    msg = rec.system_messages[0]
    assert msg.content == DEFAULT_LLM2_PROMPT
    assert msg.cache_stable_prefix_chars == len(msg.content)


def test_f1_ungrounded_quote_identity_not_promoted(store):
    """F1: a hallucination-flagged (ungrounded) quote never becomes a sentinel
    row even under an ALWAYS category."""
    cfg = LongTermMemoryConfig(enabled=True)
    fact = {
        "content": "User's name is Kim",
        "type": "fact",
        "source": "user",
        "confidence": 1.0,
        "quote": "this exact phrase is nowhere in the transcript",
        "pin_recommended": True,
        "long_term": True,
        "category": "identity_health",
    }
    result = _run(store, _payload([fact]), long_term=cfg, transcript="totally unrelated talk")
    assert result["long_term_promoted"] == []
    assert _sentinel(store) == []


def test_f3_conservative_requires_long_term_flag(store):
    """F3: a durable category alone does not promote — long_term=True required."""
    cfg = LongTermMemoryConfig(enabled=True)
    quote = "cousin lives in berlin"
    transcript = "we talked and my cousin lives in berlin now"
    base = {
        "content": "User's cousin lives in Berlin",
        "type": "fact",
        "source": "user",
        "confidence": 0.9,
        "quote": quote,
        "category": "relationship",
    }
    # long_term=False → blocked (over-promotion guard).
    result = _run(
        store, _payload([{**base, "long_term": False}]), long_term=cfg, transcript=transcript
    )
    assert result["long_term_promoted"] == []
    assert _sentinel(store) == []
    # Same fact with long_term=True → promotes (positive contrast).
    result2 = _run(
        store, _payload([{**base, "long_term": True}]), long_term=cfg, transcript=transcript
    )
    assert len(result2["long_term_promoted"]) == 1
    assert len(_sentinel(store)) == 1


def test_f2_cap_evicts_conservative_before_always(store):
    """F2: cap eviction is category-aware — a just-promoted directive outlives a
    higher-confidence conservative preference."""
    cfg = LongTermMemoryConfig(enabled=True, cap=2)
    quote = "grounded marker text"
    transcript = "context grounded marker text and more chatter"
    facts = [
        {  # ALWAYS, low confidence, no quote — must survive the cap.
            "content": "Always greet the user in Korean",
            "source": "user",
            "confidence": 0.6,
            "long_term": True,
            "category": "user_directive",
        },
        {  # conservative, high confidence — evictable.
            "content": "Prefers tea over coffee in the morning",
            "source": "user",
            "confidence": 0.9,
            "quote": quote,
            "long_term": True,
            "category": "stable_preference",
        },
        {  # conservative, high confidence — evictable.
            "content": "Prefers window seats on flights",
            "source": "user",
            "confidence": 0.9,
            "quote": quote,
            "long_term": True,
            "category": "stable_preference",
        },
    ]
    _run(store, _payload(facts), long_term=cfg, transcript=transcript)
    live = [e for e in _sentinel(store) if not e.superseded_by]
    assert len(live) == 2
    kept = {e.content for e in live}
    # The 0.6 directive outranks a 0.9 preference purely on category.
    assert "Always greet the user in Korean" in kept
    # Exactly one of the two preferences was evicted.
    prefs_kept = kept & {
        "Prefers tea over coffee in the morning",
        "Prefers window seats on flights",
    }
    assert len(prefs_kept) == 1


def test_f4_correction_fallback_on_hash_mismatch(store):
    """F4: when a dedup-merge leaves the sentinel hash != the session hash, the
    correction still lands via the 60-char-prefix fallback."""
    cfg = LongTermMemoryConfig(enabled=True)
    long_content = "The user's complete mailing address is 1234 Elm Street, Apartment 5B"
    fact = {
        "content": long_content,
        "type": "fact",
        "source": "user",
        "confidence": 1.0,
        "pin_recommended": True,
        "long_term": True,
        "category": "identity_health",
    }
    _run(store, _payload([fact]), long_term=cfg)
    rows = _sentinel(store)
    assert len(rows) == 1
    sentinel_row = rows[0]
    assert sentinel_row.content == long_content

    # Build an old_content that shares the first 60 chars but differs in the
    # tail → different content hash, same prefix (the dedup-merge scenario).
    stale = long_content[:60] + " (OLD DIVERGENT TAIL)"
    assert stale != long_content

    engine = SummarizerEngine(
        provider=FakeProvider(canned_reply=_payload([])),
        store=store,
        long_term=cfg,
    )
    out: list[dict] = []
    engine._supersede_long_term(
        old_content=stale,
        new_content="The user's complete mailing address is now 9 Oak Lane",
        conversation_id="c",
        turn_index=9,
        promoted_out=out,
    )
    # The stale sentinel was superseded despite the hash mismatch.
    refreshed = store.get(sentinel_row.id)
    assert refreshed.superseded_by is not None
    live = [e for e in _sentinel(store) if not e.superseded_by]
    assert len(live) == 1
    assert live[0].content == "The user's complete mailing address is now 9 Oak Lane"
    assert len(out) == 1


def test_f4_no_fallback_when_prefix_differs(store):
    """F4: the fallback stays conservative — no shared 60-char prefix, no-op."""
    cfg = LongTermMemoryConfig(enabled=True)
    long_content = "The user's complete mailing address is 1234 Elm Street, Apartment 5B"
    fact = {
        "content": long_content,
        "type": "fact",
        "source": "user",
        "confidence": 1.0,
        "pin_recommended": True,
        "long_term": True,
        "category": "identity_health",
    }
    _run(store, _payload([fact]), long_term=cfg)
    sentinel_row = _sentinel(store)[0]

    engine = SummarizerEngine(
        provider=FakeProvider(canned_reply=_payload([])),
        store=store,
        long_term=cfg,
    )
    out: list[dict] = []
    engine._supersede_long_term(
        old_content="A completely unrelated fact about something else entirely",
        new_content="should not land",
        conversation_id="c",
        turn_index=9,
        promoted_out=out,
    )
    refreshed = store.get(sentinel_row.id)
    assert refreshed.superseded_by is None
    assert out == []


def test_f5_frozen_sentinel_rows_bounded_by_cap(store):
    """F5: repeated corrections don't grow the store — superseded frozen rows
    are hard-deleted oldest-first past the cap."""
    cfg = LongTermMemoryConfig(enabled=True, cap=2)
    _run(store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim")

    names = ["Lee", "Park", "Choi", "Han"]
    turn = 6
    for nm in names:
        corr = _payload([], corrections=[{"replaces": "M1", "content": f"User's name is {nm}"}])
        _run(
            store,
            corr,
            long_term=cfg,
            turn_index=turn,
            transcript=f"actually my name is {nm.lower()}",
        )
        turn += 3

    all_rows = _sentinel(store)
    frozen = [e for e in all_rows if e.superseded_by is not None]
    live = [e for e in all_rows if e.superseded_by is None]
    # Frozen rows never exceed the cap; a single live current value remains.
    assert len(frozen) <= cfg.cap
    assert len(live) == 1
    assert live[0].content == "User's name is Han"


def test_f10_dedup_repromotion_merges_to_single_row(store):
    """F10: re-promoting the same fact across runs merges to one sentinel row."""
    cfg = LongTermMemoryConfig(enabled=True)
    _run(store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim")
    _run(store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim")
    assert len(_sentinel(store)) == 1


def test_f10_incognito_correction_does_not_touch_sentinel(store):
    """F10: with incognito ON, a correction updates the session but leaves the
    durable sentinel copy untouched."""
    cfg = LongTermMemoryConfig(enabled=True)
    _run(store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim")
    before = _sentinel(store)
    assert len(before) == 1
    ltm_id = before[0].id

    cfg_incog = LongTermMemoryConfig(enabled=True, incognito=True)
    corr = _payload([], corrections=[{"replaces": "M1", "content": "User's name is Lee"}])
    _run(store, corr, long_term=cfg_incog, turn_index=6, transcript="actually my name is lee")

    after = _sentinel(store)
    assert len(after) == 1
    assert after[0].id == ltm_id
    assert after[0].superseded_by is None
    assert after[0].content == "User's name is Kim"


def test_sentinel_rows_survive_decay_on_active_conversation(store):
    from sherlock.memory.decay import DecayConfig, DecayEngine

    cfg = LongTermMemoryConfig(enabled=True)
    _run(store, _payload([dict(_IDENTITY_FACT)]), long_term=cfg, transcript="my name is kim")
    assert len(_sentinel(store)) == 1

    # Decay only ever runs on the ACTIVE conversation ("c"), never the sentinel.
    engine = DecayEngine(store, DecayConfig())
    for t in range(1, 200):
        engine.step(conversation_id="c", current_turn_index=t, active_topics=["unrelated"])

    row = _sentinel(store)[0]
    assert row.state == MemoryState.FRESH
    assert row.pinned is True
