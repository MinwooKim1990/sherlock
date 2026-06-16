"""v0.9 memory-layer fixes.

Covers:
  - dedup-merge resurrects FORGOTTEN/COLD rows to FRESH (all dedup paths)
  - 60-char-prefix dedup is update-on-conflict: the NEW content wins
  - cap_pinned ignores DEEP_RESEARCH docs (count AND demotion candidates)
  - summarizer provenance defaults: missing source → LLM_INFERENCE,
    missing confidence → 0.7
  - ALREADY-KNOWN block excludes DEEP_RESEARCH + persona_summary entries
  - created_turn_index: set at creation, immutable across dedup merges,
    auto-migrated onto pre-v0.9 DBs
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import MemoryEntry, MemorySource, MemoryState, MemoryType
from sherlock.memory.summarizer import SummarizerEngine, _coerce_source
from sherlock.providers.base import ChatMessage
from sherlock.providers.fake import FakeProvider
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


# Long enough that the normalised text exceeds the 60-char prefix window,
# so only the tail differs between the "stale" and "corrected" variants.
_PREFIX_BASE = "The user's preferred vacation destination for the family trip is "


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    storage = Storage(tmp_path / "test.db")
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=build_embedding_provider(_FakeEmbedConfig()),
        vector_path=tmp_path / "vectors",
    )


class _RecordingProvider(FakeProvider):
    """FakeProvider that captures the messages each chat() call received."""

    def __init__(self, canned_reply: str) -> None:
        super().__init__(canned_reply=canned_reply)
        self.seen: list[list[ChatMessage]] = []

    def chat(self, messages, **kwargs):
        self.seen.append(list(messages))
        return super().chat(messages, **kwargs)


def _llm2_payload(facts: list[dict]) -> str:
    return json.dumps(
        {
            "summary": "",
            "facts": facts,
            "topic_label": "test",
            "topic_changed_from_previous": False,
            "retrieval_keywords": [],
        }
    )


# ---------- (a) dedup resurrects FORGOTTEN/COLD rows ----------------------


def test_restating_forgotten_fact_resurrects_to_fresh(store):
    e = store.add(
        conversation_id="c",
        content="Yujin has a soba allergy",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    store.soft_delete(e.id)
    assert store.get(e.id).state == MemoryState.FORGOTTEN
    # Exact-hash dedup path: same content merges into the same row.
    merged = store.add(
        conversation_id="c",
        content="Yujin has a soba allergy",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    assert merged.id == e.id, "should dedup-merge, not create a new row"
    assert merged.state == MemoryState.FRESH


def test_prefix_restate_resurrects_cold_entry(store):
    e = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    store.update_state(e.id, MemoryState.COLD)
    # Prefix dedup path: same 60-char head, differing tail.
    merged = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    assert merged.id == e.id
    assert merged.state == MemoryState.FRESH


# ---------- (b) prefix-dedup correction: new content wins -----------------


def test_prefix_dedup_new_content_wins(store):
    old = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    merged = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    assert merged.id == old.id
    row = store.get(old.id)
    assert row.content == _PREFIX_BASE + "Busan"
    assert row.content_hash == MemoryEntry.compute_hash(_PREFIX_BASE + "Busan")
    # Derived stores follow the correction: Chroma document + entity index.
    got = store._collection.get(ids=[old.id], include=["documents"])
    assert got["documents"][0] == _PREFIX_BASE + "Busan"
    assert any(h.id == old.id for h in store.find_by_entities("c", {"busan"}))
    assert not any(h.id == old.id for h in store.find_by_entities("c", {"tokyo"}))


# ---------- (c) cap_pinned ignores DEEP_RESEARCH docs ---------------------


def test_cap_pinned_ignores_deep_research_docs(store):
    user_pin = store.add(
        conversation_id="c",
        content="User lives in Seoul",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
    )
    # 21 pinned research docs (a 20-round run + final synthesis), all
    # demotable-source — without the exemption these blow the cap.
    docs = [
        store.add(
            conversation_id="c",
            content=f"[deep_research:r1] round {i} — topic",
            type=MemoryType.DEEP_RESEARCH,
            source=MemorySource.SEARCH,
            confidence=0.6,
            pinned=True,
            tags="deep_research,r1",
            dedup=False,
        )
        for i in range(21)
    ]
    demoted = store.cap_pinned("c", max_pinned=18)
    assert demoted == 0
    assert store.get(user_pin.id).pinned is True
    assert all(store.get(d.id).pinned for d in docs)


# ---------- (d) summarizer provenance defaults ----------------------------


def test_summarizer_missing_source_and_confidence_default_to_inference(store):
    payload = _llm2_payload([{"content": "User may switch to the night shift"}])
    engine = SummarizerEngine(provider=FakeProvider(canned_reply=payload), store=store)
    engine.run(
        conversation_id="c",
        recent_turns=[ChatMessage(role="user", content="hi")],
        turn_index=3,
    )
    fact = next(r for r in store.list(conversation_id="c") if "night shift" in r.content)
    assert fact.source == MemorySource.LLM_INFERENCE
    assert fact.confidence == pytest.approx(0.7)


def test_summarizer_explicit_user_source_still_maps_to_user(store):
    payload = _llm2_payload(
        [{"content": "User lives in Busan", "source": "user", "confidence": 1.0}]
    )
    engine = SummarizerEngine(provider=FakeProvider(canned_reply=payload), store=store)
    engine.run(
        conversation_id="c",
        recent_turns=[ChatMessage(role="user", content="hi")],
        turn_index=3,
    )
    fact = next(r for r in store.list(conversation_id="c") if "Busan" in r.content)
    assert fact.source == MemorySource.USER
    assert fact.confidence == pytest.approx(1.0)


def test_coerce_source_fallback_is_llm_inference():
    assert _coerce_source(None) == MemorySource.LLM_INFERENCE
    assert _coerce_source("not-a-source") == MemorySource.LLM_INFERENCE
    assert _coerce_source("user") == MemorySource.USER


# ---------- ALREADY-KNOWN block exclusions --------------------------------


def test_already_known_block_excludes_research_and_persona(store):
    store.add(
        conversation_id="c",
        content="PLAIN_PINNED_FACT",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
    )
    store.add(
        conversation_id="c",
        content="RESEARCH_DOC_CONTENT",
        type=MemoryType.DEEP_RESEARCH,
        source=MemorySource.SEARCH,
        pinned=True,
        tags="deep_research,r1",
        dedup=False,
    )
    store.add(
        conversation_id="c",
        content="PERSONA_SUMMARY_CONTENT",
        type=MemoryType.SUMMARY,
        source=MemorySource.LLM_INFERENCE,
        pinned=True,
        tags="persona_summary",
        dedup=False,
    )
    provider = _RecordingProvider(canned_reply=_llm2_payload([]))
    engine = SummarizerEngine(provider=provider, store=store)
    engine.run(
        conversation_id="c",
        recent_turns=[ChatMessage(role="user", content="hi")],
        turn_index=3,
    )
    user_msg = next(m.content for m in provider.seen[0] if m.role == "user")
    known = user_msg.split("--- END ALREADY-KNOWN ---", 1)[0]
    assert "PLAIN_PINNED_FACT" in known
    assert "RESEARCH_DOC_CONTENT" not in known
    assert "PERSONA_SUMMARY_CONTENT" not in known


# ---------- (e) created_turn_index ----------------------------------------


def test_created_turn_index_set_on_add_and_immutable_across_dedup(store):
    e = store.add(
        conversation_id="c",
        content="User works at Acme Corp",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=5,
    )
    assert e.created_turn_index == 5
    merged = store.add(
        conversation_id="c",
        content="User works at Acme Corp",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=12,
    )
    assert merged.id == e.id
    assert merged.last_used_turn_index == 12
    assert merged.created_turn_index == 5, "creation turn is immutable"


def test_run_migrations_adds_created_turn_index(tmp_path):
    """Pre-v0.9 memory_entry tables gain the column without crashing."""
    import sherlock.memory.entry  # noqa: F401 — register memory models in metadata
    from sherlock.storage.db import run_migrations

    db = tmp_path / "old.sqlite"
    eng = create_engine(f"sqlite:///{db}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE memory_entry (id TEXT PRIMARY KEY, conversation_id TEXT, "
                "content TEXT, type TEXT, source TEXT)"
            )
        )
    added = run_migrations(eng)
    assert "memory_entry.created_turn_index" in added, f"added={added}"
    with eng.begin() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(memory_entry)"))]
    assert "created_turn_index" in cols
