"""v1.0 reconsolidation memory-layer changes.

Covers:
  - D2: Korean-aware BM25 — Hangul runs ≥3 chars emit character bigrams
    alongside the original token (corpus AND query)
  - D1: retrieval_keywords persisted as ONE rolling entry, readable via
    latest_retrieval_keywords(), never surfaced by RAG
  - D3: LLM-2 corrections — stable [Mn] ids in ALREADY-KNOWN, supersede is
    non-destructive, frozen rows skip dedup/search/cap_pinned/decay
  - D4: memory tool renders " (superseded)" and `pinned` excludes frozen rows
  - superseded_by auto-migrated onto pre-v1.0 DBs
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.decay import DecayEngine
from sherlock.memory.entry import MemorySource, MemoryState, MemoryType
from sherlock.memory.summarizer import SummarizerEngine
from sherlock.providers.base import ChatMessage
from sherlock.providers.fake import FakeProvider
from sherlock.rag.hybrid import HybridSearch, _tokenise
from sherlock.storage import Storage
from sherlock.tools.memory_tool import memory_entity, memory_lookup, memory_pinned


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


def _llm2_payload(facts: list[dict], **extra) -> str:
    payload = {
        "summary": "",
        "facts": facts,
        "topic_label": "test",
        "topic_changed_from_previous": False,
        "retrieval_keywords": [],
    }
    payload.update(extra)
    return json.dumps(payload)


def _run_summarizer(store: MemoryStore, payload: str, *, turn_index: int = 3) -> dict:
    engine = SummarizerEngine(provider=FakeProvider(canned_reply=payload), store=store)
    return engine.run(
        conversation_id="c",
        recent_turns=[ChatMessage(role="user", content="hi")],
        turn_index=turn_index,
    )


def _rk_rows(store: MemoryStore) -> list:
    return [e for e in store.list(conversation_id="c") if "retrieval_keywords" in (e.tags or "")]


# ---------- (a) D2: Hangul bigram tokenisation -----------------------------


def test_tokenise_emits_hangul_bigrams():
    # Runs ≥3 chars: original token + every character bigram, in order.
    assert _tokenise("유진이는 메밀 알레르기") == [
        "유진이는",
        "유진",
        "진이",
        "이는",
        "메밀",  # 2-char run: bigram == token, no extra emission
        "알레르기",
        "알레",
        "레르",
        "르기",
    ]


def test_tokenise_ascii_unchanged():
    assert _tokenise("Peanut allergy EpiPen dosage for Yujin!") == [
        "peanut",
        "allergy",
        "epipen",
        "dosage",
        "yujin",
    ]


# ---------- (b) D2: hybrid retrieval via bigram overlap ---------------------


def test_hybrid_retrieves_korean_fact_via_bigrams(store):
    target = store.add(
        conversation_id="c",
        content="유진 메밀 알레르기",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    for filler in (
        "User works at Acme Corp",
        "User prefers window seats on flights",
        "Quarterly dashboard redesign is due Friday",
        "User adopted a cat named Phoebe",
    ):
        store.add(
            conversation_id="c",
            content=filler,
            type=MemoryType.FACT,
            source=MemorySource.USER,
        )
    hybrid = HybridSearch(store=store)
    # No whitespace token overlaps ("유진이는" ≠ "유진"); only the shared
    # bigram 유진 connects query and fact in the BM25 channel.
    hits = hybrid.search("유진이는 뭘 못 먹지", conversation_id="c", top_k=3)
    assert hits, "expected the Korean fact to be retrievable"
    assert hits[0][0].id == target.id


# ---------- (c) D1: retrieval_keywords rolling entry -------------------------


def test_retrieval_keywords_rolling_entry(store):
    parsed = _run_summarizer(
        store, _llm2_payload([], retrieval_keywords=["유진", "메밀", "allergy"])
    )
    # The parsed dict still carries the raw list (agent emits compact.done).
    assert parsed["retrieval_keywords"] == ["유진", "메밀", "allergy"]
    rows = _rk_rows(store)
    assert len(rows) == 1
    assert rows[0].content == "유진 메밀 allergy"
    assert rows[0].pinned is False
    assert rows[0].type == MemoryType.INFERENCE
    assert rows[0].confidence == pytest.approx(0.4)

    # Second run REPLACES the rolling entry — never accumulates.
    _run_summarizer(
        store, _llm2_payload([], retrieval_keywords=["soba", "restaurant"]), turn_index=6
    )
    rows = _rk_rows(store)
    assert len(rows) == 1
    assert rows[0].content == "soba restaurant"
    assert store.latest_retrieval_keywords("c") == "soba restaurant"

    # Never a RAG result — even on an exact-content query.
    store.add(
        conversation_id="c",
        content="User booked a soba restaurant for Saturday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    results = store.search("soba restaurant", conversation_id="c", top_k=10)
    assert results
    assert all("retrieval_keywords" not in (e.tags or "") for e, _ in results)


def test_latest_retrieval_keywords_empty_when_absent(store):
    assert store.latest_retrieval_keywords("c") == ""


# ---------- (d) D3: corrections supersede non-destructively -----------------


def test_already_known_block_renders_stable_ids(store):
    store.add(
        conversation_id="c",
        content="FACT_ALPHA",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        last_used_turn_index=5,
    )
    store.add(
        conversation_id="c",
        content="FACT_BETA",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        last_used_turn_index=2,
    )
    provider = _RecordingProvider(canned_reply=_llm2_payload([]))
    SummarizerEngine(provider=provider, store=store).run(
        conversation_id="c",
        recent_turns=[ChatMessage(role="user", content="hi")],
        turn_index=6,
    )
    user_msg = next(m.content for m in provider.seen[0] if m.role == "user")
    known = user_msg.split("--- END ALREADY-KNOWN ---", 1)[0]
    assert "- [M1] FACT_ALPHA" in known  # most recently used first
    assert "- [M2] FACT_BETA" in known


def test_correction_creates_new_pinned_row_and_freezes_old(store):
    old = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        last_used_turn_index=1,
    )
    payload = _llm2_payload([], corrections=[{"replaces": "M1", "content": _PREFIX_BASE + "Busan"}])
    _run_summarizer(store, payload)

    new = next(r for r in store.list(conversation_id="c") if r.content == _PREFIX_BASE + "Busan")
    assert new.pinned is True
    assert new.source == MemorySource.USER
    assert new.confidence == pytest.approx(0.9)
    assert new.superseded_by is None
    old_row = store.get(old.id)
    assert old_row.superseded_by == new.id
    assert old_row.pinned is False
    assert old_row.content == _PREFIX_BASE + "Tokyo", "audit trail keeps the stale text"


def test_superseded_row_excluded_from_search_cap_and_decay(store):
    old = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.LLM_INFERENCE,
        pinned=True,
        last_used_turn_index=1,
    )
    payload = _llm2_payload([], corrections=[{"replaces": "M1", "content": _PREFIX_BASE + "Busan"}])
    _run_summarizer(store, payload)
    new = next(r for r in store.list(conversation_id="c") if r.content == _PREFIX_BASE + "Busan")

    # store.search: the exact-text query would top-rank the frozen row
    # (fake embeddings hash the text) if it weren't excluded.
    ids = [e.id for e, _ in store.search(_PREFIX_BASE + "Tokyo", conversation_id="c", top_k=10)]
    assert old.id not in ids
    assert new.id in ids

    # decay: frozen — an unpinned FRESH row would otherwise go WARM.
    DecayEngine(store).step("c", current_turn_index=50)
    assert store.get(old.id).state == MemoryState.FRESH
    assert store.get(old.id).superseded_by == new.id

    # cap_pinned: even a stale pin flag on a superseded row neither counts
    # against the cap nor lands in the demotion candidates.
    store.pin(old.id, True)
    demoted = store.cap_pinned("c", max_pinned=1)
    assert demoted == 0
    assert store.get(old.id).pinned is True, "frozen row is not a demotion candidate"


def test_dedup_never_merges_into_superseded_row(store):
    old = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        last_used_turn_index=1,
    )
    payload = _llm2_payload([], corrections=[{"replaces": "M1", "content": _PREFIX_BASE + "Busan"}])
    _run_summarizer(store, payload)
    new = next(r for r in store.list(conversation_id="c") if r.content == _PREFIX_BASE + "Busan")

    # Exact re-add of the corrected text merges into the NEW row.
    merged = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=9,
    )
    assert merged.id == new.id

    # Re-adding the STALE text: its exact hash lives only on the frozen row,
    # which every dedup scan now skips — the merge lands on the new row
    # (60-char-prefix path), never resurrecting the superseded one.
    res = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=9,
    )
    assert res.id != old.id
    assert res.id == new.id
    frozen = store.get(old.id)
    assert frozen.use_count == 0, "frozen row was never touched by dedup"
    assert frozen.superseded_by == new.id


# ---------- (e) corrections omitted / unknown id → no-op --------------------


def test_corrections_omitted_is_noop(store):
    old = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        last_used_turn_index=1,
    )
    _run_summarizer(store, _llm2_payload([]))
    assert all(not r.superseded_by for r in store.list(conversation_id="c"))
    old_row = store.get(old.id)
    assert old_row.pinned is True
    assert old_row.content == _PREFIX_BASE + "Tokyo"


def test_unknown_correction_id_silently_ignored(store):
    old = store.add(
        conversation_id="c",
        content=_PREFIX_BASE + "Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        last_used_turn_index=1,
    )
    payload = _llm2_payload([], corrections=[{"replaces": "M7", "content": "ghost fact"}])
    _run_summarizer(store, payload)
    assert not any(r.content == "ghost fact" for r in store.list(conversation_id="c"))
    old_row = store.get(old.id)
    assert old_row.superseded_by is None
    assert old_row.pinned is True


# ---------- (f) D4: memory tool visibility ----------------------------------


def test_memory_tool_marks_superseded_and_pinned_excludes(store):
    old = store.add(
        conversation_id="c",
        content="User lives in Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
    )
    new = store.add(
        conversation_id="c",
        content="User lives in Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        pinned=True,
        dedup=False,
    )
    store.supersede(old.id, new.id)

    # Deliberate entity recall may surface the frozen row — marked. (The
    # marker lives in _entry_to_dict, the serializer lookup shares.)
    hits = memory_entity("Tokyo", store=store, conversation_id="c")
    marked = [h for h in hits if h["id"] == old.id]
    assert marked
    assert marked[0]["content"] == "User lives in Tokyo (superseded)"

    # lookup (hybrid RAG path) never surfaces the frozen row at all.
    hybrid = HybridSearch(store=store)
    res = memory_lookup("User lives in Tokyo", store=store, hybrid=hybrid, conversation_id="c")
    assert res
    assert all(r["id"] != old.id for r in res)

    # pinned excludes superseded rows even when a stale pin flag survives.
    store.pin(old.id, True)
    pinned_ids = [p["id"] for p in memory_pinned(store=store, conversation_id="c")]
    assert old.id not in pinned_ids
    assert new.id in pinned_ids


# ---------- (g) migration ----------------------------------------------------


def test_run_migrations_adds_superseded_by(tmp_path):
    """Pre-v1.0 memory_entry tables gain the column without crashing."""
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
    assert "memory_entry.superseded_by" in added, f"added={added}"
    with eng.begin() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(memory_entry)"))]
    assert "superseded_by" in cols
