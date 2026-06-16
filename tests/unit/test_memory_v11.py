"""v1.1 memory-layer roadmap items.

Covers:
  - R29: composite retrieval scoring — recency + importance boosts gated
    behind ``current_turn_index`` (default None = byte-identical legacy
    ranking); entity hits are never outranked by boosted vector hits.
  - R30: per-type result caps via ``max_per_type``.
  - R33: A-Mem-style memory links — created on add for ≥0.55-cosine
    same-conversation neighbours, readable via ``links_for``, surfaced by
    the 1-hop expansion in ``HybridSearch.search`` (gated by ``expand_links``).
  - R34: bi-temporal invalidation — ``supersede(..., turn_index=)`` sets
    ``invalid_at_turn`` and the memory-tool marker says "(superseded at tN)".
  - R31: structured memory reading — serialization includes "type",
    "created_turn", "last_used_turn".
  - migration: ``invalid_at_turn`` is added to pre-v1.1 DBs; the
    ``memory_link`` table exists on a fresh DB.

NOTE on the fake embedder: it hashes the FULL text, so two different
strings have near-zero cosine and only identical strings reach the
production link threshold (0.55). The link tests therefore insert
identical content twice with ``dedup=False`` (cosine exactly 1.0); the
production threshold is untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import MemorySource, MemoryState, MemoryType
from sherlock.rag.hybrid import HybridSearch
from sherlock.storage import Storage
from sherlock.tools.memory_tool import _entry_to_dict, memory_entity


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    storage = Storage(tmp_path / "test.db")
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=build_embedding_provider(_FakeEmbedConfig()),
        vector_path=tmp_path / "vectors",
    )


# ---------- (a) R29: recency / importance boosts ---------------------------


def test_recency_boost_flips_equal_vector_hits_but_never_entity(store):
    ent = store.add(
        conversation_id="c",
        content="Yujin has a peanut allergy",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        tags="yujin",
        last_used_turn_index=1,
    )
    old = store.add(
        conversation_id="c",
        content="project deadline reminder alpha",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=0,  # created_turn_index = 0
    )
    new = store.add(
        conversation_id="c",
        content="project deadline reminder omega",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=50,  # created_turn_index = 50
    )
    h = HybridSearch(store=store)
    q = "Yujin project deadline reminder"

    # Legacy (no current_turn_index): entity first, then the fused order
    # old-before-new — recency plays no part.
    legacy = h.search(q, conversation_id="c", top_k=5)
    assert [e.id for e, _ in legacy] == [ent.id, old.id, new.id]

    # Explicit None is the same default path — byte-identical ordering+scores.
    explicit_none = h.search(q, conversation_id="c", top_k=5, current_turn_index=None)
    assert [(e.id, s) for e, s in explicit_none] == [(e.id, s) for e, s in legacy]

    # With current_turn_index the much-more-recent entry overtakes the old
    # one, but the entity hit stays on top (boost ceiling 0.3 < 0.5).
    boosted = h.search(q, conversation_id="c", top_k=5, current_turn_index=50)
    assert [e.id for e, _ in boosted] == [ent.id, new.id, old.id]


def test_importance_boost_flips_equal_vector_hits(store):
    a = store.add(
        conversation_id="c",
        content="weekly grocery budget note one",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=0,
    )
    b = store.add(
        conversation_id="c",
        content="weekly grocery budget note two",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=0,
    )
    h = HybridSearch(store=store)
    q = "weekly grocery budget note"
    legacy = h.search(q, conversation_id="c", top_k=5)
    assert {e.id for e, _ in legacy} == {a.id, b.id}
    first, second = legacy[0][0], legacy[1][0]

    # Make the legacy-SECOND entry heavily used; same created turn, so the
    # recency boost is identical and only importance (min(use,10)*0.01)
    # differs — enough to flip the tiny fused-score gap.
    for _ in range(10):
        store.touch(second.id, 0)
    boosted = h.search(q, conversation_id="c", top_k=5, current_turn_index=0)
    assert [e.id for e, _ in boosted] == [second.id, first.id]

    # Default call STILL ignores use_count — ordering unchanged from legacy.
    again = h.search(q, conversation_id="c", top_k=5)
    assert [e.id for e, _ in again] == [first.id, second.id]


# ---------- (b) R30: per-type result caps -----------------------------------


def test_max_per_type_caps_inference_results(store):
    for i in range(4):
        store.add(
            conversation_id="c",
            content=f"likely prefers quiet cafes variant {i}",
            type=MemoryType.INFERENCE,
            source=MemorySource.LLM_INFERENCE,
            confidence=0.9,
        )
    fact = store.add(
        conversation_id="c",
        content="works from a cafe most mornings",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    h = HybridSearch(store=store)
    q = "quiet cafes mornings"

    uncapped = h.search(q, conversation_id="c", top_k=4)
    assert sum(1 for e, _ in uncapped if e.type == MemoryType.INFERENCE) > 2

    capped = h.search(q, conversation_id="c", top_k=4, max_per_type={"inference": 2})
    assert sum(1 for e, _ in capped if e.type == MemoryType.INFERENCE) == 2
    # Types absent from the dict are uncapped — the fact backfills.
    assert any(e.id == fact.id for e, _ in capped)


# ---------- (c) R33: memory links + 1-hop expansion --------------------------


def test_links_created_on_add_and_links_for(store):
    # Identical content + dedup=False → genuinely-new row whose fake-hash
    # vector has cosine exactly 1.0 with the first (≥ the 0.55 threshold).
    a = store.add(
        conversation_id="c",
        content="the quarterly report deadline is friday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    b = store.add(
        conversation_id="c",
        content="the quarterly report deadline is friday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    assert a.id != b.id
    # links_for is direction-agnostic: each side sees the other.
    assert store.links_for(b.id) == [(a.id, pytest.approx(1.0))]
    assert store.links_for(a.id) == [(b.id, pytest.approx(1.0))]

    # A dissimilar entry (fake-hash cosine ≈ 0 < 0.55) gets no link.
    c = store.add(
        conversation_id="c",
        content="completely different topic zebra",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    assert store.links_for(c.id) == []


def test_one_hop_expansion_surfaces_linked_entry(store):
    a = store.add(
        conversation_id="c",
        content="the quarterly report deadline is friday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    b = store.add(
        conversation_id="c",
        content="the quarterly report deadline is friday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    h = HybridSearch(store=store)
    q = "quarterly report deadline"

    # top_k=1 → only ONE direct hit; the linked twin arrives via expansion,
    # appended AFTER the direct hit with score = 0.3 * link_score.
    res = h.search(q, conversation_id="c", top_k=1)
    assert len(res) == 2
    assert {res[0][0].id, res[1][0].id} == {a.id, b.id}
    assert res[1][1] == pytest.approx(0.3)

    # expand_links=False disables the expansion entirely.
    res_off = h.search(q, conversation_id="c", top_k=1, expand_links=False)
    assert len(res_off) == 1

    # Expansion respects the superseded exclusion.
    direct = res_off[0][0]
    other = b if direct.id == a.id else a
    store.supersede(other.id, direct.id)
    res_sup = h.search(q, conversation_id="c", top_k=1)
    assert [e.id for e, _ in res_sup] == [direct.id]


def test_expansion_respects_forgotten_exclusion(store):
    a = store.add(
        conversation_id="c",
        content="the quarterly report deadline is friday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    b = store.add(
        conversation_id="c",
        content="the quarterly report deadline is friday",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    h = HybridSearch(store=store)
    res = h.search("quarterly report deadline", conversation_id="c", top_k=1)
    direct = res[0][0]
    other = b if direct.id == a.id else a
    store.update_state(other.id, MemoryState.FORGOTTEN)
    res2 = h.search("quarterly report deadline", conversation_id="c", top_k=1)
    assert [e.id for e, _ in res2] == [direct.id]


# ---------- (d) R34: bi-temporal invalidation --------------------------------


def test_supersede_with_turn_index_sets_invalid_at_turn_and_tool_marker(store):
    old = store.add(
        conversation_id="c",
        content="User lives in Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    new = store.add(
        conversation_id="c",
        content="User lives in Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    store.supersede(old.id, new.id, turn_index=7)
    assert store.get(old.id).invalid_at_turn == 7

    hits = memory_entity("Tokyo", store=store, conversation_id="c")
    marked = [h for h in hits if h["id"] == old.id]
    assert marked
    assert marked[0]["content"] == "User lives in Tokyo (superseded at t7)"


def test_supersede_without_turn_index_keeps_plain_marker(store):
    old = store.add(
        conversation_id="c",
        content="User lives in Tokyo",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    new = store.add(
        conversation_id="c",
        content="User lives in Busan",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        dedup=False,
    )
    store.supersede(old.id, new.id)
    assert store.get(old.id).invalid_at_turn is None
    hits = memory_entity("Tokyo", store=store, conversation_id="c")
    marked = [h for h in hits if h["id"] == old.id]
    assert marked
    assert marked[0]["content"] == "User lives in Tokyo (superseded)"


# ---------- (e) R31: structured memory reading -------------------------------


def test_entry_serialization_includes_temporal_keys(store):
    e = store.add(
        conversation_id="c",
        content="User works at Acme Corp",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        last_used_turn_index=5,
    )
    store.touch(e.id, 9)
    d = _entry_to_dict(store.get(e.id))
    assert d["type"] == "fact"
    assert d["created_turn"] == 5
    assert d["last_used_turn"] == 9
    # Existing keys unchanged.
    assert d["last_used_turn_index"] == 9
    assert d["content"] == "User works at Acme Corp"


# ---------- (f) migration -----------------------------------------------------


def test_run_migrations_adds_invalid_at_turn(tmp_path):
    """Pre-v1.1 memory_entry tables gain the column without crashing."""
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
    assert "memory_entry.invalid_at_turn" in added, f"added={added}"
    with eng.begin() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(memory_entry)"))]
    assert "invalid_at_turn" in cols


def test_memory_link_table_created_on_fresh_db(store):
    """create_all (run in MemoryStore.__init__) creates the brand-new
    memory_link table; run_migrations only ALTERs pre-existing tables."""
    tables = set(inspect(store._engine).get_table_names())
    assert "memory_link" in tables
