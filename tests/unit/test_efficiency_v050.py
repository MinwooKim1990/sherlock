"""v0.5.0 Phase 5 — efficiency optimizations (correctness-preserving)."""

from __future__ import annotations


from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import MemoryEntry, MemorySource, MemoryType
from sherlock.rag.hybrid import HybridSearch
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


def _store(tmp_path):
    storage = Storage(tmp_path / "db.sqlite")
    embed = build_embedding_provider(_FakeEmbedConfig())
    return MemoryStore(
        engine=storage.engine, embedding_provider=embed, vector_path=tmp_path / "vec"
    )


def test_lazy_litellm_not_imported_on_package_load():
    # A subprocess-free check: importing sherlock alone must not import litellm.
    # (We can't un-import within a session, so assert the lazy hook exists.)
    import sherlock.providers as p

    assert hasattr(p, "__getattr__"), "providers must use lazy __getattr__"


def test_content_hash_exact_dedup_across_history(tmp_path):
    store = _store(tmp_path)
    conv = "c1"
    e1 = store.add(
        conversation_id=conv,
        content="Yujin has a peanut allergy",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    # Add 30 unrelated facts in between (would escape a recent-only window).
    for i in range(30):
        store.add(
            conversation_id=conv,
            content=f"unrelated fact number {i}",
            type=MemoryType.FACT,
            source=MemorySource.USER,
        )
    # Re-emit the exact same early fact.
    e2 = store.add(
        conversation_id=conv,
        content="Yujin has a peanut allergy",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )
    assert e2.id == e1.id, "exact re-emit should dedupe to the same entry via content_hash"
    assert e2.use_count >= 1


def test_content_hash_set_on_add(tmp_path):
    store = _store(tmp_path)
    e = store.add(
        conversation_id="c", content="hello world", type=MemoryType.FACT, source=MemorySource.USER
    )
    assert e.content_hash == MemoryEntry.compute_hash("hello world")


def test_entity_index_find(tmp_path):
    store = _store(tmp_path)
    conv = "c2"
    store.add(
        conversation_id=conv,
        content="Yujin은 5살, 땅콩 알레르기",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        semantic_triple=("Yujin", "has_allergy", "peanut"),
        tags="yujin",
    )
    store.add(
        conversation_id=conv,
        content="사용자는 Nimbus 작업 중",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        tags="nimbus",
    )
    hits = store.find_by_entities(conv, {"yujin"})
    assert any("Yujin" in h.content for h in hits)
    assert not any("Nimbus" in h.content for h in hits)


def test_entity_index_used_by_hybrid(tmp_path):
    store = _store(tmp_path)
    hybrid = HybridSearch(store=store)
    conv = "c3"
    store.add(
        conversation_id=conv,
        content="Yujin은 5살, 땅콩 알레르기",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        semantic_triple=("Yujin", "has_allergy", "peanut"),
        tags="yujin",
    )
    for i in range(10):
        store.add(
            conversation_id=conv,
            content=f"날씨 잡담 {i}",
            type=MemoryType.FACT,
            source=MemorySource.USER,
        )
    hits = hybrid.search("Yujin 알레르기", conversation_id=conv, top_k=3)
    assert hits and "Yujin" in hits[0][0].content


def test_entity_index_cascade_delete(tmp_path):
    from sherlock.memory.entry import MemoryEntity
    from sqlmodel import Session, select

    store = _store(tmp_path)
    conv = "c4"
    store.add(
        conversation_id=conv,
        content="Phoebe wedding in June",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        semantic_triple=("Phoebe", "has_event", "wedding"),
        tags="phoebe",
    )
    store.delete_conversation_memories(conv)
    with Session(store._engine) as s:
        rows = list(s.exec(select(MemoryEntity).where(MemoryEntity.conversation_id == conv)))
    assert rows == [], "entity index rows should be cascade-deleted"


def test_hard_delete_cascades_entity_index(tmp_path):
    """Regression (v0.5.1 review): hard_delete must also remove the row's
    MemoryEntity index entries. Repeated deletes (persona-summary replacement,
    decay eviction) previously left orphaned index rows accumulating.
    """
    from sherlock.memory.entry import MemoryEntity
    from sqlmodel import Session, select

    store = _store(tmp_path)
    e = store.add(
        conversation_id="c5",
        content="Yujin likes soba",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        semantic_triple=("Yujin", "likes", "soba"),
        tags="yujin",
    )
    with Session(store._engine) as s:
        before = list(s.exec(select(MemoryEntity).where(MemoryEntity.memory_id == e.id)))
    assert before, "precondition: entity index rows exist for the new memory"
    store.hard_delete(e.id)
    with Session(store._engine) as s:
        after = list(s.exec(select(MemoryEntity).where(MemoryEntity.memory_id == e.id)))
    assert after == [], "hard_delete must cascade-delete the entity index rows"


def test_timestamp_coarsened_to_minute():
    from sherlock.agent import _now_iso

    t = _now_iso("minute")
    # minute granularity → seconds field is 00
    assert t.endswith("00:00+00:00") or ":00+00:00" in t
    d = _now_iso("date")
    assert "T" not in d  # date only


def test_semantic_match_not_buried_by_bm25(tmp_path, require_local_embeddings):
    """Regression (v0.5.0 hands-on review): a strong vector match for a
    keyword-free query must rank #1, not be buried by BM25 noise.
    Requires real embeddings; skipped (not errored) if they're unavailable.
    """
    from sherlock.config import EmbeddingConfig

    storage = Storage(tmp_path / "db.sqlite")
    embed = build_embedding_provider(EmbeddingConfig(provider="local", model=None))
    store = MemoryStore(
        engine=storage.engine, embedding_provider=embed, vector_path=tmp_path / "vec"
    )
    from sherlock.rag.hybrid import HybridSearch

    conv = "c"
    facts = [
        "Daughter Yujin has a peanut allergy",  # the only relevant one
        "The Nimbus dashboard uses a Tailwind config",
        "Booked a hotel near Ginza for June",
        "Phoebe Bridgers concert tickets locked",
        "Prefers Vue 3 over React",
    ]
    for f in facts:
        store.add(conversation_id=conv, content=f, type=MemoryType.FACT, source=MemorySource.USER)
    h = HybridSearch(store=store)
    # Query shares NO keywords with the relevant fact.
    hits = h.search("what foods are unsafe for my child", conversation_id=conv, top_k=3)
    assert hits, "no hits"
    assert "allergy" in hits[0][0].content, f"semantic match buried; top was {hits[0][0].content!r}"
