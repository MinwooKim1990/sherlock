"""Entity-indexed retrieval (Tier 2 of v0.4.0 memory architecture)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.rag.hybrid import HybridSearch, extract_entities, _entry_entity_pool
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


# ---------- extract_entities ---------------------------------------------


def test_extract_capitalised_english():
    out = extract_entities("Yujin loves apples")
    assert "yujin" in out


def test_extract_korean_words():
    out = extract_entities("유진은 다섯살이야")
    assert any("유진" in t for t in out)


def test_extract_mixed():
    out = extract_entities("Yujin은 한솔초 다녀")
    # Should pull both Yujin and 한솔초
    assert "yujin" in out
    assert any("한솔" in t for t in out)


def test_extract_skips_short_tokens():
    out = extract_entities("a b c")
    assert not out


# ---------- entry entity pool --------------------------------------------


def test_entry_pool_includes_triple_subject():
    from sherlock.memory.entry import MemoryEntry, MemoryType, MemorySource

    e = MemoryEntry(
        conversation_id="x",
        content="Yujin has peanut allergy",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        semantic_triple_subject="Yujin",
        semantic_triple_relation="has_allergy",
        semantic_triple_object="peanut",
        tags="medical,allergy",
    )
    pool = _entry_entity_pool(e)
    assert "yujin" in pool
    assert "peanut" in pool
    assert "medical" in pool


# ---------- HybridSearch entity-boost behavior ---------------------------


@pytest.fixture
def hybrid(tmp_path: Path):
    storage = Storage(tmp_path / "test.db")
    embed = build_embedding_provider(_FakeEmbedConfig())
    store = MemoryStore(
        engine=storage.engine,
        embedding_provider=embed,
        vector_path=tmp_path / "vectors",
    )
    conv = storage.create_conversation(project="test")
    # Yujin fact with rich entity tags:
    store.add(
        conversation_id=conv.id,
        content="Yujin은 5살, 땅콩 알레르기",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        tags="yujin,allergy",
        semantic_triple=("Yujin", "has_allergy", "peanut"),
    )
    # Unrelated noise:
    for i in range(8):
        store.add(
            conversation_id=conv.id,
            content=f"오늘 날씨 좋다 {i}",
            type=MemoryType.FACT,
            source=MemorySource.USER,
            confidence=1.0,
        )
    return HybridSearch(store=store), conv.id


def test_entity_query_returns_entity_match_first(hybrid):
    h, conv_id = hybrid
    hits = h.search("Yujin 알레르기 알려줘", conversation_id=conv_id, top_k=3)
    assert hits, "no hits"
    top_entry, _ = hits[0]
    assert "Yujin" in top_entry.content


def test_no_entity_falls_through_to_rag(hybrid):
    h, conv_id = hybrid
    # Generic non-entity query — should still return something via vector match.
    hits = h.search("날씨 어때", conversation_id=conv_id, top_k=3)
    assert hits  # vector path produced results
