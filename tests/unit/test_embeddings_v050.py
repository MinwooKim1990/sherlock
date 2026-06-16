"""v0.5.0 Phase 2 — hybrid embeddings + collection namespacing."""

from __future__ import annotations

from sherlock.config import EmbeddingConfig
from sherlock.memory.embeddings import (
    FakeEmbeddingProvider,
    LocalEmbeddingProvider,
    build_embedding_provider,
)


def test_fake_collection_signature():
    p = FakeEmbeddingProvider(dim=64)
    assert p.collection_signature == "fake-64"


def test_build_fake_when_provider_fake():
    p = build_embedding_provider(EmbeddingConfig(provider="fake"))
    assert isinstance(p, FakeEmbeddingProvider)


def test_local_signature_is_model_specific():
    p = LocalEmbeddingProvider(model="BAAI/bge-small-en-v1.5")
    assert p.collection_signature.startswith("local-")
    assert "bge-small" in p.collection_signature


def test_build_local_falls_back_to_fake_when_unavailable(monkeypatch):
    """If fastembed import fails, build must fall back to fake (with warning),
    not crash — keeps offline/CI runs working.
    """
    import sherlock.memory.embeddings as emb

    class _Boom(emb.LocalEmbeddingProvider):
        @property
        def dimension(self):  # force the probe to fail
            raise RuntimeError("simulated missing fastembed")

    monkeypatch.setattr(emb, "LocalEmbeddingProvider", _Boom)
    p = build_embedding_provider(EmbeddingConfig(provider="local"))
    assert isinstance(p, FakeEmbeddingProvider)


def test_local_ignores_fake_sentinel_model(monkeypatch):
    """Regression (v0.5.0 review): EmbeddingConfig's default model is the
    fake sentinel 'fake-embedding'. With provider=local that must mean 'use
    the local default model', NOT literally load a fastembed model called
    'fake-embedding' (which fails → silent fake fallback, defeating `local`).
    Hermetic: we capture the model arg without loading any real weights.
    """
    import sherlock.memory.embeddings as emb

    captured = {}

    class _Spy(emb.LocalEmbeddingProvider):
        def __init__(self, model=None):
            captured["model"] = model
            super().__init__(model=model)

        @property
        def dimension(self):  # avoid touching fastembed in the probe
            return 384

    monkeypatch.setattr(emb, "LocalEmbeddingProvider", _Spy)
    # provider=local, model left at the EmbeddingConfig default ('fake-embedding')
    p = build_embedding_provider(EmbeddingConfig(provider="local"))
    assert isinstance(p, _Spy), "should build a local provider, not fall back to fake"
    assert captured["model"] is None, (
        f"fake sentinel must be normalized to None (local default), " f"got {captured['model']!r}"
    )


def test_collection_namespaced_by_embedder(tmp_path):
    """Two stores with different embedders must use different Chroma
    collections so dims never collide.
    """
    from sherlock.memory import MemoryStore
    from sherlock.storage import Storage

    storage = Storage(tmp_path / "db.sqlite")
    fake64 = FakeEmbeddingProvider(dim=64)
    fake128 = FakeEmbeddingProvider(dim=128)
    s1 = MemoryStore(engine=storage.engine, embedding_provider=fake64, vector_path=tmp_path / "vec")
    s2 = MemoryStore(
        engine=storage.engine, embedding_provider=fake128, vector_path=tmp_path / "vec"
    )
    assert s1._collection.name != s2._collection.name
    assert "fake-64" in s1._collection.name
    assert "fake-128" in s2._collection.name


def test_local_embeddings_semantic_quality(require_local_embeddings):
    """Cross-lingual semantic match should beat unrelated text — proves the
    local embedder is real (not hash noise like fake).
    """
    import math

    def cos(a, b):
        n = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        return n / (da * db) if da and db else 0.0

    p = LocalEmbeddingProvider()
    v = p.embed(
        [
            "my daughter has a peanut allergy",
            "딸이 땅콩 알레르기가 있어요",
            "the stock market crashed today",
        ]
    )
    related = cos(v[0], v[1])
    unrelated = cos(v[0], v[2])
    assert related > unrelated + 0.3, f"semantic gap too small: {related} vs {unrelated}"
    assert p.dimension == len(v[0])
