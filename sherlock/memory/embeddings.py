"""Embedding provider abstraction.

litellm covers OpenAI / Cohere / Voyage / BAAI/bge-m3 etc. via one call. The
abstraction also has a deterministic FakeEmbeddingProvider for hermetic tests.
"""
from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod

import litellm

litellm.suppress_debug_info = True


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider: str, model: str, api_key_env: str | None = None) -> None:
        self._provider = provider.lower()
        self._model = model
        if api_key_env:
            key = os.environ.get(api_key_env)
            if key:
                canonical = {
                    "openai": "OPENAI_API_KEY",
                    "cohere": "COHERE_API_KEY",
                    "voyage": "VOYAGE_API_KEY",
                }.get(self._provider)
                if canonical and not os.environ.get(canonical):
                    os.environ[canonical] = key
        self._dim_cached: int | None = None

    def _model_id(self) -> str:
        if self._provider == "openai":
            return self._model
        return f"{self._provider}/{self._model}"

    @property
    def dimension(self) -> int:
        if self._dim_cached is None:
            # Trigger a cheap sample to learn the dimension.
            v = self.embed_one("dimension probe")
            self._dim_cached = len(v)
        return self._dim_cached

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = litellm.embedding(model=self._model_id(), input=texts)
        return [item["embedding"] for item in resp["data"]]


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing-based 'embeddings' for hermetic tests.

    Not semantically meaningful; just stable per text so search/decay logic
    can be exercised without network.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch the 32-byte digest into self._dim floats in [-1, 1]
        out: list[float] = []
        for i in range(self._dim):
            b = h[i % 32]
            out.append((b / 127.5) - 1.0)
        # L2-normalise for cosine similarity to behave nicely.
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vec(t) for t in texts]


def build_embedding_provider(config) -> EmbeddingProvider:
    """Construct an embedding provider from config.storage.embedding (or fake)."""
    if config is None:
        return FakeEmbeddingProvider()
    prov = (config.provider or "").lower()
    if prov in {"fake", "test", ""}:
        return FakeEmbeddingProvider()
    return LiteLLMEmbeddingProvider(
        provider=prov,
        model=config.model,
        api_key_env=getattr(config, "api_key_env", None),
    )
