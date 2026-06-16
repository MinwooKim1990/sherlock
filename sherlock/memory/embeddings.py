"""Embedding provider abstraction.

Three providers:
- `LocalEmbeddingProvider` (v0.5.0) — fastembed (ONNX, no torch). Default
  realistic embedder; multilingual small model so Korean+English both work,
  no API key, runs offline after the one-time model download.
- `LiteLLMEmbeddingProvider` — OpenAI / Cohere / Voyage / bge via litellm
  (requires a key). litellm is imported lazily so `import sherlock` stays fast.
- `FakeEmbeddingProvider` — deterministic hash vectors for hermetic tests.

Each provider exposes `collection_signature` so the vector store can
namespace its Chroma collection by embedder (different embedders have
different dimensions; mixing them in one collection corrupts search).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from abc import ABC, abstractmethod


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40] or "x"


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def collection_signature(self) -> str:
        """Stable short id used to namespace the vector collection.

        Must NOT require a network call / model download to compute —
        it's used at store-construction time.
        """
        return self.__class__.__name__.lower()


class LocalEmbeddingProvider(EmbeddingProvider):
    """fastembed-backed local embeddings (ONNX). No API key, offline after
    the first model download. Multilingual default for KO/EN mixing.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self._model = None  # lazy
        self._dim_cached: int | None = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # lazy; optional dependency

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim_cached is None:
            self._dim_cached = len(self.embed_one("dimension probe"))
        return self._dim_cached

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        out: list[list[float]] = []
        for vec in model.embed(list(texts)):
            # fastembed yields numpy arrays; convert to plain floats.
            out.append([float(x) for x in vec])
        return out

    @property
    def collection_signature(self) -> str:
        return f"local-{_slug(self._model_name)}"


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
            v = self.embed_one("dimension probe")
            self._dim_cached = len(v)
        return self._dim_cached

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import litellm  # lazy — keeps `import sherlock` fast + offline-friendly

        litellm.suppress_debug_info = True
        resp = litellm.embedding(model=self._model_id(), input=texts)
        return [item["embedding"] for item in resp["data"]]

    @property
    def collection_signature(self) -> str:
        return f"{_slug(self._provider)}-{_slug(self._model)}"


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
        out: list[float] = []
        for i in range(self._dim):
            b = h[i % 32]
            out.append((b / 127.5) - 1.0)
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vec(t) for t in texts]

    @property
    def collection_signature(self) -> str:
        return f"fake-{self._dim}"


def build_embedding_provider(config) -> EmbeddingProvider:
    """Construct an embedding provider from `config.storage.embedding`.

    - provider in {fake, test} → FakeEmbeddingProvider
    - provider in {auto, "", local, fastembed} → LocalEmbeddingProvider
      (graceful fallback to fake + stderr warning if fastembed/model
      unavailable — keeps offline/CI runs working). `auto` is the
      "just works" default: real semantic memory when the extra is
      installed, deterministic fake otherwise.
    - otherwise → LiteLLMEmbeddingProvider (openai/cohere/voyage/…)
    """
    if config is None:
        return FakeEmbeddingProvider()
    prov = (config.provider or "").lower()
    # `auto` = the "just works" default: real local semantic memory, falling
    # back to fake if unavailable. Resolved via env so hermetic test suites can
    # force `fake` (conftest sets SHERLOCK_AUTO_EMBEDDING=fake) WITHOUT touching
    # explicit `local`/`fake` choices.
    if prov in {"auto", ""}:
        prov = (os.environ.get("SHERLOCK_AUTO_EMBEDDING") or "local").lower()
    if prov in {"fake", "test"}:
        return FakeEmbeddingProvider()
    # The EmbeddingConfig default model is the fake sentinel "fake-embedding".
    # For a REAL provider that means "use the provider's own default model",
    # not a literal model by that name — otherwise `provider: local` with the
    # default model would try to load a fastembed model called
    # "fake-embedding", fail, and silently fall back to fake (defeating the
    # whole point of selecting `local`).
    model = getattr(config, "model", None)
    if model in (None, "", "fake-embedding"):
        model = None
    if prov in {"local", "fastembed"}:
        try:
            p = LocalEmbeddingProvider(model=model)
            # Probe once so a missing package/model fails HERE (→ fallback),
            # not deep inside the first chat turn.
            _ = p.dimension
            return p
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                f"[sherlock] local embeddings unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to FakeEmbeddingProvider. Install with "
                f"`pip install sherlock[embeddings]` for real semantic memory.",
                file=sys.stderr,
            )
            return FakeEmbeddingProvider()
    return LiteLLMEmbeddingProvider(
        provider=prov,
        model=model,
        api_key_env=getattr(config, "api_key_env", None),
    )
