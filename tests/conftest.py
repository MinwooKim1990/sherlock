"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

# Make sure the repo root is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@functools.lru_cache(maxsize=1)
def _local_embed_probe() -> "tuple[bool, str]":
    """Probe whether the local embedder can actually RUN — i.e. the model is
    downloadable/loadable and produces a vector — not merely whether the
    ``fastembed`` package is importable.

    The previous gate (``importlib.util.find_spec('fastembed')``) passed when
    the package was installed but the model weights were absent and the
    network was blocked, turning a skip into a hard ERROR in CI. Probing the
    real code path fixes that. Cached so the (slow) one-time model load runs
    at most once per session.
    """
    try:
        from sherlock.memory.embeddings import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()
        vecs = provider.embed(["probe"])
        if not vecs or not vecs[0]:
            return False, "local embedder returned an empty vector"
        return True, ""
    except Exception as exc:  # ImportError, model download failure, ONNX, …
        return False, f"{type(exc).__name__}: {exc}"


@pytest.fixture(autouse=True)
def _hermetic_auto_embedding(monkeypatch):
    """Keep the suite hermetic + fast: resolve the `auto` embedding default to
    `fake` during tests. Explicit `embedding="local"`/`"fake"` are unaffected —
    only the `auto` default consults this env var (see build_embedding_provider).
    """
    monkeypatch.setenv("SHERLOCK_AUTO_EMBEDDING", "fake")
    # Keep the suite deterministic: don't let the v0.6 smart auto-infer fire
    # extra LLM-3 calls in tests that aren't about inference (fake embeddings
    # make topic_changed over-trigger). Tag-driven infer still works; tests
    # that exercise auto-infer set this env to "smart"/"always" explicitly.
    monkeypatch.setenv("SHERLOCK_AUTO_INFER", "off")


@pytest.fixture
def require_local_embeddings():
    """Skip (not error) the test unless real local embeddings are usable."""
    ok, reason = _local_embed_probe()
    if not ok:
        pytest.skip(f"local embeddings unavailable: {reason}")
    return True


@pytest.fixture
def tmp_prompt(tmp_path: Path) -> Path:
    p = tmp_path / "main.md"
    p.write_text("You are a helpful test assistant.", encoding="utf-8")
    return p


@pytest.fixture
def fake_yaml(tmp_path: Path, tmp_prompt: Path) -> Path:
    yaml_path = tmp_path / "sherlock.yaml"
    yaml_path.write_text(
        f"""
project: sherlock_test
main_system_prompt:
  path: {tmp_prompt}
models:
  main:
    provider: fake
    model: echo
storage:
  sqlite_path: {tmp_path / "test.db"}
""",
        encoding="utf-8",
    )
    return yaml_path
