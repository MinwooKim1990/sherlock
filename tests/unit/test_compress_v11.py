"""R17 — optional LLMLingua-2 compression (`sherlock.compress`).

These tests run WITHOUT llmlingua installed: the module must fall back to
exactly the legacy ``text[:target_chars]`` truncation, warn once (and only
once) when compression was explicitly requested, and use a fake llmlingua
(sys.modules injection) to exercise the compression + failure paths.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from sherlock import compress

LONG_TEXT = "Boilerplate navigation header. " * 200  # ~6200 chars


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test starts with a cold singleton and a fresh warning latch."""
    monkeypatch.setattr(compress, "_compressor", None)
    monkeypatch.setattr(compress, "_warned_missing", False)


def _fake_llmlingua_module(compress_prompt_impl) -> types.ModuleType:
    mod = types.ModuleType("llmlingua")

    class PromptCompressor:
        def __init__(self, *args, **kwargs):
            pass

        def compress_prompt(self, text, **kwargs):
            return compress_prompt_impl(text, **kwargs)

    mod.PromptCompressor = PromptCompressor
    return mod


# ---------------------------------------------------------------------------
# 1. Without llmlingua: unavailable + exact legacy truncation.
# ---------------------------------------------------------------------------


def test_unavailable_falls_back_to_exact_legacy_truncation():
    assert compress.is_available() is False
    target = 100
    out = compress.maybe_compress(LONG_TEXT, target_chars=target)
    assert out == LONG_TEXT[:target]


# ---------------------------------------------------------------------------
# 2. Short text returned unchanged.
# ---------------------------------------------------------------------------


def test_short_text_returned_unchanged():
    short = "A short fetched snippet."
    out = compress.maybe_compress(short, target_chars=2500)
    assert out == short


# ---------------------------------------------------------------------------
# 3. requested=True without llmlingua warns once; second call stays silent.
# ---------------------------------------------------------------------------


def test_requested_without_llmlingua_warns_exactly_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compress.maybe_compress(LONG_TEXT, target_chars=100, requested=True)
    assert out == LONG_TEXT[:100]
    relevant = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(relevant) == 1
    assert "sherlock[compress]" in str(relevant[0].message)

    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        compress.maybe_compress(LONG_TEXT, target_chars=100, requested=True)
    assert [w for w in caught_again if issubclass(w.category, RuntimeWarning)] == []


def test_default_silent_fallback_emits_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compress.maybe_compress(LONG_TEXT, target_chars=100)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ---------------------------------------------------------------------------
# 4. Fake llmlingua: compression result used; exceptions fall back to slice.
# ---------------------------------------------------------------------------


def test_fake_llmlingua_compressed_output_is_returned(monkeypatch):
    fake = _fake_llmlingua_module(lambda text, **kw: {"compressed_prompt": "X"})
    monkeypatch.setitem(sys.modules, "llmlingua", fake)
    assert compress.is_available() is True
    out = compress.maybe_compress(LONG_TEXT, target_chars=100, query="what is sherlock")
    assert out == "X"


def test_fake_llmlingua_exception_falls_back_to_slice(monkeypatch):
    def _boom(text, **kw):
        raise RuntimeError("model exploded")

    fake = _fake_llmlingua_module(_boom)
    monkeypatch.setitem(sys.modules, "llmlingua", fake)
    out = compress.maybe_compress(LONG_TEXT, target_chars=100)
    assert out == LONG_TEXT[:100]
