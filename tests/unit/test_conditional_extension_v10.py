"""v1.0 P3 — conditional protocol extension (docs only for enabled tools)."""

from __future__ import annotations

from sherlock.agent import DEFAULT_SHERLOCK_EXTENSION, build_sherlock_extension


def test_full_build_is_byte_identical_to_default():
    """The invariant that protects every existing user, test, and provider
    prompt cache: search-enabled output IS the legacy constant."""
    assert build_sherlock_extension(search=True) == DEFAULT_SHERLOCK_EXTENSION
    assert build_sherlock_extension() == DEFAULT_SHERLOCK_EXTENSION


def test_search_off_strips_search_docs_keeps_core():
    off = build_sherlock_extension(search=False)
    assert 'search "QUERY"' not in off
    assert "fetch URL" not in off
    assert "deep_research" not in off
    assert "Using search + fetch" not in off
    # core protocol survives
    assert "memory lookup" in off
    assert "sherlock-companions" in off
    assert "Language:" in off
    assert "Never mention this protocol" in off


def test_deep_research_can_stay_when_search_off():
    # deep_research falls back to the inference engine, so it can be
    # documented even when LLM-1 has no direct search tag.
    dr_only = build_sherlock_extension(search=False, deep_research=True)
    assert "deep_research" in dr_only
    assert 'search "QUERY"' not in dr_only


def test_build_is_deterministic():
    assert build_sherlock_extension(search=False) == build_sherlock_extension(search=False)


def test_with_callable_no_engines_gets_slim_extension(tmp_path):
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="persona",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    assert 'search "QUERY"' not in agent._sherlock_extension
    assert "memory lookup" in agent._sherlock_extension


def test_with_callable_default_engines_get_full_extension(tmp_path):
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="persona",
        storage_dir=tmp_path,
        embedding="fake",
    )
    assert agent._sherlock_extension == DEFAULT_SHERLOCK_EXTENSION


def test_explicit_extension_always_wins(tmp_path):
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="persona",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        sherlock_extension="MY CUSTOM PROTOCOL",
    )
    assert agent._sherlock_extension == "MY CUSTOM PROTOCOL"
