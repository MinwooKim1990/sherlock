"""Unit tests for the v0.4.0 slot-budget infrastructure."""

from __future__ import annotations


from sherlock.budget import (
    DEFAULT_PROFILE,
    SMALL_MODEL_PROFILE,
    apply_overrides,
    count_tokens,
    resolve_context_window,
    select_profile_for_window,
)

# -- registry --------------------------------------------------------------


def test_resolve_context_window_haiku():
    assert resolve_context_window("claude-haiku-4-5-20251001") == 200_000


def test_resolve_context_window_opus_with_provider_prefix():
    assert resolve_context_window("anthropic/claude-opus-4-7") == 1_000_000


def test_resolve_context_window_gpt4o():
    assert resolve_context_window("gpt-4o-mini") == 128_000


def test_resolve_context_window_unknown_falls_back():
    assert resolve_context_window("totally-random-model-xyz") == 128_000


def test_resolve_context_window_explicit_override():
    assert resolve_context_window("claude-haiku-4-5", override=50_000) == 50_000


# -- SlotBudget ------------------------------------------------------------


def test_default_profile_total_reserved():
    # Sum of all reservations should match what k_turn_budget subtracts.
    expected = (
        DEFAULT_PROFILE.sherlock_system_max
        + DEFAULT_PROFILE.tool_prompt_max
        + DEFAULT_PROFILE.user_system_max
        + DEFAULT_PROFILE.compacted_memory_max
        + DEFAULT_PROFILE.inference_data_max
        + DEFAULT_PROFILE.rag_max
        + DEFAULT_PROFILE.output_reserve
    )
    assert DEFAULT_PROFILE.total_reserved() == expected


def test_k_turn_budget_haiku_200k():
    # Default profile leaves ~89K for raw turns on 200K context.
    assert DEFAULT_PROFILE.k_turn_budget(200_000) == 200_000 - DEFAULT_PROFILE.total_reserved()


def test_k_turn_budget_opus_1m_is_huge():
    assert DEFAULT_PROFILE.k_turn_budget(1_000_000) > 800_000


def test_k_turn_budget_small_model_uses_floor():
    # SMALL profile reserves 47K; on a 50K window it would dip below floor
    # → floor (4K) wins.
    assert SMALL_MODEL_PROFILE.k_turn_budget(50_000) == SMALL_MODEL_PROFILE.floor_k_turn_budget


def test_select_profile_small_under_200k():
    assert select_profile_for_window(128_000) is SMALL_MODEL_PROFILE
    assert select_profile_for_window(64_000) is SMALL_MODEL_PROFILE


def test_select_profile_default_at_or_above_200k():
    assert select_profile_for_window(200_000) is DEFAULT_PROFILE
    assert select_profile_for_window(1_000_000) is DEFAULT_PROFILE


def test_apply_overrides_returns_new_instance():
    overridden = apply_overrides(DEFAULT_PROFILE, {"compacted_memory_max": 5_000})
    assert overridden is not DEFAULT_PROFILE
    assert overridden.compacted_memory_max == 5_000
    assert overridden.rag_max == DEFAULT_PROFILE.rag_max


def test_apply_overrides_ignores_unknown_keys():
    overridden = apply_overrides(DEFAULT_PROFILE, {"not_a_field": 99_999})
    assert overridden == DEFAULT_PROFILE


def test_apply_overrides_none_skipped():
    overridden = apply_overrides(DEFAULT_PROFILE, {"rag_max": None})
    assert overridden.rag_max == DEFAULT_PROFILE.rag_max


# -- count_tokens ----------------------------------------------------------


def test_count_tokens_empty():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_tokens_short_text_positive():
    assert count_tokens("hello world") >= 1


def test_count_tokens_roughly_linear():
    a = count_tokens("a" * 4000)
    b = count_tokens("a" * 8000)
    # Should scale roughly 2× — allow 30% slack for tokenizer chunking.
    assert b > a
    assert b <= a * 2 + 200
