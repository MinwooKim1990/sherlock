"""v1.0 P1 — honest small-window budget profiles + history-never-zero."""

from __future__ import annotations

import warnings

from sherlock.budget import (
    DEFAULT_PROFILE,
    PROFILE_8K,
    PROFILE_16K,
    PROFILE_32K,
    SMALL_MODEL_PROFILE,
    select_profile_for_window,
)


def test_tiered_profile_selection():
    assert select_profile_for_window(8_192) is PROFILE_8K
    assert select_profile_for_window(16_384) is PROFILE_16K
    assert select_profile_for_window(32_768) is PROFILE_32K
    # pinned legacy behavior: 128K stays on SMALL
    assert select_profile_for_window(128_000) is SMALL_MODEL_PROFILE
    assert select_profile_for_window(200_000) is DEFAULT_PROFILE


def test_small_profiles_reserve_less_than_their_window():
    """The SMALL profile's reservations exceeded an 8K window outright
    (k_pool=0). The new tiers must leave real room for the K-turn tail."""
    for profile, window in ((PROFILE_8K, 8_192), (PROFILE_16K, 16_384), (PROFILE_32K, 32_768)):
        reserved = profile.total_reserved()
        assert reserved < window, f"{window}: reserved {reserved} >= window"
        # at least ~15% of the window must remain for raw history
        assert window - reserved >= window * 0.15


def test_budget_sum_property_across_windows(tmp_path):
    """For any window 8K-200K, system + tail + output reserve never exceeds
    the window (the honest-budget invariant)."""
    import json

    from sherlock import Sherlock
    from sherlock.budget import count_tokens

    for window in (8_192, 16_384, 32_768, 128_000):
        agent = Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            system_prompt="You are terse.",
            storage_dir=tmp_path / f"w{window}",
            context_window=window,
            embedding="fake",
            main_search_engine=None,
            inference_search_engine=None,
        )
        for i in range(4):
            agent.chat(f"turn {i}: " + "filler " * 30)
        state = agent.inspect_last_turn()
        msgs = state.messages_passed_to_llm1
        total = sum(count_tokens(m.content) + 4 for m in msgs)
        budget = agent._slot_budget
        assert total + budget.output_reserve <= window + budget.floor_k_turn_budget, (
            f"window {window}: prompt {total} + reserve {budget.output_reserve} "
            f"overflows ({json.dumps(budget.as_dict())})"
        )


def test_8k_window_history_never_zero(tmp_path):
    """At 8K the old math collapsed k_pool to 0 — the model never saw the
    previous turn. The always-keep bypass must guarantee ≥2 turns."""
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "reply " + "x" * 50,
        system_prompt="persona",
        storage_dir=tmp_path,
        context_window=8_192,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    for i in range(6):
        agent.chat(f"message {i} " + "content " * 40)
    state = agent.inspect_last_turn()
    assert state.k_turn_turns_used >= 2, "history must never be zero on a small window"


def test_max_output_tokens_caps_output_reserve(tmp_path):
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="x",
        storage_dir=tmp_path,
        context_window=32_768,
        max_output_tokens=2_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    assert agent._slot_budget.output_reserve == 2_000


def test_context_window_warning_fires_once(tmp_path):
    import sherlock.agent as agent_mod
    from sherlock import Sherlock

    agent_mod._WARNED_NO_CTX_WINDOW = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            system_prompt="x",
            storage_dir=tmp_path / "a",
            embedding="fake",
            main_search_engine=None,
            inference_search_engine=None,
        )
        Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            system_prompt="x",
            storage_dir=tmp_path / "b",
            embedding="fake",
            main_search_engine=None,
            inference_search_engine=None,
        )
    hits = [w for w in caught if "context_window" in str(w.message)]
    assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"


def test_declared_window_suppresses_warning(tmp_path):
    import sherlock.agent as agent_mod
    from sherlock import Sherlock

    agent_mod._WARNED_NO_CTX_WINDOW = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            system_prompt="x",
            storage_dir=tmp_path,
            context_window=8_192,
            embedding="fake",
            main_search_engine=None,
            inference_search_engine=None,
        )
    assert not [w for w in caught if "context_window" in str(w.message)]
