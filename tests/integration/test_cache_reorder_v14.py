"""v1.4 — cache-optimal slot reorder + fill-based compaction + LLM-2→LLM-3 cascade.

The volatile this-turn block (inference + search) moved OUT of the system message
to the FINAL user message, so [system + conversation history] is one cacheable
prefix and only the last user message pays full price. Compaction now auto-fires
on context FILL (≥ memory.compact_at_fill_ratio), not a fixed turn cadence; and
LLM-2 can trigger LLM-3 when it surfaces worth_digging threads. Region labels keep
a small LLM-1 from confusing protocol / prior conversation / this-turn analysis /
the user's actual words.
"""

from __future__ import annotations

import json

import pytest

from sherlock import Sherlock

_HYP = json.dumps(
    {
        "hypotheses": [
            {
                "intent": "INTENT_MARK",
                "probability": 0.6,
                "evidence": [],
                "search_keywords": [],
                "reasoning_type": "abduction",
            },
            {
                "intent": "b",
                "probability": 0.3,
                "evidence": [],
                "search_keywords": [],
                "reasoning_type": "deduction",
            },
            {
                "intent": "c",
                "probability": 0.1,
                "evidence": [],
                "search_keywords": [],
                "reasoning_type": "pragmatic",
            },
        ],
        "tools_recommended": [],
        "freshness_required": [],
        "confidence_overall": 0.6,
        "evolution_signals": {},
    }
)


def _summary(**extra):
    base = {
        "summary": "s",
        "facts": [],
        "topic_label": "t",
        "topic_changed_from_previous": False,
        "retrieval_keywords": [],
        "persona_summary": "",
        "predicted_directions": [],
        "worth_digging": [],
    }
    base.update(extra)
    return json.dumps(base)


# ---- layout: volatile block rides the FINAL user message, system stays stable ----


def test_inference_rides_final_user_message_not_system(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: infer>>",
        inference_chat=lambda m: _HYP,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent._turn_index = 10
    agent.chat("a")  # infer fires post-response → pending
    agent.chat("b")  # carried into this slot
    msgs = agent.inspect_last_turn().messages_passed_to_llm1
    system = msgs[0].content
    final_user = msgs[-1].content
    # inference + this-turn analysis are in the FINAL user message, never the system
    assert "INFERENCE HYPOTHESES" in final_user
    assert "INTENT_MARK" in final_user
    assert "INFERENCE HYPOTHESES" not in system
    assert "CURRENT TIME" not in system
    # region fences are present and unambiguous
    assert "SYSTEM ANALYSIS FOR THIS TURN" in final_user
    assert "THE USER'S ACTUAL MESSAGE" in final_user
    assert final_user.rstrip().endswith("b")  # the user's real words are LAST


def test_system_message_is_fully_cacheable(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="ROLE: helper",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
    )
    agent.chat("hi")
    sys_msg = agent.inspect_last_turn().messages_passed_to_llm1[0]
    # whole system message is the cached prefix; breakpoint at its end
    assert sys_msg.cache_breakpoints is not None
    assert sys_msg.cache_breakpoints[-1] == len(sys_msg.content)
    # TIER-4 trailer marks where the (separate) history messages begin
    assert "TIER 4: PRIOR CONVERSATION" in sys_msg.content


def test_context_fill_line_in_final_message(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.", system_prompt="x", storage_dir=tmp_path, embedding="fake"
    )
    agent.chat("hello")
    final_user = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "CONTEXT FILL" in final_user
    assert 0.0 < agent._last_fill_ratio <= 1.0


# ---- fill-based compaction trigger (not turn cadence) ----


def test_fill_ratio_triggers_compact_without_tag(tmp_path):
    """At/above compact_at_fill_ratio, compaction auto-fires even though LLM-1
    never emits the compact tag."""
    compacts = {"n": 0}
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",  # NO compact tag
        summary_chat=lambda m: (compacts.__setitem__("n", compacts["n"] + 1) or _summary()),
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
    )
    agent._turn_index = 10  # past cold-start
    agent.config.memory.compact_at_fill_ratio = 0.01  # any non-empty prompt clears it
    agent.chat("hello")
    assert agent._last_fill_ratio > 0.01
    assert compacts["n"] >= 1, "fill≥threshold must auto-trigger compaction without a tag"


def test_low_fill_does_not_autocompact(tmp_path):
    """A normal short turn well under the threshold does NOT auto-compact — proves
    it is fill-gated, not a fixed cadence."""
    compacts = {"n": 0}
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",  # NO compact tag
        summary_chat=lambda m: (compacts.__setitem__("n", compacts["n"] + 1) or _summary()),
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
    )
    agent._turn_index = 10
    agent.config.memory.compact_at_fill_ratio = 0.99  # effectively unreachable here
    agent.chat("hello")
    agent.chat("again")
    assert compacts["n"] == 0, "below threshold + no tag → no auto-compaction"


# ---- LLM-2 → LLM-3 cascade ----


def test_worth_digging_triggers_infer(tmp_path):
    """When compaction surfaces worth_digging threads, LLM-2 triggers LLM-3 even
    though LLM-1 requested only compact."""
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: compact>>",  # compact only, NO infer
        summary_chat=lambda m: _summary(worth_digging=[{"thread": "dig here", "evidence": ["e"]}]),
        inference_chat=lambda m: _HYP,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent._turn_index = 10
    agent.chat("something worth a deeper look")
    # infer fired off the LLM-2 cascade (hypotheses produced) despite no infer tag
    assert agent.inspect_last_turn().hypotheses, "worth_digging did not cascade into infer"


@pytest.mark.asyncio
async def test_worth_digging_triggers_infer_async(tmp_path):
    """Async parity (audit P1): a compact-only achat() turn whose LLM-2 surfaces
    worth_digging must STILL cascade into LLM-3 — previously async ran infer BEFORE
    compaction, so the cascade could never fire on the async path."""

    async def _amain(messages):
        return "ok.\n<<sherlock-companions: compact>>"  # compact only, NO infer tag

    agent = Sherlock.with_callable(
        main_chat=_amain,
        summary_chat=lambda m: _summary(worth_digging=[{"thread": "dig here", "evidence": ["e"]}]),
        inference_chat=lambda m: _HYP,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent._turn_index = 10
    await agent.achat("something worth a deeper look")
    assert agent.inspect_last_turn().hypotheses, "async worth_digging did not cascade into infer"
