"""v0.5.0 Phase 3 — true background execution."""

from __future__ import annotations

import json
import threading
import time

from sherlock import Sherlock


def _infer_json():
    return json.dumps(
        {
            "hypotheses": [
                {
                    "intent": "bg intent",
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
            "context_to_expand": [],
            "context_to_exclude": [],
            "freshness_required": [],
            "confidence_overall": 0.6,
            "evolution_signals": {},
        }
    )


def test_background_returns_before_companions_finish(tmp_path):
    """With background=True, chat() returns the main reply BEFORE the slow
    companion work completes.
    """
    infer_started = threading.Event()
    infer_release = threading.Event()

    def main(messages):
        return "fast reply.\n<<sherlock-companions: infer>>"

    def inference(messages):
        infer_started.set()
        infer_release.wait(timeout=5)  # block until the test releases
        return _infer_json()

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
        background=True,
    )
    agent._turn_index = 10

    t0 = time.time()
    reply = agent.chat("hello")
    elapsed = time.time() - t0

    # Main reply returned quickly even though inference is still blocked.
    assert "fast reply" in reply
    assert infer_started.wait(timeout=2), "background inference never started"
    assert elapsed < 2.0, f"chat() blocked on background work ({elapsed:.1f}s)"

    # Now release the companion and drain.
    infer_release.set()
    agent.drain()
    state = agent.inspect_last_turn()
    assert state.hypotheses, "background hypotheses never landed after drain"


def test_background_pending_carries_after_drain(tmp_path):
    """Background-produced pending context must reach the next turn's slot."""
    systems: list[str] = []
    calls = {"n": 0}

    def main(messages):
        calls["n"] += 1
        if messages and messages[0]["role"] == "system" and "TIER 1" in messages[0]["content"]:
            # v1.4: the carried hypothesis rides the FINAL user message now.
            systems.append(messages[-1]["content"])
        # First LLM-1 turn requests infer.
        if calls["n"] == 1:
            return "r1.\n<<sherlock-companions: infer>>"
        return "r2."

    def inference(messages):
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "intent": "BG_CARRY_MARKER",
                        "probability": 0.7,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "abduction",
                    },
                    {
                        "intent": "x",
                        "probability": 0.2,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "deduction",
                    },
                    {
                        "intent": "y",
                        "probability": 0.1,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "pragmatic",
                    },
                ],
                "tools_recommended": [],
                "context_to_expand": [],
                "context_to_exclude": [],
                "freshness_required": [],
                "confidence_overall": 0.7,
                "evolution_signals": {},
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
        background=True,
    )
    agent._turn_index = 10
    agent.chat("turn one")
    agent.drain()  # ensure bg finished + pending set
    agent.chat("turn two")  # next-turn slot should carry it

    assert any(
        "BG_CARRY_MARKER" in s for s in systems[1:]
    ), "background hypothesis did not carry into the next slot"


def test_rapid_turns_do_not_crash(tmp_path):
    """Sending several turns back-to-back with background on must not crash
    (bounded-wait + lock serialise memory access)."""

    def main(messages):
        return "ok.\n<<sherlock-companions: compact, infer>>"

    def companion(messages):
        # Valid JSON for both summary + inference shapes (inference path
        # ignores unknown keys; summary path reads facts).
        return json.dumps(
            {
                "summary": "s",
                "facts": [],
                "topic_label": "t",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
                "persona_summary": "p",
                "predicted_directions": [],
                "worth_digging": [],
                "hypotheses": [
                    {
                        "intent": "a",
                        "probability": 0.5,
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
                        "probability": 0.2,
                        "evidence": [],
                        "search_keywords": [],
                        "reasoning_type": "pragmatic",
                    },
                ],
                "tools_recommended": [],
                "context_to_expand": [],
                "context_to_exclude": [],
                "freshness_required": [],
                "confidence_overall": 0.5,
                "evolution_signals": {},
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=companion,
        inference_chat=companion,
        system_prompt="x",
        storage_dir=tmp_path,
        background=True,
    )
    agent._turn_index = 10
    for i in range(6):
        reply = agent.chat(f"message {i}")
        assert "ok" in reply
    agent.drain()
    # Memory survived the concurrent writes.
    assert agent.memory.list(conversation_id=agent.conversation_id)


def test_inline_mode_unchanged(tmp_path):
    """background=False keeps the deterministic inline path: hypotheses are
    present immediately after chat() returns (no drain needed)."""

    def main(messages):
        return "ok.\n<<sherlock-companions: infer>>"

    def inference(messages):
        return _infer_json()

    agent = Sherlock.with_callable(
        main_chat=main,
        inference_chat=inference,
        system_prompt="x",
        storage_dir=tmp_path,
        background=False,
    )
    agent._turn_index = 10
    agent.chat("hello")
    state = agent.inspect_last_turn()
    assert state.hypotheses, "inline mode should populate hypotheses synchronously"
