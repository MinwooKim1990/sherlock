"""LLM-2 persona summary + predictions end-to-end (v0.4.0)."""

from __future__ import annotations

import json

from sherlock import Sherlock


def _llm2_response(persona: str, predictions: list[dict]) -> str:
    return json.dumps(
        {
            "summary": "compact test summary",
            "facts": [],
            "topic_label": "test",
            "topic_changed_from_previous": False,
            "retrieval_keywords": [],
            "persona_summary": persona,
            "predicted_directions": predictions,
            "worth_digging": [],
        }
    )


def test_persona_summary_appears_in_next_slot(tmp_path):
    seen_systems: list[str] = []
    call_count = {"main": 0}

    def main(messages):
        seen_systems.append(messages[0]["content"])
        call_count["main"] += 1
        # Tag-gate compact on turn 1 only.
        if call_count["main"] == 1:
            return "first reply.\n<<sherlock-companions: compact>>"
        return "second reply."

    def summary(messages):
        return _llm2_response(
            persona="The user prefers terse, technical replies. Active project: Nimbus.",
            predictions=[],
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="ROLE: helpful.",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10  # bypass cold-start so compact actually fires
    agent.chat("hi")
    agent.chat("again")

    # The second system prompt should contain the persona summary.
    assert any(
        "PERSONA SUMMARY" in s for s in seen_systems[1:]
    ), f"persona summary not injected: {seen_systems}"
    assert any("Nimbus" in s for s in seen_systems[1:])


def test_predictions_persist_when_confidence_above_threshold(tmp_path):
    def main(messages):
        return "ok.\n<<sherlock-companions: compact>>"

    def summary(messages):
        return _llm2_response(
            persona="...",
            predictions=[
                {
                    "direction": "user will switch topics to sleep",
                    "confidence": 0.8,
                    "evidence": ["yawn"],
                },
                {"direction": "user might mention weather", "confidence": 0.3, "evidence": []},
            ],
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10
    agent.chat("hello")

    preds = agent._fetch_recent_llm2_predictions(agent.conversation_id, limit=10)
    # Only the 0.8-confidence prediction should make it through.
    directions = [p["direction"] for p in preds]
    assert any("sleep" in d for d in directions)
    assert not any("weather" in d for d in directions)


def test_persona_summary_excluded_from_pinned_facts_block(tmp_path):
    seen_systems: list[str] = []

    def main(messages):
        seen_systems.append(messages[0]["content"])
        return "ok.\n<<sherlock-companions: compact>>"

    def summary(messages):
        return _llm2_response(
            persona="UNIQUE_PERSONA_MARKER_X",
            predictions=[],
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent._turn_index = 10
    agent.chat("a")
    agent.chat("b")

    # Persona summary appears ONCE in PERSONA SUMMARY block, NOT in PINNED FACTS.
    last_sys = seen_systems[-1]
    assert "UNIQUE_PERSONA_MARKER_X" in last_sys
    # The persona-summary entry has pinned=True + tag=persona_summary;
    # _format_pinned_block filters it out. So the PINNED FACTS section
    # (if present) must not contain the marker.
    if "PINNED FACTS" in last_sys:
        pinned_section = last_sys.split("PINNED FACTS", 1)[1].split("[", 1)[0]
        assert "UNIQUE_PERSONA_MARKER_X" not in pinned_section


def test_persona_summary_not_dedup_merged_into_plain_summary(tmp_path, require_local_embeddings):
    """Regression (v0.5.0 hands-on review): with real embeddings the persona
    summary text is ~identical to the plain summary, which previously caused
    store.add's fuzzy dedup to merge them and strip the persona_summary tag.
    The persona must survive as its own tagged, pinned entry.
    """

    def main(messages):
        return "ok.\n<<sherlock-companions: compact>>"

    def summary(messages):
        return json.dumps(
            {
                "summary": "Jiwon, Seoul designer; Yujin(5) peanut allergy",
                "facts": [],
                "topic_label": "p",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
                # near-identical to `summary` above on purpose:
                "persona_summary": "Jiwon: Seoul designer, parent of Yujin (peanut allergy).",
                "predicted_directions": [],
                "worth_digging": [],
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="local",
    )
    agent._turn_index = 10
    agent.chat("hi")
    block = agent._format_persona_summary_block(agent.conversation_id)
    assert "PERSONA SUMMARY" in block
    assert "parent of Yujin" in block


def test_worth_digging_persisted(tmp_path):
    """v0.6: LLM-2 worth_digging threads (previously discarded) are now persisted
    as retrievable INFERENCE memories tagged 'worth_digging'."""
    import json as _json

    def main(messages):
        return "ok.\n<<sherlock-companions: compact>>"

    def summary(messages):
        return _json.dumps(
            {
                "summary": "s",
                "facts": [],
                "topic_label": "t",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
                "persona_summary": "p",
                "predicted_directions": [],
                "worth_digging": [
                    {
                        "topic": "the user's pending job change",
                        "reason": "mentioned then dropped",
                        "confidence": 0.8,
                    }
                ],
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main, summary_chat=summary, system_prompt="x", storage_dir=tmp_path
    )
    agent._turn_index = 10
    agent.chat("hi")
    agent.drain()
    mems = agent.memory.list(conversation_id=agent.conversation_id)
    wd = [m for m in mems if "worth_digging" in (m.tags or "")]
    assert wd, "worth_digging thread should be persisted"
    assert "job change" in wd[0].content
