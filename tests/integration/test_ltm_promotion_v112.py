"""v1.12 Stage A1 — long-term memory promotion, end-to-end via Sherlock.

Exercises the agent-level wiring the summarizer unit tests can't reach:
  - kill switch: default-off leaves the LLM-2 prompt clean and never writes
    a sentinel row;
  - promotion + restart: a promoted fact survives a fresh Sherlock reopened
    on the same storage_dir (cross-conversation durability);
  - decay isolation: sentinel rows are never touched by the per-turn decay;
  - events: memory.promoted fires (with count) on BOTH the sync chat() and the
    async achat() companion paths (parity);
  - wipe_long_term() clears the sentinel scope.

Fake-provider pattern (hermetic): main forces a compaction each turn via the
companion tag; summary returns a canned LLM-2 JSON payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock import Sherlock
from sherlock.memory.entry import LTM_CONVERSATION_ID, MemoryState

# A grounded identity fact — the quote appears verbatim in the user's turn.
_USER_MSG = "Hi, my name is Kim and I am allergic to peanuts."
_IDENTITY_FACT = {
    "content": "User's name is Kim",
    "type": "fact",
    "source": "user",
    "confidence": 1.0,
    "quote": "my name is kim",
    "pin_recommended": True,
    "let_fade": False,
    "long_term": True,
    "category": "identity_health",
}


def _main_chat(_messages):
    # Force LLM-2 compaction every turn so promotion runs deterministically.
    return "Noted.\n<<sherlock-companions: compact>>"


def _make_summary_chat(payload: str, captured: list | None = None):
    def summary_chat(messages):
        if captured is not None:
            for m in messages:
                role = m["role"] if isinstance(m, dict) else m.role
                content = m["content"] if isinstance(m, dict) else m.content
                if role == "system":
                    captured.append(content)
                    break
        return payload

    return summary_chat


def _llm2_payload(facts, **extra) -> str:
    body = {
        "summary": "",
        "facts": facts,
        "topic_label": "t",
        "topic_changed_from_previous": False,
        "retrieval_keywords": [],
    }
    body.update(extra)
    return json.dumps(body)


def _agent(tmp_path, *, long_term, summary_chat, main_chat=_main_chat):
    return Sherlock.with_callable(
        main_chat=main_chat,
        summary_chat=summary_chat,
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
        background=False,  # inline companions → deterministic, inspectable
        companions_mode="turbo",  # compaction fires every turn
        long_term=long_term,
    )


# ---------------- kill switch ----------------


def test_kill_switch_no_suffix_no_sentinel_rows(tmp_path):
    captured: list[str] = []
    agent = _agent(
        tmp_path,
        long_term=None,  # feature OFF (default)
        summary_chat=_make_summary_chat(_llm2_payload([dict(_IDENTITY_FACT)]), captured),
    )
    agent.chat(_USER_MSG)
    # The LLM-2 prompt never gained the long-term instruction block.
    assert captured, "summary_chat never ran"
    assert all("LONG-TERM MEMORY" not in p for p in captured)
    # No sentinel rows ever created.
    assert agent.long_term_memory() == []
    assert agent.memory.list(conversation_id=LTM_CONVERSATION_ID) == []


# ---------------- promotion + restart durability ----------------


def test_promotion_survives_restart(tmp_path):
    captured: list[str] = []
    agent = _agent(
        tmp_path,
        long_term=True,
        summary_chat=_make_summary_chat(_llm2_payload([dict(_IDENTITY_FACT)]), captured),
    )
    agent.chat(_USER_MSG)
    conv_id = agent.conversation_id  # created lazily on the first chat()

    # Prompt gained the suffix; a pinned sentinel row exists with provenance.
    assert any("LONG-TERM MEMORY" in p for p in captured)
    ltm = agent.long_term_memory()
    assert len(ltm) == 1
    entry = ltm[0]
    assert entry["content"] == "User's name is Kim"
    assert entry["category"] == "identity_health"
    assert entry["origin_conversation_id"] == conv_id
    row = agent.memory.list(conversation_id=LTM_CONVERSATION_ID)[0]
    assert row.pinned is True
    assert json.loads(row.evidence)[0]["quote"] == "my name is kim"

    # Simulate a process restart: a brand-new Sherlock on the SAME storage_dir.
    agent2 = _agent(
        tmp_path,
        long_term=True,
        summary_chat=_make_summary_chat(_llm2_payload([])),
    )
    ltm2 = agent2.long_term_memory()
    assert len(ltm2) == 1
    assert ltm2[0]["content"] == "User's name is Kim"
    assert ltm2[0]["origin_conversation_id"] == conv_id


# ---------------- decay isolation across real turns ----------------


def test_sentinel_survives_many_turns_of_decay(tmp_path):
    # Turn 1 promotes; later turns emit nothing durable but keep decay running.
    payloads = iter([_llm2_payload([dict(_IDENTITY_FACT)])] + [_llm2_payload([]) for _ in range(6)])

    def summary_chat(messages):
        try:
            return next(payloads)
        except StopIteration:
            return _llm2_payload([])

    agent = _agent(tmp_path, long_term=True, summary_chat=summary_chat)
    for i in range(7):
        agent.chat(f"turn {i}: {_USER_MSG if i == 0 else 'unrelated chatter'}")

    rows = agent.memory.list(conversation_id=LTM_CONVERSATION_ID)
    assert len(rows) == 1
    assert rows[0].state == MemoryState.FRESH  # never decayed
    assert rows[0].pinned is True

    # Direct proof of isolation: decay the ACTIVE conversation at a huge turn
    # index — the sentinel scope is structurally out of reach.
    agent._decay.step(
        conversation_id=agent.conversation_id,
        current_turn_index=100_000,
        active_topics=["unrelated"],
    )
    assert agent.memory.list(conversation_id=LTM_CONVERSATION_ID)[0].state == MemoryState.FRESH


# ---------------- wipe ----------------


def test_wipe_long_term_clears_sentinel(tmp_path):
    agent = _agent(
        tmp_path,
        long_term=True,
        summary_chat=_make_summary_chat(_llm2_payload([dict(_IDENTITY_FACT)])),
    )
    agent.chat(_USER_MSG)
    assert len(agent.long_term_memory()) == 1
    # v1.12 Stage A4: wipe_long_term returns {"removed", "backup_path"} and (with
    # auto_export_on_wipe defaulting True) writes a Markdown backup first.
    result = agent.wipe_long_term()
    assert result["removed"] == 1
    assert result["backup_path"] and Path(result["backup_path"]).exists()
    assert agent.long_term_memory() == []


# ---------------- events: memory.promoted (sync) ----------------


def test_memory_promoted_event_sync(tmp_path):
    events: list[dict] = []
    agent = _agent(
        tmp_path,
        long_term=True,
        summary_chat=_make_summary_chat(_llm2_payload([dict(_IDENTITY_FACT)])),
    )
    agent.set_event_sink(events.append)
    agent.chat(_USER_MSG)

    promoted = [e for e in events if e["type"] == "memory.promoted"]
    assert len(promoted) == 1
    data = promoted[0]["data"]
    assert data["count"] == 1
    assert data["items"][0]["category"] == "identity_health"
    assert "Kim" in data["items"][0]["content"]


def test_no_memory_promoted_event_when_disabled(tmp_path):
    events: list[dict] = []
    agent = _agent(
        tmp_path,
        long_term=None,
        summary_chat=_make_summary_chat(_llm2_payload([dict(_IDENTITY_FACT)])),
    )
    agent.set_event_sink(events.append)
    agent.chat(_USER_MSG)
    assert [e for e in events if e["type"] == "memory.promoted"] == []


# ---------------- events: achat parity (async) ----------------


@pytest.mark.asyncio
async def test_memory_promoted_event_achat_parity(tmp_path):
    events: list[dict] = []

    async def main_chat(_messages):
        return "Noted.\n<<sherlock-companions: compact>>"

    agent = _agent(
        tmp_path,
        long_term=True,
        summary_chat=_make_summary_chat(_llm2_payload([dict(_IDENTITY_FACT)])),
        main_chat=main_chat,
    )
    agent.set_event_sink(events.append)
    await agent.achat(_USER_MSG)

    promoted = [e for e in events if e["type"] == "memory.promoted"]
    assert len(promoted) == 1
    assert promoted[0]["data"]["count"] == 1
    # The promotion actually landed via the async path.
    assert len(agent.long_term_memory()) == 1
