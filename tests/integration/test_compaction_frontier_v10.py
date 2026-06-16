"""v1.0 P7 — compaction frontier: summarized raw turns leave the K-turn tail
(infinite memory with a bounded per-turn prompt); nothing leaves SQLite."""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.budget import count_tokens


def _llm2(messages):
    last = messages[-1].get("content", "")
    if "TRANSCRIPT" in last or "not parseable" in last:
        return json.dumps(
            {
                "summary": "early turns: user introduced themselves and their cat",
                "facts": [],
                "topic_label": "intro",
            }
        )
    return "{}"


def _agent(tmp_path, **kw):
    return Sherlock.with_callable(
        main_chat=lambda m: "short reply.",
        summary_chat=_llm2,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
        **kw,
    )


def _run_turns(agent, n, compact_at=()):
    for i in range(1, n + 1):
        marker = f"TURNMARK{i:02d} " + "filler words here " * 10
        if i in compact_at:
            agent._provider._fn_marker = None  # no-op; compaction via tag below
        agent.chat(marker)
        if i in compact_at:
            # force a compaction at this turn boundary
            agent._summarizer.run(
                conversation_id=agent.conversation_id,
                recent_turns=agent._format_last_k_turns(agent.conversation_id, max(5, i)),
                turn_index=i,
            )
            agent._last_compact_turn = i


def test_frontier_evicts_summarized_turns_keeps_recent_four(tmp_path):
    agent = _agent(tmp_path)
    _run_turns(agent, 12, compact_at=(8,))
    tail, _ = agent._build_k_turn_tail(agent.conversation_id, 200_000)
    text = "\n".join(m.content for m in tail)
    # frontier=8, KEEP_RAW=4 → turns 9-12 raw always; ≤8 evicted
    for i in (9, 10, 11, 12):
        assert f"TURNMARK{i:02d}" in text, f"recent turn {i} must stay raw"
    for i in (1, 2, 3, 4, 5):
        assert f"TURNMARK{i:02d}" not in text, f"summarized turn {i} must be evicted"


def test_evicted_turns_remain_in_sqlite_and_timeline(tmp_path):
    agent = _agent(tmp_path)
    _run_turns(agent, 12, compact_at=(8,))
    all_msgs = agent.messages()
    assert any("TURNMARK01" in m.content for m in all_msgs), "raw transcript must be intact"
    from sherlock.tools.memory_tool import memory_timeline

    rows = memory_timeline(50, storage=agent._storage, conversation_id=agent.conversation_id)
    assert any("TURNMARK01" in str(r) for r in rows), "memory tool must still reach evicted turns"


def test_prompt_tokens_plateau_after_compaction(tmp_path):
    """The point of B4: per-turn prompt size stops growing linearly once
    summaries cover the old turns."""
    agent = _agent(tmp_path)
    _run_turns(agent, 6, compact_at=())
    state = agent.inspect_last_turn()
    before = sum(count_tokens(m.content) for m in state.messages_passed_to_llm1)
    # compact, then add the same number of turns again
    agent._summarizer.run(
        conversation_id=agent.conversation_id,
        recent_turns=agent._format_last_k_turns(agent.conversation_id, 6),
        turn_index=6,
    )
    _run_turns(agent, 1)  # one more turn after compaction
    for i in range(8, 13):
        agent.chat(f"TURNMARK{i:02d} " + "filler words here " * 10)
    state = agent.inspect_last_turn()
    after = sum(count_tokens(m.content) for m in state.messages_passed_to_llm1)
    # without the frontier `after` would exceed `before` by ~6 turns of filler;
    # with it the tail is bounded by KEEP_RAW + the summary line.
    assert after <= before + 260, f"prompt grew unbounded: {before} -> {after}"


def test_killswitch_disables_eviction(tmp_path):
    agent = _agent(tmp_path)
    agent.config.memory.compaction_frontier = False
    _run_turns(agent, 12, compact_at=(8,))
    tail, _ = agent._build_k_turn_tail(agent.conversation_id, 200_000)
    text = "\n".join(m.content for m in tail)
    assert "TURNMARK01" in text, "killswitch off must restore legacy behavior"


def test_message_turn_index_migration(tmp_path):
    from sqlalchemy import create_engine, text

    import sherlock.storage.db  # noqa: F401 — register models
    from sherlock.storage.db import run_migrations

    db = tmp_path / "old.sqlite"
    eng = create_engine(f"sqlite:///{db}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE message (id TEXT PRIMARY KEY, conversation_id TEXT, "
                "role TEXT, content TEXT)"
            )
        )
    added = run_migrations(eng)
    assert any("message.turn_index" in a for a in added), f"added={added}"
