"""Session management API end-to-end (v0.4.0)."""

from __future__ import annotations

from sherlock import Sherlock


def test_list_sessions_starts_empty(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    sessions = agent.list_sessions()
    assert sessions == []


def test_chat_creates_session_appears_in_list(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("hello")
    sessions = agent.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s.id == agent.conversation_id
    assert s.turn_count >= 1


def test_new_session_switches_to_fresh(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("first session turn")
    first_id = agent.conversation_id

    new_id = agent.new_session()
    assert new_id != first_id
    assert agent.conversation_id == new_id
    # First session still listed:
    ids = {s.id for s in agent.list_sessions()}
    assert first_id in ids and new_id in ids


def test_switch_session_restores_turn_index(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("a")
    agent.chat("b")
    agent.chat("c")
    sid1 = agent.conversation_id
    turn_count = agent._turn_index

    agent.new_session()  # switch away
    assert agent._turn_index == 0

    agent.switch_session(sid1)
    assert agent.conversation_id == sid1
    assert agent._turn_index == turn_count


def test_delete_session_cascades(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("hello")
    sid = agent.conversation_id

    info = agent.delete_session(sid)
    assert info["session_id"] == sid
    assert info["messages_removed"] >= 1

    # Listing after deletion shouldn't include the session.
    ids = {s.id for s in agent.list_sessions()}
    assert sid not in ids

    # Active session was the deleted one — agent should have cleared state.
    assert agent._conversation is None
    assert agent._turn_index == 0


def test_delete_other_session_keeps_active(tmp_path):
    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    agent.chat("turn in session 1")
    sid1 = agent.conversation_id
    agent.new_session()
    agent.chat("turn in session 2")
    sid2 = agent.conversation_id

    agent.delete_session(sid1)
    # Active session is still sid2:
    assert agent.conversation_id == sid2
    # Continue using sid2 without error:
    agent.chat("still here")
    assert agent._turn_index == 2


def test_switch_unknown_session_raises(tmp_path):
    import pytest

    agent = Sherlock.with_callable(
        main_chat=lambda msgs: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
    )
    with pytest.raises(ValueError):
        agent.switch_session("not-a-real-id")
