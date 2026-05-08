"""SQLite baseline tests."""
from __future__ import annotations

from pathlib import Path

from sherlock.storage import Storage


def test_create_conversation_and_add_messages(tmp_path: Path) -> None:
    s = Storage(tmp_path / "test.db")
    conv = s.create_conversation(project="proj")
    assert conv.id
    assert conv.project == "proj"

    s.add_message(conv.id, role="system", content="sys-prompt")
    s.add_message(conv.id, role="user", content="hi")
    s.add_message(
        conv.id,
        role="assistant",
        content="hello back",
        model="fake/echo",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0,
    )
    msgs = s.list_messages(conv.id)
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2].model == "fake/echo"
    assert msgs[2].prompt_tokens == 10


def test_storage_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    s1 = Storage(db)
    conv = s1.create_conversation(project="p")
    s1.add_message(conv.id, role="user", content="kept")

    s2 = Storage(db)
    msgs = s2.list_messages(conv.id)
    assert len(msgs) == 1
    assert msgs[0].content == "kept"
