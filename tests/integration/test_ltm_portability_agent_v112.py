"""v1.12 Stage A4 — long-term memory PORTABILITY (agent-level, end-to-end).

Drives ``Sherlock.export_memory`` / ``import_memory`` / ``wipe_long_term`` and
the chat-driven ``memory wipe-confirm`` backup hook. Hermetic: fake embeddings,
canned callables, ``background=False`` so state is settled when chat() returns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sherlock import Sherlock
from sherlock.memory.entry import LTM_CONVERSATION_ID, MemorySource, MemoryType

_TOKEN_RE = re.compile(r"CONFIRM TOKEN:\s*([0-9a-f]+)")


def _agent(tmp_path, main=None, *, long_term):
    return Sherlock.with_callable(
        main_chat=main or (lambda _m: "네."),
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        companions_mode="off",
        long_term=long_term,
    )


def _seed(agent, content, category="identity_health", conf=1.0, origin=None):
    return agent.memory.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=conf,
        pinned=True,
        tags=f"ltm,{category}",
        evidence=json.dumps([{"quote": content, "turn": 1}], ensure_ascii=False),
        origin_conversation_id=origin,
        dedup=False,
    )


def _events_of(events, typ):
    return [e for e in events if e["type"] == typ]


class ToolMain:
    """Stateful fake LLM-1: emits a queued tag on a real turn, records the
    tool-result block (and any confirm token) on finalisation."""

    def __init__(self):
        self.next_tag: str | None = None
        self.token: str | None = None

    def __call__(self, messages):
        last_user = ""
        for m in reversed(messages):
            role = m["role"] if isinstance(m, dict) else m.role
            if role == "user":
                last_user = m["content"] if isinstance(m, dict) else m.content
                break
        if "SHERLOCK TOOL RESULTS" in last_user:
            mm = _TOKEN_RE.search(last_user)
            if mm:
                self.token = mm.group(1)
            return "완료."
        tag = self.next_tag
        self.next_tag = None
        return tag or "네."


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_works_when_disabled(tmp_path):
    # Feature OFF: reading/exporting existing rows is harmless and still works.
    agent = _agent(tmp_path, long_term=False)
    _seed(agent, "User is allergic to peanuts")
    md = agent.export_memory("markdown")
    assert "User is allergic to peanuts" in md
    assert json.loads(agent.export_memory("json"))["facts"]
    assert "INSERT INTO memory_entry" in agent.export_memory("sql")


def test_export_writes_file_and_emits_event(tmp_path):
    events: list[dict] = []
    agent = _agent(tmp_path, long_term=True)
    agent.set_event_sink(events.append)
    _seed(agent, "Always answer in metric", category="user_directive")

    out = tmp_path / "export.json"
    text = agent.export_memory("json", path=str(out))
    assert out.exists() and out.read_text(encoding="utf-8") == text
    exported = _events_of(events, "memory.exported")
    assert exported and exported[-1]["data"] == {
        "format": "json",
        "count": 1,
        "path": str(out),
    }


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def test_import_requires_enabled(tmp_path):
    agent = _agent(tmp_path, long_term=False)  # disabled
    envelope = {
        "format": "sherlock-ltm",
        "version": 1,
        "facts": [{"content": "x", "category": "user_directive"}],
    }
    result = agent.import_memory(json.dumps(envelope))
    assert "error" in result and result["imported"] == 0
    assert agent.long_term_memory() == []  # nothing written


def test_import_round_trip_via_agent_and_event(tmp_path):
    events: list[dict] = []
    src = _agent(tmp_path / "a", long_term=True)
    _seed(src, "User is allergic to peanuts", category="identity_health")
    _seed(src, "Always answer in metric", category="user_directive", conf=0.9)
    exported = src.export_memory("json")

    dst = _agent(tmp_path / "b", long_term=True)
    dst.set_event_sink(events.append)
    result = dst.import_memory(exported)  # raw text, auto-detected as json
    assert result["imported"] == 2 and result["skipped"] == 0
    assert {r["content"] for r in dst.long_term_memory()} == {
        "User is allergic to peanuts",
        "Always answer in metric",
    }
    imp = _events_of(events, "memory.imported")
    assert imp and imp[-1]["data"] == {"imported": 2, "skipped": 0}


def test_import_from_file_path(tmp_path):
    src = _agent(tmp_path / "a", long_term=True)
    _seed(src, "저는 소바 알레르기가 있어요", category="identity_health")
    path = tmp_path / "mem.md"
    src.export_memory("markdown", path=str(path))

    dst = _agent(tmp_path / "b", long_term=True)
    result = dst.import_memory(str(path))  # a filesystem path, auto-detected md
    assert result["imported"] == 1
    assert {r["content"] for r in dst.long_term_memory()} == {"저는 소바 알레르기가 있어요"}


# ---------------------------------------------------------------------------
# wipe backup
# ---------------------------------------------------------------------------


def test_wipe_backup_writes_file_before_delete(tmp_path):
    events: list[dict] = []
    agent = _agent(tmp_path, long_term=True)
    agent.set_event_sink(events.append)
    _seed(agent, "fact one")
    _seed(agent, "fact two", category="user_directive")

    result = agent.wipe_long_term(backup=True)
    assert result["removed"] == 2
    backup = Path(result["backup_path"])
    assert backup.exists() and backup.name.startswith("ltm_backup_")
    # backup captured the facts BEFORE deletion
    body = backup.read_text(encoding="utf-8")
    assert "fact one" in body and "fact two" in body
    assert agent.long_term_memory() == []
    wiped = _events_of(events, "memory.wiped")
    assert wiped and wiped[-1]["data"]["backup_path"] == str(backup)


def test_wipe_backup_default_from_config(tmp_path):
    # auto_export_on_wipe defaults True (Stage A4 flip) → backup even with no arg.
    agent = _agent(tmp_path, long_term=True)
    _seed(agent, "remember me")
    result = agent.wipe_long_term()
    assert result["backup_path"] and Path(result["backup_path"]).exists()


def test_wipe_backup_opt_out(tmp_path):
    agent = _agent(tmp_path, long_term={"auto_export_on_wipe": False})
    _seed(agent, "no backup please")
    result = agent.wipe_long_term()  # None → resolves to config False
    assert result["removed"] == 1 and result["backup_path"] is None


def test_wipe_confirm_via_chat_backs_up(tmp_path):
    # auto_export_on_wipe True (default) → the chat-driven wipe-confirm writes a
    # Markdown backup to the storage dir first.
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)
    _seed(agent, "chat fact one")
    _seed(agent, "chat fact two", category="user_directive")

    main.next_tag = "<<sherlock-tool: memory wipe>>"
    agent.chat("장기기억 다 지워")
    assert main.token is not None

    main.next_tag = f"<<sherlock-tool: memory wipe-confirm {main.token}>>"
    agent.chat("응 전부 지워")

    assert agent.memory.list(conversation_id=LTM_CONVERSATION_ID) == []
    backups = list(tmp_path.glob("ltm_backup_*.md"))
    assert backups, "no backup file was written before the wipe"
    body = backups[0].read_text(encoding="utf-8")
    assert "chat fact one" in body and "chat fact two" in body
    wiped = _events_of(events, "memory.wiped")
    assert wiped and wiped[-1]["data"]["count"] == 2
    assert wiped[-1]["data"]["backup_path"]


# ---------------------------------------------------------------------------
# F2 (audit): backup fail-CLOSED — a failed backup write aborts the wipe
# ---------------------------------------------------------------------------


def test_wipe_backup_failure_aborts_agent_api(tmp_path):
    # F2 (audit): if the backup write fails, wipe_long_term must NOT proceed to
    # an unrecoverable delete — rows survive, an error is returned, no event.
    events: list[dict] = []
    agent = _agent(tmp_path, long_term=True)
    agent.set_event_sink(events.append)
    _seed(agent, "precious fact one")
    _seed(agent, "precious fact two", category="user_directive")

    # Point the backup dir at an existing FILE → mkdir raises → fail-closed.
    bad = tmp_path / "not_a_dir"
    bad.write_text("i am a file", encoding="utf-8")
    agent._ltm_storage_dir = lambda: bad  # type: ignore[assignment]

    result = agent.wipe_long_term(backup=True)
    assert "error" in result and result["removed"] == 0
    assert result["backup_path"] is None
    assert {r["content"] for r in agent.long_term_memory()} == {
        "precious fact one",
        "precious fact two",
    }
    assert _events_of(events, "memory.wiped") == []  # no wipe event on abort


def test_wipe_confirm_via_chat_backup_failure_aborts(tmp_path):
    # F2 (audit): a failed backup during the CHAT-driven wipe-confirm aborts the
    # wipe (fail-closed) — rows survive, no memory.wiped event. The confirm token
    # is consumed, so the user must re-preview (correct for a destructive op).
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)
    _seed(agent, "keep me one")
    _seed(agent, "keep me two", category="user_directive")

    bad = tmp_path / "blocker_file"
    bad.write_text("file where a dir should be", encoding="utf-8")
    agent._ltm_storage_dir = lambda: bad  # type: ignore[assignment]

    main.next_tag = "<<sherlock-tool: memory wipe>>"
    agent.chat("장기기억 지워")
    assert main.token is not None

    main.next_tag = f"<<sherlock-tool: memory wipe-confirm {main.token}>>"
    agent.chat("응 지워")

    survivors = {e.content for e in agent.memory.list(conversation_id=LTM_CONVERSATION_ID)}
    assert survivors == {"keep me one", "keep me two"}
    assert _events_of(events, "memory.wiped") == []


# ---------------------------------------------------------------------------
# F4 (audit): export path handling — parent mkdir
# ---------------------------------------------------------------------------


def test_export_to_nested_nonexistent_dir(tmp_path):
    # F4 (audit): export_memory creates missing parent dirs instead of crashing.
    agent = _agent(tmp_path, long_term=True)
    _seed(agent, "Always answer in metric", category="user_directive")
    out = tmp_path / "deep" / "nested" / "dir" / "export.json"
    text = agent.export_memory("json", path=str(out))
    assert out.exists() and out.read_text(encoding="utf-8") == text


# ---------------------------------------------------------------------------
# F5 (audit): no empty backup file on a double-wipe
# ---------------------------------------------------------------------------


def test_double_wipe_writes_no_second_backup(tmp_path):
    # F5 (audit): the first wipe backs up the live facts; a second wipe (now
    # empty) writes NO backup file and returns backup_path None.
    agent = _agent(tmp_path, long_term=True)
    _seed(agent, "one and only")

    first = agent.wipe_long_term(backup=True)
    assert first["backup_path"] and Path(first["backup_path"]).exists()

    second = agent.wipe_long_term(backup=True)
    assert second["removed"] == 0 and second["backup_path"] is None

    # exactly one backup file exists — the one from the first (non-empty) wipe.
    assert len(list(tmp_path.glob("ltm_backup_*.md"))) == 1
