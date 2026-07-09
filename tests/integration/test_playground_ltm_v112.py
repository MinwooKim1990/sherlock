"""v1.12 Stage A5 — playground LONG-TERM MEMORY integration.

The A1–A4 library gave the agent cross-conversation long-term memory (a reserved
sentinel scope, promotion gate, natural-language management, export/import). A5
surfaces it in the playground: a stable per-profile storage dir (so memory
survives a restart), an eviction guard that never deletes a profile, and a small
set of live endpoints + a UI tab over the library API.

This pins the server-side half:
  * build_agent — profile dir under ~/.sherlock_playground when on, tempdir when
    off, profile-name sanitization (path-traversal → "default").
  * the eviction rmtree guard — a profile dir survives, a tempdir is reclaimed.
  * every endpoint — long_term/incognito live toggles, snapshot, one-row delete
    (rejecting a session-scoped id), wipe, export (each format), import round-trip,
    and the unknown-session / invalid-input error paths.
  * the memory.promoted event shape the Flow-tab SUMMARY renderer reads.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock.memory.entry import (  # noqa: E402
    LTM_CONVERSATION_ID,
    MemorySource,
    MemoryType,
)

# settings that keep the real agent hermetic (no embedder download, no network)
_HERMETIC = {"embedding": "fake", "search_engine": "off", "background": False}


# ============================================================ build_agent
def _build(monkeypatch, tmp_home, settings):
    """Call the REAL build_agent with fake role callables + HOME redirected."""
    import playground.providers as providers
    import playground.session as session_mod

    monkeypatch.setattr(providers, "make_role_callable", lambda role, sess, emit: (lambda m: "ok"))
    monkeypatch.setenv("HOME", str(tmp_home))
    sess = session_mod.Session(sid="s", models={}, loop=None, queue=None)
    agent = session_mod.build_agent(sess, "sys", {**_HERMETIC, **settings})
    return agent, sess


def test_build_agent_long_term_uses_profile_dir(monkeypatch, tmp_path):
    agent, sess = _build(monkeypatch, tmp_path, {"long_term": True, "ltm_profile": "work"})
    expected = str(tmp_path / ".sherlock_playground" / "work")
    assert sess.storage_dir == expected
    assert os.path.isdir(sess.storage_dir)
    assert agent.config.memory.long_term.enabled is True
    assert agent.config.memory.long_term.incognito is False


def test_build_agent_incognito_carried_through(monkeypatch, tmp_path):
    agent, _ = _build(
        monkeypatch, tmp_path, {"long_term": True, "ltm_profile": "p", "ltm_incognito": True}
    )
    assert agent.config.memory.long_term.enabled is True
    assert agent.config.memory.long_term.incognito is True


@pytest.mark.parametrize("evil", ["../x", "a/b", "", "  ", "UPPER", "x" * 40, "he;rm -rf"])
def test_build_agent_sanitizes_profile_to_default(monkeypatch, tmp_path, evil):
    _, sess = _build(monkeypatch, tmp_path, {"long_term": True, "ltm_profile": evil})
    # anything not matching [a-z0-9_-]{1,32} collapses to the single "default" dir
    assert sess.storage_dir == str(tmp_path / ".sherlock_playground" / "default")
    # and it stays strictly inside the playground root (no traversal escaped)
    root = str(tmp_path / ".sherlock_playground")
    assert os.path.abspath(sess.storage_dir).startswith(root + os.sep)


def test_build_agent_off_uses_tempdir(monkeypatch, tmp_path):
    agent, sess = _build(monkeypatch, tmp_path, {})  # long_term absent → off
    assert "sherlock_pg_" in sess.storage_dir
    assert ".sherlock_playground" not in sess.storage_dir
    assert agent.config.memory.long_term.enabled is False


# ============================================================ eviction guard
def test_eviction_guard_spares_profile_reclaims_tempdir(monkeypatch, tmp_path):
    import playground.server as server

    profile_dir = tmp_path / ".sherlock_playground" / "keepme"
    profile_dir.mkdir(parents=True)
    (profile_dir / "sherlock.db").write_text("durable")
    temp_dir = tmp_path / "sherlock_pg_throwaway"
    temp_dir.mkdir()
    (temp_dir / "sherlock.db").write_text("throwaway")

    monkeypatch.setenv("HOME", str(tmp_path))
    # the profile dir must be refused; the tempdir-prefixed path is reclaimable
    assert server._safe_to_rmtree(str(profile_dir)) is False
    assert server._safe_to_rmtree(str(temp_dir)) is True
    assert server._safe_to_rmtree("") is False


# ============================================================ endpoints
def _client(monkeypatch, tmp_path):
    """A TestClient whose sessions build a REAL long-term-enabled agent on a
    tmp HOME (so /api/session lands under ~/.sherlock_playground)."""
    import playground.providers as providers
    import playground.server as server

    monkeypatch.setattr(providers, "make_role_callable", lambda role, sess, emit: (lambda m: "ok"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return TestClient(server.app), server


def _start(client, **settings):
    body = {
        "api_key": "x",
        "models": {"main": "m"},
        "system_prompt": "p.",
        "settings": {**_HERMETIC, "long_term": True, "ltm_profile": "t", **settings},
    }
    return client.post("/api/session", json=body).json()["session_id"]


def _seed(agent, content, category, confidence=0.9):
    row = agent.memory.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=confidence,
        pinned=True,
        tags=f"ltm,{category}",
    )
    return row.id


def test_long_term_and_incognito_toggles_flip_live(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    assert sess.agent.config.memory.long_term.enabled is True  # started on

    r = client.post("/api/long_term", json={"session_id": sid, "on": False})
    assert r.json() == {"ok": True, "on": False}
    assert sess.agent.config.memory.long_term.enabled is False
    assert sess.settings["long_term"] is False

    r = client.post("/api/long_term", json={"session_id": sid, "on": True})
    assert r.json() == {"ok": True, "on": True}
    assert sess.agent.config.memory.long_term.enabled is True

    r = client.post("/api/incognito", json={"session_id": sid, "on": True})
    assert r.json() == {"ok": True, "on": True}
    assert sess.agent.config.memory.long_term.incognito is True
    assert sess.settings["ltm_incognito"] is True


def test_snapshot_returns_promoted_rows(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    _seed(sess.agent, "User's name is Minwoo", "identity_health")
    _seed(sess.agent, "Prefers concise replies", "stable_preference")

    j = client.get("/api/memory/long_term", params={"session_id": sid}).json()
    rows = j["rows"]
    assert len(rows) == 2
    contents = {r["content"] for r in rows}
    assert "User's name is Minwoo" in contents
    # every row carries the exact shape the UI table + snapshot render
    for r in rows:
        assert set(r) >= {"id", "category", "content", "confidence", "created_at", "origin"}
    cats = {r["category"] for r in rows}
    assert cats == {"identity_health", "stable_preference"}


def test_delete_removes_exactly_one_and_rejects_session_row(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    id1 = _seed(sess.agent, "fact one", "user_directive")
    id2 = _seed(sess.agent, "fact two", "user_directive")
    # a NON-sentinel (ordinary conversation) row must never be deletable here
    session_row = sess.agent.memory.add(
        conversation_id="some_conversation",
        content="a transient session fact",
        type=MemoryType.FACT,
        source=MemorySource.USER,
    )

    r = client.post("/api/memory/delete", json={"session_id": sid, "id": session_row.id})
    assert "error" in r.json()
    assert sess.agent.memory.get(session_row.id) is not None  # still there

    r = client.post("/api/memory/delete", json={"session_id": sid, "id": "no-such-id"})
    assert "error" in r.json()

    r = client.post("/api/memory/delete", json={"session_id": sid, "id": id1})
    assert r.json().get("ok") is True
    assert sess.agent.memory.get(id1) is None  # removed
    assert sess.agent.memory.get(id2) is not None  # exactly one gone
    assert len(sess.agent.long_term_memory()) == 1


def test_wipe_returns_removed_count(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    _seed(sess.agent, "durable a", "identity_health")
    _seed(sess.agent, "durable b", "relationship")

    j = client.post("/api/memory/wipe", json={"session_id": sid}).json()
    assert j["removed"] == 2
    assert "backup_path" in j  # auto backup honoured (default on)
    assert sess.agent.long_term_memory() == []


def test_export_each_format_has_right_content_type(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    _seed(sess.agent, "exportable fact", "identity_health")

    for fmt, ct in (
        ("md", "text/markdown"),
        ("json", "application/json"),
        ("sql", "application/sql"),
    ):
        r = client.get("/api/memory/export", params={"session_id": sid, "fmt": fmt})
        assert r.status_code == 200
        assert ct in r.headers["content-type"]
        assert "attachment" in r.headers["content-disposition"]
        assert f".{fmt if fmt != 'md' else 'md'}" in r.headers["content-disposition"]
        assert r.text.strip()

    r = client.get("/api/memory/export", params={"session_id": sid, "fmt": "bogus"})
    assert "error" in r.json()


def test_import_round_trip(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    _seed(sess.agent, "round trip fact", "identity_health")

    exported = client.get("/api/memory/export", params={"session_id": sid, "fmt": "json"}).text
    client.post("/api/memory/wipe", json={"session_id": sid})
    assert sess.agent.long_term_memory() == []

    j = client.post(
        "/api/memory/import", json={"session_id": sid, "text": exported, "fmt": "json"}
    ).json()
    assert j.get("imported", 0) >= 1
    assert any(r["content"] == "round trip fact" for r in sess.agent.long_term_memory())


def test_import_requires_enabled(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    client.post("/api/long_term", json={"session_id": sid, "on": False})
    j = client.post(
        "/api/memory/import",
        json={"session_id": sid, "text": '{"format":"sherlock-ltm","version":1,"facts":[]}'},
    ).json()
    assert "error" in j  # writes are gated on enabled


def test_import_rejects_local_file_path(monkeypatch, tmp_path):
    """F1: /api/memory/import must NOT become a local file-read primitive — a
    short existing path is refused before it reaches import_memory (which would
    otherwise read the file and slurp its facts in)."""
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    # a real, existing file whose contents WOULD import if the path were read
    secret = tmp_path / "backup.json"
    secret.write_text(
        '{"format":"sherlock-ltm","version":1,"facts":['
        '{"content":"leaked from disk","category":"identity_health","confidence":1.0}]}'
    )
    before = len(sess.agent.long_term_memory())
    j = client.post("/api/memory/import", json={"session_id": sid, "text": str(secret)}).json()
    assert "error" in j
    assert "path import" in j["error"]
    # nothing was read off disk / imported
    assert len(sess.agent.long_term_memory()) == before


def test_import_rejects_oversized_body(monkeypatch, tmp_path):
    """F5: an over-5MB import body is rejected up front (single-process guard)."""
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    j = client.post("/api/memory/import", json={"session_id": sid, "text": "x" * 5_000_001}).json()
    assert "error" in j
    assert "too large" in j["error"]


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/long_term", {"session_id": "nope", "on": True}),
        ("post", "/api/incognito", {"session_id": "nope", "on": True}),
        ("post", "/api/memory/delete", {"session_id": "nope", "id": "x"}),
        ("post", "/api/memory/wipe", {"session_id": "nope"}),
        ("post", "/api/memory/import", {"session_id": "nope", "text": "x"}),
    ],
)
def test_unknown_session_errors(monkeypatch, tmp_path, method, path, payload):
    client, _ = _client(monkeypatch, tmp_path)
    r = getattr(client, method)(path, json=payload)
    assert "error" in r.json()


def test_snapshot_and_export_unknown_session_error(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    assert "error" in client.get("/api/memory/long_term", params={"session_id": "nope"}).json()
    assert "error" in client.get("/api/memory/export", params={"session_id": "nope"}).json()


# ============================================================ event shape
def test_memory_promoted_event_reaches_events_log(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    # drive the REAL promotion-emit path the summarizer uses
    sess.agent._emit_memory_promoted(
        {"long_term_promoted": [{"category": "identity_health", "content": "allergic to peanuts"}]}
    )
    promoted = [e for e in sess.events_log if e.get("type") == "memory.promoted"]
    assert promoted, "memory.promoted must land in events_log for the Flow tab"
    d = promoted[-1]["data"]
    # the exact shape SUMMARY["memory.promoted"] reads: d.count + d.items[].content/category
    assert d["count"] == 1
    assert d["items"][0]["content"].startswith("allergic to peanuts")
    assert d["items"][0]["category"] == "identity_health"
    assert promoted[-1]["actor"] == "llm2"
