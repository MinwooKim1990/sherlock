"""v1.12 Stage H1 — playground persistent history: /api/history list + open
(switch_session + viz-id registration) + new + title, and the deterministic
first-turn auto-title in /api/chat."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock import Sherlock  # noqa: E402

VALID = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>ok</span></div>\n"
    "<script>window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');"
    "parent.postMessage({sherlockViz:'ready'}, '*');</script></body></html>"
)


def _make_fake_build(tmp_path, reply="ok."):
    def fake_build_agent(session, system_prompt, settings):
        session.settings = settings or {}
        session.system_prompt = system_prompt or "…"
        session.storage_dir = str(tmp_path / "store")
        agent = Sherlock.with_callable(
            main_chat=lambda m: reply,
            summary_chat=lambda m: "{}",
            inference_chat=lambda m: "{}",
            viz_chat=lambda m: VALID,
            system_prompt=system_prompt or "…",
            storage_dir=session.storage_dir,
            embedding="fake",
            background=False,
            companions_mode="off",
            main_search_engine="disabled",
            inference_search_engine="disabled",
            visualization=True,
        )
        agent.set_event_sink(session.emit)
        session.agent = agent
        return agent

    return fake_build_agent


def _client(monkeypatch, tmp_path, reply="ok."):
    import playground.server as server

    monkeypatch.setattr(server, "build_agent", _make_fake_build(tmp_path, reply))
    return TestClient(server.app), server


def _start(client):
    r = client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "p.", "settings": {}},
    )
    return r.json()["session_id"]


def test_history_lists_conversations_with_auto_title(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)

    client.post("/api/chat", json={"session_id": sid, "message": "what is the tallest tower?"})
    r = client.get("/api/history", params={"session_id": sid}).json()
    convs = r["conversations"]
    assert len(convs) == 1
    assert convs[0]["title"] == "what is the tallest tower?"  # deterministic auto-title
    assert convs[0]["active"] is True
    assert convs[0]["messages"] >= 2  # user + assistant persisted


def test_history_new_open_roundtrip(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]

    client.post("/api/chat", json={"session_id": sid, "message": "first conversation"})
    first_cid = sess.agent.conversation_id

    r = client.post("/api/history/new", json={"session_id": sid}).json()
    assert r["ok"] is True
    assert sess.agent.conversation_id != first_cid
    client.post("/api/chat", json={"session_id": sid, "message": "second conversation"})

    convs = client.get("/api/history", params={"session_id": sid}).json()["conversations"]
    assert len(convs) == 2
    assert convs[0]["title"] == "second conversation"  # newest first

    # reopen the first — active flips, messages come back, chat continues there
    r = client.post(
        "/api/history/open", json={"session_id": sid, "conversation_id": first_cid}
    ).json()
    assert r["ok"] is True
    roles = [m["role"] for m in r["messages"]]
    assert roles.count("user") == 1
    assert "first conversation" in r["messages"][0]["content"]
    assert sess.agent.conversation_id == first_cid

    client.post("/api/chat", json={"session_id": sid, "message": "back again"})
    msgs = sess.agent._storage.list_messages(first_cid)
    assert any("back again" in m.content for m in msgs if m.role == "user")


def test_history_open_registers_viz_ids(monkeypatch, tmp_path):
    reply = "chart:\n<<sherlock-viz: bar chart | A 1, B 2>>\ndone"
    client, server = _client(monkeypatch, tmp_path, reply=reply)
    sid = _start(client)
    sess = server.SESSIONS[sid]

    client.post("/api/chat", json={"session_id": sid, "message": "viz please"})
    assert sess.agent.wait_for_viz(timeout=5) is True
    cid = sess.agent.conversation_id

    # a FRESH session on the same store must be able to rehydrate after open
    sid2 = _start(client)
    sess2 = server.SESSIONS[sid2]
    assert "t1-1" not in sess2.viz_ids
    r = client.post("/api/history/open", json={"session_id": sid2, "conversation_id": cid}).json()
    assert r["ok"] is True
    assert any("⟦viz:t1-1⟧" in m["content"] for m in r["messages"])
    assert "t1-1" in sess2.viz_ids  # registered → /api/viz/t1-1 can serve it

    art = client.get("/api/viz/t1-1", params={"session_id": sid2})
    assert art.status_code == 200
    assert "sherlock-viz-validated" in art.text


def test_history_title_rename(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    cid = sess.agent.conversation_id

    r = client.post(
        "/api/history/title",
        json={"session_id": sid, "conversation_id": cid, "title": "재무 분석 세션"},
    ).json()
    assert r["ok"] is True
    convs = client.get("/api/history", params={"session_id": sid}).json()["conversations"]
    assert convs[0]["title"] == "재무 분석 세션"

    r = client.post(
        "/api/history/title",
        json={"session_id": sid, "conversation_id": "nope", "title": "x"},
    ).json()
    assert r == {"error": "no such conversation"}


def test_history_open_unknown_conversation_errors(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    r = client.post(
        "/api/history/open", json={"session_id": sid, "conversation_id": "missing"}
    ).json()
    assert "error" in r


def test_history_unknown_session_errors(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    assert "error" in client.get("/api/history", params={"session_id": "zz"}).json()
    assert "error" in client.post("/api/history/new", json={"session_id": "zz"}).json()


def test_viz_artifact_serving_is_conversation_scoped(monkeypatch, tmp_path):
    # audit P1: two conversations each render a t1-1 — /api/viz must serve the
    # artifact of the conversation the caller opened, never a first-match rglob.
    import itertools

    counter = itertools.count()

    def viz_chat(messages):
        word = "alpha" if next(counter) == 0 else "beta"
        return VALID.replace("<span>ok</span>", f"<span>{word}</span>")

    import playground.server as server

    def fake_build(session, system_prompt, settings):
        session.settings = settings or {}
        session.storage_dir = str(tmp_path / "store")
        agent = Sherlock.with_callable(
            main_chat=lambda m: "c:\n<<sherlock-viz: a chart>>",
            summary_chat=lambda m: "{}",
            inference_chat=lambda m: "{}",
            viz_chat=viz_chat,
            system_prompt="p.",
            storage_dir=session.storage_dir,
            embedding="fake",
            background=False,
            companions_mode="off",
            main_search_engine="disabled",
            inference_search_engine="disabled",
            visualization=True,
        )
        agent.set_event_sink(session.emit)
        session.agent = agent
        return agent

    monkeypatch.setattr(server, "build_agent", fake_build)
    client = TestClient(server.app)
    sid = _start(client)
    sess = server.SESSIONS[sid]

    client.post("/api/chat", json={"session_id": sid, "message": "one"})
    assert sess.agent.wait_for_viz(timeout=5) is True
    conv_a = sess.agent.conversation_id
    client.post("/api/history/new", json={"session_id": sid})
    client.post("/api/chat", json={"session_id": sid, "message": "two"})
    assert sess.agent.wait_for_viz(timeout=5) is True
    conv_b = sess.agent.conversation_id
    assert conv_a != conv_b

    # open A → its t1-1 must be ALPHA; open B → BETA (regardless of walk order)
    client.post("/api/history/open", json={"session_id": sid, "conversation_id": conv_a})
    art = client.get("/api/viz/t1-1", params={"session_id": sid, "conv": conv_a})
    assert "alpha" in art.text and "beta" not in art.text
    client.post("/api/history/open", json={"session_id": sid, "conversation_id": conv_b})
    art = client.get("/api/viz/t1-1", params={"session_id": sid, "conv": conv_b})
    assert "beta" in art.text and "alpha" not in art.text
    # without conv=, the ACTIVE conversation (B) scopes the lookup
    art = client.get("/api/viz/t1-1", params={"session_id": sid})
    assert "beta" in art.text


def test_viz_error_contract_is_json_not_html(monkeypatch, tmp_path):
    # audit: the browser's rehydrate treats a non-text/html response as
    # degrade — pin that a registered id WITHOUT an artifact answers JSON.
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]
    sess.viz_ids.add("t9-9")  # registered but never rendered/persisted
    r = client.get("/api/viz/t9-9", params={"session_id": sid})
    assert r.status_code == 200
    assert "text/html" not in (r.headers.get("content-type") or "")
    assert r.json() == {"error": "no such artifact"}


def test_auto_title_uses_first_message_of_reopened_conversation(monkeypatch, tmp_path):
    # audit: an untitled (legacy) conversation reopened from history must be
    # titled with its FIRST user message, not whatever turn came latest.
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]

    client.post("/api/chat", json={"session_id": sid, "message": "the original first question"})
    cid = sess.agent.conversation_id
    sess.agent._storage.set_conversation_title(cid, "")  # simulate a legacy NULL title

    client.post("/api/history/open", json={"session_id": sid, "conversation_id": cid})
    client.post("/api/chat", json={"session_id": sid, "message": "a much later follow-up"})
    conv = sess.agent._storage.get_conversation(cid)
    assert conv.title == "the original first question"


def test_history_delete_conversation(monkeypatch, tmp_path):
    client, server = _client(monkeypatch, tmp_path)
    sid = _start(client)
    sess = server.SESSIONS[sid]

    client.post("/api/chat", json={"session_id": sid, "message": "first"})
    first_cid = sess.agent.conversation_id
    client.post("/api/history/new", json={"session_id": sid})
    client.post("/api/chat", json={"session_id": sid, "message": "second"})

    # delete the INACTIVE one
    r = client.post(
        "/api/history/delete", json={"session_id": sid, "conversation_id": first_cid}
    ).json()
    assert r["ok"] is True and r["switched"] is False and r["removed"] >= 2
    assert sess.agent._storage.get_conversation(first_cid) is None
    assert len(client.get("/api/history", params={"session_id": sid}).json()["conversations"]) == 1

    # delete the ACTIVE one → server switches to a fresh conversation first
    active = sess.agent.conversation_id
    r = client.post(
        "/api/history/delete", json={"session_id": sid, "conversation_id": active}
    ).json()
    assert r["ok"] is True and r["switched"] is True
    assert sess.agent.conversation_id != active

    r = client.post(
        "/api/history/delete", json={"session_id": sid, "conversation_id": "nope"}
    ).json()
    assert r == {"error": "no such conversation"}
