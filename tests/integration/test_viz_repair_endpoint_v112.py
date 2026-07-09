"""v1.12 Stage B3 — playground /api/viz/repair (LLM-4 runtime repair) endpoint.

The browser sandbox (B4) renders each viz artifact; on a REAL runtime failure it
posts the current HTML + the exact error here. The server runs ONE LLM-4 repair
with that error, re-lints, and returns the fixed HTML marked runtime-validated.
Rounds are bounded PER viz_id ACROSS calls. Also covers the unknown-session /
unknown-viz_id rejects, a still-broken repair, exhaustion, and the artifact GET.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock import Sherlock  # noqa: E402

# CSP + ready signal + data (12, 19). A full valid artifact.
VALID = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>Q1 12</span><span>Q2 19</span></div>\n"
    "<script>parent.postMessage('viz-ready', '*');</script></body></html>"
)

# The HTML the browser holds (has the data, so a repair's numbers trace) but is
# missing the ready signal — a plausible "rendered but threw at runtime" artifact.
BROKEN_INPUT = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>Q1 12</span><span>Q2 19</span></div></body></html>"
)

# A repair output that STILL fails the static lint (no ready signal).
STILL_BROKEN = BROKEN_INPUT


class _ScriptViz:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, messages):
        prompt = "\n".join((m["content"] if isinstance(m, dict) else m.content) for m in messages)
        with self._lock:
            self.prompts.append(prompt)
            return self._responses.pop(0) if self._responses else "NO MORE"


def _make_fake_build(viz_responses, viz_cfg):
    def fake_build_agent(session, system_prompt, settings):
        session.settings = settings or {}
        session.system_prompt = system_prompt or "…"
        viz = _ScriptViz(*viz_responses)
        agent = Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            inference_chat=lambda m: "{}",
            viz_chat=viz,
            system_prompt=system_prompt or "…",
            storage_dir=session.storage_dir or None,
            embedding="fake",
            background=False,
            main_search_engine="disabled",
            inference_search_engine="disabled",
            visualization=viz_cfg,
        )
        agent.set_event_sink(session.emit)
        session.agent = agent
        return agent

    return fake_build_agent


def _client(monkeypatch, *, viz_responses, viz_cfg):
    import playground.server as server

    monkeypatch.setattr(server, "build_agent", _make_fake_build(viz_responses, viz_cfg))
    return TestClient(server.app), server


def _start(client):
    return client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "p.", "settings": {}},
    ).json()["session_id"]


def _events_of(sess, type_):
    return [e for e in sess.events_log if e["type"] == type_]


# ------------------------------------------------------------ happy path


def test_repair_fixes_and_marks_runtime(monkeypatch):
    client, server = _client(
        monkeypatch,
        viz_responses=[VALID],
        viz_cfg={"enabled": True, "max_repair_rounds": 2},
    )
    sid = _start(client)
    sess = server.SESSIONS[sid]
    sess.viz_ids.add("t1-1")  # the browser only ever repairs a rendered id

    r = client.post(
        "/api/viz/repair",
        json={
            "session_id": sid,
            "viz_id": "t1-1",
            "html": BROKEN_INPUT,
            "error": "Uncaught TypeError: x is not a function",
        },
    ).json()

    assert r["ok"] is True
    assert r["validated"] == "runtime"
    assert "viz-ready" in r["html"]
    assert 'name="sherlock-viz-validated"' in r["html"]
    # flow-log events fired through the session sink
    assert len(_events_of(sess, "viz.repairing")) == 1
    rendered = _events_of(sess, "viz.rendered")
    assert len(rendered) == 1
    assert rendered[0]["data"]["validated"] == "runtime"
    assert rendered[0]["data"]["viz_id"] == "t1-1"


# ------------------------------------------------------------ round bounding


def test_repair_rounds_bounded_per_viz_id(monkeypatch):
    # cap=1: the FIRST call runs a repair, the SECOND is refused as exhausted.
    client, server = _client(
        monkeypatch,
        viz_responses=[VALID, VALID],
        viz_cfg={"enabled": True, "max_repair_rounds": 1},
    )
    sid = _start(client)
    sess = server.SESSIONS[sid]
    sess.viz_ids.add("t1-1")

    body = {"session_id": sid, "viz_id": "t1-1", "html": BROKEN_INPUT, "error": "boom"}
    r1 = client.post("/api/viz/repair", json=body).json()
    assert r1["ok"] is True

    r2 = client.post("/api/viz/repair", json=body).json()
    assert r2["ok"] is False
    assert r2["exhausted"] is True
    # only ONE actual repair ran despite two calls
    assert sess.viz_repair_rounds["t1-1"] == 1


# ------------------------------------------------------------ still-broken output


def test_repair_output_still_failing_lint(monkeypatch):
    client, server = _client(
        monkeypatch,
        viz_responses=[STILL_BROKEN],
        viz_cfg={"enabled": True, "max_repair_rounds": 2},
    )
    sid = _start(client)
    sess = server.SESSIONS[sid]
    sess.viz_ids.add("t1-1")

    r = client.post(
        "/api/viz/repair",
        json={"session_id": sid, "viz_id": "t1-1", "html": BROKEN_INPUT, "error": "boom"},
    ).json()

    assert r["ok"] is False
    assert "ready signal" in r["error"]
    assert r["exhausted"] is False  # one round left (cap 2)
    # no successful render emitted
    assert _events_of(sess, "viz.rendered") == []


# ------------------------------------------------------------ rejects


def test_repair_unknown_session(monkeypatch):
    client, _ = _client(monkeypatch, viz_responses=[VALID], viz_cfg={"enabled": True})
    r = client.post(
        "/api/viz/repair",
        json={"session_id": "nope", "viz_id": "t1-1", "html": VALID, "error": "e"},
    ).json()
    assert r["ok"] is False
    assert "no such session" in r["error"]


def test_repair_unknown_viz_id(monkeypatch):
    client, server = _client(monkeypatch, viz_responses=[VALID], viz_cfg={"enabled": True})
    sid = _start(client)
    r = client.post(
        "/api/viz/repair",
        json={"session_id": sid, "viz_id": "ghost-9", "html": VALID, "error": "e"},
    ).json()
    assert r["ok"] is False
    assert "unknown viz_id" in r["error"]


# ------------------------------------------------------------ kill switch


def test_repair_works_even_when_visualization_disabled(monkeypatch):
    """The endpoint is explicit — it runs regardless of the DR/chat kill switch
    (visualization disabled), using the default repair-round budget."""
    client, server = _client(
        monkeypatch,
        viz_responses=[VALID],
        viz_cfg=None,  # visualization DISABLED
    )
    sid = _start(client)
    sess = server.SESSIONS[sid]
    assert sess.agent.config.visualization.enabled is False
    sess.viz_ids.add("t1-1")

    r = client.post(
        "/api/viz/repair",
        json={"session_id": sid, "viz_id": "t1-1", "html": BROKEN_INPUT, "error": "boom"},
    ).json()
    assert r["ok"] is True
    assert r["validated"] == "runtime"


# ------------------------------------------------------------ artifact GET


def test_get_artifact_after_successful_repair(monkeypatch):
    client, server = _client(
        monkeypatch,
        viz_responses=[VALID],
        viz_cfg={"enabled": True, "max_repair_rounds": 2, "save_artifacts": True},
    )
    sid = _start(client)
    sess = server.SESSIONS[sid]
    sess.viz_ids.add("t1-1")

    client.post(
        "/api/viz/repair",
        json={"session_id": sid, "viz_id": "t1-1", "html": BROKEN_INPUT, "error": "boom"},
    )
    # the runtime-validated artifact was persisted → served as text/html
    g = client.get("/api/viz/t1-1", params={"session_id": sid})
    assert g.status_code == 200
    assert "text/html" in g.headers["content-type"]
    assert "viz-ready" in g.text

    # a missing id → structured error, not a served file
    g2 = client.get("/api/viz/does-not-exist", params={"session_id": sid})
    assert g2.json().get("error")


# ------------------------------------------------------------ F4(a) emit-register


def test_repair_accepts_emit_time_registered_id(monkeypatch):
    """v1.12 F4(a): registration happens at EMIT time. Drive an agent viz.pending
    through the real event sink (Session.emit) — NOT a manual sess.viz_ids.add —
    and /api/viz/repair accepts that id. Proves the auto-register path the chat/DR
    flows actually rely on (they never touch sess.viz_ids directly)."""
    client, server = _client(
        monkeypatch,
        viz_responses=[VALID],
        viz_cfg={"enabled": True, "max_repair_rounds": 2},
    )
    sid = _start(client)
    sess = server.SESSIONS[sid]

    # the agent surfaces a pending viz through its sink (== session.emit), exactly
    # like _apply_deep_research_viz / the chat marker strip do.
    sess.agent._emit("viz.pending", "llm4", {"viz_id": "dr9-1", "anchor": "⟦viz:dr9-1⟧"})
    assert "dr9-1" in sess.viz_ids  # emit-time registration, no manual add

    r = client.post(
        "/api/viz/repair",
        json={
            "session_id": sid,
            "viz_id": "dr9-1",
            "html": BROKEN_INPUT,
            "error": "Uncaught TypeError: x is not a function",
        },
    ).json()
    assert r["ok"] is True
    assert r["validated"] == "runtime"


# ------------------------------------------------------------ F4(b) path traversal


def test_get_artifact_rejects_path_traversal(monkeypatch):
    """v1.12 F4(b): a GET whose id encodes a path escape must never serve a file
    outside the session's viz dir — it 404s / returns a structured error and the
    outside file's bytes never appear in the response."""
    from pathlib import Path as _Path

    client, server = _client(monkeypatch, viz_responses=[VALID], viz_cfg={"enabled": True})
    sid = _start(client)
    sess = server.SESSIONS[sid]

    # plant a secret one level ABOVE the viz dir (viz dir is <storage>/viz), the
    # target a `../secret` escape would reach if sanitization/confinement failed.
    storage_root = _Path(sess.agent.config.storage.sqlite_path).resolve().parent
    (storage_root / "secret.html").write_text("TOP-SECRET-DO-NOT-SERVE", encoding="utf-8")

    for bad in ("..%2Fsecret", "../x"):
        g = client.get(f"/api/viz/{bad}", params={"session_id": sid})
        assert g.status_code == 404 or g.json().get("error")
        assert "TOP-SECRET-DO-NOT-SERVE" not in g.text
