"""v0.7 Phase 4 — playground wiring for deep_research.

Uses a real FastAPI TestClient but swaps `build_agent` for a scripted agent
(BYO-LLM callables + a fake counting engine) so no Gemini key / network is
needed. Verifies: a proposal surfaces `deep_research.approval_needed` and does
NOT run; `/api/deep_research/approve` runs the loop (round events + docs);
`/api/deep_research/skip` cancels.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sherlock import Sherlock  # noqa: E402
from sherlock.memory.entry import MemoryType  # noqa: E402
from sherlock.tools.web_search import SearchEngine  # noqa: E402


class CountingEngine(SearchEngine):
    def __init__(self):
        self.calls = []

    def search(self, query, *, max_results=5):
        self.calls.append((query, max_results))
        return [
            {"title": f"{query} #{i}", "url": f"https://ex.com/{i}", "content": f"c {i}"}
            for i in range(max_results)
        ]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": f"page {url}"}


def _main(messages):
    last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    c = last.get("content", "")
    if "Answer these meta-questions" in c:
        # sufficient after 2 rounds
        Mn["n"] += 1
        return json.dumps(
            {
                "answers": "facts https://ex.com/1",
                "key_finding": "k",
                "summary": "s",
                "sufficient": Mn["n"] >= 2,
                "next_queries": [] if Mn["n"] >= 2 else ["more"],
            }
        )
    if "RESEARCH DOCUMENTS:" in c:  # synthesis prompt (v0.7.1 embeds user text)
        return "FINAL: cited synthesis https://ex.com/1"
    if "RESEARCHME" in c:
        return 'I can dig deeper.\n<<sherlock-tool: deep_research "the topic">>'
    return "plain."


Mn = {"n": 0}


def _make_fake_build(captured):
    def fake_build_agent(session, system_prompt, settings):
        Mn["n"] = 0
        agent = Sherlock.with_callable(
            main_chat=_main,
            inference_chat=lambda m: json.dumps(["q1?", "q2?", "q3?"]),
            system_prompt=system_prompt or "…",
            storage_dir=session.storage_dir or None,
            embedding="fake",
            background=False,
            main_search_engine=CountingEngine(),
            inference_search_engine="disabled",
        )

        def sink(ev):
            captured.append(ev)
            session.emit(ev)

        agent.set_event_sink(sink)
        session.agent = agent
        return agent

    return fake_build_agent


def _client(monkeypatch, captured):
    import playground.server as server

    # storage_dir is set inside the real build_agent; our fake reads
    # session.storage_dir which the server leaves "" → with_callable makes a
    # temp dir when None.
    monkeypatch.setattr(server, "build_agent", _make_fake_build(captured))
    return TestClient(server.app), server


def test_playground_proposal_then_approve(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)

    r = client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "…", "settings": {}},
    )
    sid = r.json()["session_id"]

    # Propose: LLM-1 emits the deep_research tag → approval asked, NOTHING runs.
    r = client.post("/api/chat", json={"session_id": sid, "message": "please RESEARCHME"})
    reply = r.json()["reply"]
    assert "run it" in reply.lower() or "yes" in reply.lower()
    sess = server.SESSIONS[sid]
    assert sess.agent.pending_deep_research is not None
    assert sess.agent._main_search_engine.calls == [], "research ran before approval"
    assert any(e["type"] == "deep_research.approval_needed" for e in captured)

    # Approve via the UI endpoint → loop runs in the background; wait for it.
    r = client.post("/api/deep_research/approve", json={"session_id": sid})
    assert "ack" in r.json()
    sess.agent.wait_for_background(timeout=15)

    rounds = [e for e in captured if e["type"] == "deep_research.round"]
    assert len(rounds) == 2, f"expected 2 rounds, got {len(rounds)}"
    assert any(e["type"] == "deep_research.done" for e in captured)
    assert any(e["type"] == "deep_research.documents" for e in captured)
    docs = [
        m
        for m in sess.agent.memory.list(conversation_id=sess.agent.conversation_id)
        if m.type == MemoryType.DEEP_RESEARCH
    ]
    assert docs, "no DEEP_RESEARCH documents persisted"


def test_playground_skip_cancels(monkeypatch):
    captured: list[dict] = []
    client, server = _client(monkeypatch, captured)
    r = client.post(
        "/api/session",
        json={"api_key": "x", "models": {"main": "m"}, "system_prompt": "…", "settings": {}},
    )
    sid = r.json()["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "please RESEARCHME"})
    sess = server.SESSIONS[sid]
    assert sess.agent.pending_deep_research is not None

    r = client.post("/api/deep_research/skip", json={"session_id": sid})
    assert r.json()["ok"] is True
    assert sess.agent.pending_deep_research is None
    assert sess.agent._main_search_engine.calls == [], "research ran despite skip"
