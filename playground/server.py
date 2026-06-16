"""FastAPI backend for the Sherlock Live Inspector.

Endpoints:
  POST /api/models   {provider, api_key, base_url}   -> live model list for ONE provider
  POST /api/session  {providers, models, system_prompt, settings} -> {session_id}
  POST /api/chat     {session_id, message, mode}     -> {reply, latency_ms, baseline?}
                     mode: sherlock (default) | single (bare LLM only) | both (A/B)
  GET  /api/export   ?session_id=...                 -> the session as a markdown file
  WS   /ws/{sid}                                      -> per-session event stream
  GET  /                                              -> the single-page UI

Providers: gemini | openai | anthropic | local (any OpenAI-compatible server —
Ollama, LM Studio, vLLM...). API keys stay in the server-side Session and are
never echoed back to the browser.

Run:  python -m uvicorn playground.server:app --reload   (then open http://localhost:8000)
"""

from __future__ import annotations

import asyncio
import datetime
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from playground import providers as prov
from playground.session import Session, build_agent, carry_snapshot, memory_snapshot

app = FastAPI(title="Sherlock Live Inspector")
STATIC = Path(__file__).parent / "static"
SESSIONS: dict[str, Session] = {}
MAX_SESSIONS = 8  # evict the oldest session (and its tempdir) beyond this


class ModelsReq(BaseModel):
    provider: str = "gemini"
    api_key: str = ""
    base_url: str = ""  # local provider only


@app.post("/api/models")
async def api_models(req: ModelsReq):
    loop = asyncio.get_running_loop()
    try:
        models = await loop.run_in_executor(
            None, prov.list_models, req.provider, req.api_key, req.base_url
        )
        return {"models": models}
    except Exception as exc:
        return {"models": [], "error": f"{type(exc).__name__}: {exc}"}


class SessionReq(BaseModel):
    models: dict
    providers: dict = {}  # {provider: {api_key, base_url}} — kept server-side
    api_key: str = ""  # legacy single-Gemini-key shape (kept for tests/scripts)
    system_prompt: str = "You are a helpful assistant."
    settings: dict = {}


@app.post("/api/session")
async def api_session(req: SessionReq):
    sid = uuid.uuid4().hex[:12]
    loop = asyncio.get_running_loop()
    creds = dict(req.providers or {})
    if req.api_key and "gemini" not in creds:
        creds["gemini"] = {"api_key": req.api_key}
    sess = Session(
        sid=sid,
        models=req.models,
        providers=creds,
        loop=loop,
        queue=asyncio.Queue(),
    )
    try:
        await loop.run_in_executor(None, build_agent, sess, req.system_prompt, req.settings)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    SESSIONS[sid] = sess
    while len(SESSIONS) > MAX_SESSIONS:  # dicts keep insertion order → oldest first
        evicted = SESSIONS.pop(next(iter(SESSIONS)))
        if evicted.storage_dir:
            shutil.rmtree(evicted.storage_dir, ignore_errors=True)
    return {
        "session_id": sid,
        "embedding": sess.settings.get("embedding", "local"),
        "background": sess.settings.get("background", True),
    }


class ChatReq(BaseModel):
    session_id: str
    message: str
    mode: str = "sherlock"
    # fair baseline: the single LLM gets one naive search pass by default
    baseline_search: bool = True  # sherlock | single (bare LLM only) | both (A/B comparison)


def _emit_baseline(sess: Session, message: str, baseline: dict) -> None:
    """Record a bare-model A/B reply as a synthetic event — sess.emit puts it on
    the WS stream (the flow tab shows it) AND in events_log for /api/export."""
    sess.emit(
        {
            "type": "baseline.reply",
            "actor": "system",
            "turn": sess.turn,
            "data": {"user_text": message, **baseline},
        }
    )


@app.post("/api/chat")
async def api_chat(req: ChatReq):
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session (start a session first)"}
    sess.turn += 1
    loop = asyncio.get_running_loop()

    # "single": the bare-model baseline only — the Sherlock agent never runs.
    if req.mode == "single":
        baseline = await loop.run_in_executor(
            None, lambda: prov.baseline_chat(sess, req.message, use_search=req.baseline_search)
        )
        _emit_baseline(sess, req.message, baseline)
        return {"reply": None, "baseline": baseline}

    # Run the (sync) turn in a worker thread so the event loop keeps draining
    # the WS queue — events stream to the browser WHILE the turn runs. In
    # "both" mode the bare-model baseline runs CONCURRENTLY in a second worker
    # so the wall-clock comparison is fair (works even while deep research is
    # pending/running on the sherlock side — the baseline is independent).
    def _timed_chat() -> tuple:
        t0 = time.time()
        text = sess.agent.chat(req.message)
        return text, int((time.time() - t0) * 1000)

    baseline = None
    if req.mode == "both":
        (reply, sherlock_ms), baseline = await asyncio.gather(
            loop.run_in_executor(None, _timed_chat),
            loop.run_in_executor(
                None, lambda: prov.baseline_chat(sess, req.message, use_search=req.baseline_search)
            ),
        )
        _emit_baseline(sess, req.message, baseline)
    else:
        reply, sherlock_ms = await loop.run_in_executor(None, _timed_chat)
    # v1.3: record the sherlock-side wall-clock as a synthetic event (same
    # pattern as baseline.reply) so /api/export can render per-turn latency —
    # the core's turn.completed event can't be touched from here.
    sess.emit(
        {
            "type": "sherlock.latency",
            "actor": "system",
            "turn": sess.turn,
            "data": {"latency_ms": sherlock_ms},
        }
    )
    # Let the background companions finish, then push full snapshots. Skip the
    # drain while a deep-research run is active — it would block this endpoint
    # for the whole multi-minute run; the approve endpoint's _finish() task
    # already snapshots after research completes.
    if not sess.agent.is_deep_researching:
        await loop.run_in_executor(None, sess.agent.drain)
    sess.emit(
        {
            "type": "memory.snapshot",
            "actor": "memory",
            "turn": sess.turn,
            "data": {"rows": memory_snapshot(sess.agent)},
        }
    )
    sess.emit(
        {
            "type": "carry.snapshot",
            "actor": "carry",
            "turn": sess.turn,
            "data": carry_snapshot(sess.agent),
        }
    )
    sess.emit({"type": "turn.done", "actor": "system", "turn": sess.turn, "data": {}})
    out = {"reply": reply, "latency_ms": sherlock_ms}
    if baseline is not None:
        out["baseline"] = baseline
    return out


class DeepResearchReq(BaseModel):
    session_id: str


@app.post("/api/deep_research/approve")
async def api_dr_approve(req: DeepResearchReq):
    """v0.7: UI approval for a pending deep-research proposal. Kicks off the
    background loop (which streams `deep_research.round` events over WS), then
    pushes a memory snapshot once it finishes."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    loop = asyncio.get_running_loop()
    ack = await loop.run_in_executor(None, sess.agent.approve_deep_research)
    if ack is None:
        return {"error": "nothing pending to approve"}

    async def _finish():
        # Wait for the background research to complete, then snapshot memory
        # (the DEEP_RESEARCH docs land there) + signal idle.
        await loop.run_in_executor(None, sess.agent.drain)
        sess.emit(
            {
                "type": "memory.snapshot",
                "actor": "memory",
                "turn": sess.turn,
                "data": {"rows": memory_snapshot(sess.agent)},
            }
        )
        sess.emit({"type": "turn.done", "actor": "system", "turn": sess.turn, "data": {}})

    asyncio.create_task(_finish())
    return {"ack": ack}


@app.post("/api/deep_research/skip")
async def api_dr_skip(req: DeepResearchReq):
    """v0.7: UI 'Skip' — cancel the pending deep-research proposal."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    ok = sess.agent.cancel_deep_research()
    return {"ok": bool(ok)}


class SelectReq(BaseModel):
    session_id: str
    models: dict


@app.post("/api/select_models")
async def api_select_models(req: SelectReq):
    """Live-update the per-role model selection mid-session (takes effect next turn)."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    sess.models.update(req.models or {})
    return {"ok": True, "models": sess.models}


def _md_text(text) -> str:
    """Inline short text as-is; fence multiline LLM text as a blockquote."""
    text = str(text or "").strip()
    if not text:
        return "—"
    if "\n" in text:
        return "\n" + "\n".join("> " + line for line in text.splitlines())
    return text


def _one(text) -> str:
    """Collapse whitespace for one-line markdown fields."""
    return " ".join(str(text or "").split())


def _tok_pair(pt, ct) -> str:
    """'in/out' token pair; '—' when usage is unknown or zero (never '?/?')."""
    if not pt and not ct:
        return "—"
    return f"{int(pt or 0)}/{int(ct or 0)}"


def _model_label(spec) -> str:
    if isinstance(spec, dict):
        return f"{spec.get('provider', '?')}/{spec.get('model', '?')}"
    return str(spec) if spec else "—"


def build_export_markdown(sess: Session) -> str:
    """Rebuild the session from events_log as a human-readable debugging doc:
    per turn — user msg, LLM-1 reply, what LLM-2/LLM-3 did, deep-research
    activity, the A/B baseline reply, and token/latency numbers."""
    by_turn: dict[int, list[dict]] = {}
    for ev in list(sess.events_log):
        if isinstance(ev, dict):
            by_turn.setdefault(int(ev.get("turn") or 0), []).append(ev)

    def first(evts, typ):
        return next((e for e in evts if e.get("type") == typ), None)

    def last(evts, typ):
        return next((e for e in reversed(evts) if e.get("type") == typ), None)

    def data(ev) -> dict:
        d = (ev or {}).get("data")
        return d if isinstance(d, dict) else {}

    models = sess.models or {}
    lines = [
        f"# Sherlock session export — {sess.sid}",
        "",
        f"exported: {datetime.datetime.now().isoformat(timespec='seconds')}"
        f" · models: main={_model_label(models.get('main'))},"
        f" summary={_model_label(models.get('summary'))},"
        f" inference={_model_label(models.get('inference'))}",
    ]

    for turn in sorted(t for t in by_turn if t > 0):
        evts = by_turn[turn]
        base = data(last(evts, "baseline.reply"))
        user_text = data(first(evts, "turn.start")).get("user_text") or base.get("user_text", "")
        lines += ["", f"## Turn {turn}", f"**User:** {_md_text(user_text)}"]

        done = last(evts, "turn.completed")
        d = data(done)
        lat = data(last(evts, "sherlock.latency")).get("latency_ms")
        lat_str = f" · ⏱ {lat}ms" if lat is not None else ""
        lines.append(f"**Sherlock (LLM-1){lat_str}:** {_md_text(d.get('response_text'))}")
        if done is not None:
            lines.append(
                f"- tokens in/out: {_tok_pair(d.get('prompt_tokens'), d.get('completion_tokens'))}"
                f" · cache read: {d.get('cache_read_tokens', 0)}"
            )
        if base:
            srch = " (+web search)" if base.get("searched") else ""
            lines.append(
                f"**Single LLM{srch}** (⏱ {base.get('latency_ms', '—')}ms · tokens"
                f" {_tok_pair(base.get('prompt_tokens'), base.get('completion_tokens'))}):"
                f" {_md_text(base.get('text') or base.get('error'))}"
            )

        c = data(last(evts, "compact.done"))
        if c:
            kw = ", ".join(str(k) for k in (c.get("retrieval_keywords") or []))
            lines.append(
                f'**LLM-2 (compaction):** summary: "{_one(c.get("summary"))}"'
                f" · {len(c.get('facts') or [])} facts"
                f" · corrections: {len(c.get('corrections') or [])}"
                f" · keywords: [{kw}]"
            )
        else:
            lines.append("**LLM-2 (compaction):** —")

        i = data(last(evts, "infer.done"))
        if i:
            hyps = i.get("hypotheses") or []
            top = hyps[0] if hyps else {}
            parts = [f'top: "{_one(top.get("intent", "—"))}" (p={top.get("probability", "?")})']
            # really_asking / implied_chain / anticipated_next are being added
            # concurrently by another engineer — include them only IF present.
            if i.get("really_asking"):
                parts.append(f'really asking: "{_one(i["really_asking"])}"')
            if i.get("implied_chain"):
                parts.append("chain: " + " → ".join(_one(s) for s in i["implied_chain"]))
            if i.get("anticipated_next"):
                qs = "; ".join(
                    _one(a.get("question")) if isinstance(a, dict) else _one(a)
                    for a in i["anticipated_next"]
                )
                parts.append(f"anticipated: {qs}")
            lines.append("**LLM-3 (inference):** " + " · ".join(parts))
        else:
            lines.append("**LLM-3 (inference):** —")

        if any(str(e.get("type", "")).startswith("deep_research.") for e in evts):
            lines.append("**Deep research:**")
            strat = data(last(evts, "deep_research.strategy"))
            if strat.get("objective"):
                lines.append(f"- strategy: {_one(strat['objective'])}")
            for r in (e for e in evts if e.get("type") == "deep_research.round"):
                rd = data(r)
                key = rd.get("key_finding") or rd.get("summary") or f"{rd.get('hits', 0)} hits"
                lines.append(f"- R{rd.get('round', '?')}: {_one(key)}")
            stop = data(last(evts, "deep_research.synthesizing")) or data(
                last(evts, "deep_research.documents")
            )
            if stop.get("stop_reason"):
                lines.append(f"- stop reason: {stop['stop_reason']}")
            answer = data(last(evts, "deep_research.done")).get("answer")
            if answer:
                lines.append(f"- final answer: {_md_text(str(answer)[:800])}")
            t = data(last(evts, "deep_research.tokens"))
            if t.get("calls"):
                by = " · ".join(
                    f"{k} {v.get('in', 0)}/{v.get('out', 0)}"
                    for k, v in (t.get("by_stage") or {}).items()
                    if isinstance(v, dict)
                )
                lines.append(
                    f"- tokens: {t['calls']} calls · in {t.get('in', 0)} / out {t.get('out', 0)}"
                    + (f" — {by}" if by else "")
                )

    tot = {"in": 0, "out": 0, "cached": 0}
    for ev in sess.events_log:
        if isinstance(ev, dict) and ev.get("type") == "llm.call":
            d = data(ev)
            tot["in"] += d.get("prompt_tokens") or 0
            tot["out"] += d.get("completion_tokens") or 0
            tot["cached"] += d.get("cache_read_tokens") or 0
    mem_count = "?"
    try:
        mem_count = len(sess.agent.memory.list(conversation_id=sess.agent.conversation_id))
    except Exception:
        pass
    lines += [
        "",
        "## Session totals",
        f"- sherlock tokens in/out: {tot['in']}/{tot['out']} (cached {tot['cached']})",
        f"- single tokens in/out: {sess.baseline_tokens.get('in', 0)}"
        f"/{sess.baseline_tokens.get('out', 0)}",
        f"- turns: {sess.turn}",
        f"- memory entries: {mem_count}",
        "",
    ]
    return "\n".join(lines)


@app.get("/api/export")
async def api_export(session_id: str):
    """The whole session as a readable markdown doc (for handing to a coding
    agent / debugging) — built from the session's captured event log."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"error": "no such session"}
    md = build_export_markdown(sess)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="sherlock-session-{sess.sid}.md"'},
    )


@app.websocket("/ws/{sid}")
async def ws(websocket: WebSocket, sid: str):
    await websocket.accept()
    sess = SESSIONS.get(sid)
    if sess is None:
        await websocket.send_json({"type": "error", "data": {"message": "no such session"}})
        await websocket.close()
        return
    try:
        while True:
            event = await sess.queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    except Exception:
        return


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
