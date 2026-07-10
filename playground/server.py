"""FastAPI backend for the Sherlock Live Inspector.

Endpoints:
  POST /api/models   {provider, api_key, base_url}   -> live model list for ONE provider
  POST /api/session  {providers, models, system_prompt, settings} -> {session_id}
  POST /api/chat     {session_id, message, mode}     -> {reply, latency_ms, baseline?}
                     mode: sherlock (default) | single (bare LLM only) | both (A/B)
  GET  /api/export   ?session_id=...                 -> the session as a markdown file
  WS   /ws/{sid}                                      -> per-session event stream
  GET  /                                              -> the single-page UI

Providers: gemini | openai | anthropic | deepinfra | together | openrouter |
local (any OpenAI-compatible server — Ollama, LM Studio, vLLM...). The three
open-source-model aggregators are descriptor-driven (see playground/providers.py
``OPENAI_COMPAT``). API keys stay in the server-side Session and are never
echoed back to the browser.

Run:  python -m uvicorn playground.server:app --reload   (then open http://localhost:8000)
"""

from __future__ import annotations

import asyncio
import datetime
import os
import shutil
import tempfile
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


def _safe_to_rmtree(path: str) -> bool:
    """v1.12 Stage A5: guard the eviction rmtree so it can ONLY delete throwaway
    session tempdirs, never a persistent long-term-memory PROFILE dir.

    A profile lives under ``~/.sherlock_playground/<profile>/`` and is SHARED by
    every session on that profile — deleting it on eviction/close would wipe the
    user's durable memory. So: explicitly REFUSE anything under the playground
    long-term root, and only permit paths that are inside the OS temp dir or
    carry the ``sherlock_pg_`` tempdir prefix (what ``build_agent`` uses when
    long-term memory is off).
    """
    if not path:
        return False
    # v1.12 F6: resolve symlinks (realpath, not abspath) on BOTH the candidate and
    # the roots — a symlinked temp/profile path must not slip past the refuse-first
    # profile guard. Ordering is unchanged: REFUSE a profile before permitting.
    p = os.path.realpath(path)
    ltm_root = os.path.realpath(os.path.join(os.path.expanduser("~"), ".sherlock_playground"))
    if p == ltm_root or p.startswith(ltm_root + os.sep):
        return False  # a persistent profile — never delete on evict/close
    tmp = os.path.realpath(tempfile.gettempdir())
    if p.startswith(tmp + os.sep):
        return True
    return "sherlock_pg_" in p


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
        # v1.12 F3: release the evicted agent's SQLite engine (pooled connections
        # otherwise linger until GC) so a NEW session reopening the SAME profile
        # doesn't hit "database is locked". The store, storage and prompt store
        # all share one engine, so a single dispose covers them. Best-effort.
        try:
            evicted.agent.memory._engine.dispose()
        except Exception:
            pass
        # v1.12 Stage A5: only reclaim throwaway tempdirs — a persistent
        # long-term-memory profile dir survives eviction (see _safe_to_rmtree).
        if _safe_to_rmtree(evicted.storage_dir):
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
    # LLM-1 has already answered (its turn.completed event streamed to the
    # browser). The companions (LLM-2/LLM-3) keep running in the agent's OWN
    # background thread — so DON'T block this response on them. Drain + snapshot
    # in a detached task; the user gets the reply (and the composer) back
    # immediately, and the memory/carry panels update over WS when companions
    # finish. (Blocking here was what made "Sherlock thinking" linger and lock
    # the input until the background work was done.)
    turn_no = sess.turn

    async def _finish():
        if not sess.agent.is_deep_researching:
            await loop.run_in_executor(None, sess.agent.drain)
        sess.emit(
            {
                "type": "memory.snapshot",
                "actor": "memory",
                "turn": turn_no,
                "data": {"rows": memory_snapshot(sess.agent)},
            }
        )
        sess.emit(
            {
                "type": "carry.snapshot",
                "actor": "carry",
                "turn": turn_no,
                "data": carry_snapshot(sess.agent),
            }
        )
        sess.emit({"type": "turn.done", "actor": "system", "turn": turn_no, "data": {}})

    asyncio.create_task(_finish())
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


class StopReq(BaseModel):
    session_id: str


@app.post("/api/stop")
async def api_stop(req: StopReq):
    """Stop button: cooperatively halt the current turn — stops the streaming
    reply + further tool rounds, skips this turn's companions, and cancels any
    pending deep research. Takes effect at the next token/round boundary."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    sess.agent.request_stop()
    return {"ok": True}


class CompanionsReq(BaseModel):
    session_id: str
    mode: str  # off | cold_start | turbo


@app.post("/api/companions")
async def api_companions(req: CompanionsReq):
    """Live-switch the companion gating mode mid-session (takes effect next turn).
    The gate reads ``config.companions.mode`` each turn, so this is immediate; in
    turbo we also flip the force-companions tag so both panels fill every turn."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    if req.mode not in ("off", "cold_start", "turbo"):
        return {"error": f"invalid mode: {req.mode}"}
    sess.agent.config.companions.mode = req.mode
    sess.settings["force_companions"] = req.mode == "turbo"
    return {"ok": True, "mode": req.mode}


class BackgroundReq(BaseModel):
    session_id: str
    on: bool


@app.post("/api/background")
async def api_background(req: BackgroundReq):
    """Live-switch async (background companions) on/off mid-session. chat() reads
    ``agent._background_enabled`` fresh each turn, so this takes effect on the
    NEXT turn: ON → the LLM-1 reply returns immediately and LLM-2/LLM-3 + decay
    run in the background worker; OFF → companions run inline (the reply waits on
    them). Either way the composer is freed at turn.completed."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    sess.agent._background_enabled = bool(req.on)
    sess.settings["background"] = bool(req.on)
    return {"ok": True, "on": bool(req.on)}


class VerifyReq(BaseModel):
    session_id: str
    tier: str  # off | faithfulness | faithfulness+web


@app.post("/api/verify")
async def api_verify(req: VerifyReq):
    """Live-switch the deep-research VERIFY tier mid-session (takes effect on the
    NEXT research run). config.search.deep_research_verify is read fresh per run:
    off = skip the LLM-2 accuracy pass; faithfulness = the no-web report-vs-raw
    check + whole-report consistency sweep (default); faithfulness+web = ALSO
    re-verify the flagged claims via an LLM-3 web search."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    if req.tier not in ("off", "faithfulness", "faithfulness+web"):
        return {"error": f"invalid tier: {req.tier}"}
    sess.agent.config.search.deep_research_verify = req.tier
    sess.settings["deep_research_verify"] = req.tier
    return {"ok": True, "tier": req.tier}


class VisualizationReq(BaseModel):
    session_id: str
    on: bool


@app.post("/api/visualization")
async def api_visualization(req: VisualizationReq):
    """v1.12 Stage B4: live-flip the LLM-4 inline visualizer on/off mid-session.
    The chat marker-extraction seam and the deep-research report hook both read
    ``config.visualization.enabled`` fresh, so this takes effect on the NEXT turn:
    ON → LLM-1 may drop ``<<sherlock-viz: …>>`` markers that render into sandboxed
    charts; OFF → any stray marker stays verbatim (the byte-identical off-state).
    Mirrors /api/long_term."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    try:
        sess.agent.config.visualization.enabled = bool(req.on)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    sess.settings["visualization"] = bool(req.on)
    return {"ok": True, "on": bool(req.on)}


# ============== v1.12 Stage B3: LLM-4 visualizer runtime repair ==============
# The browser sandbox (B4) renders each ``viz.rendered`` artifact inside a
# locked-down iframe. If it throws a REAL runtime error (a JS exception, a blank
# frame), the host posts it here with the current HTML + the exact error; the
# server runs ONE LLM-4 repair with that error, re-lints the result, and returns
# the fixed HTML marked runtime-validated. Rounds are bounded PER viz_id ACROSS
# calls (session.viz_repair_rounds) so a stuck visual can't loop the model
# forever. Same localhost-trust posture as every other route in this file.


class VizRepairReq(BaseModel):
    session_id: str
    viz_id: str
    html: str
    error: str = ""


@app.post("/api/viz/repair")
async def api_viz_repair(req: VizRepairReq):
    """Run ONE LLM-4 repair round on a viz artifact that failed at RUNTIME in the
    browser sandbox. Returns ``{ok, html, validated:"runtime"}`` on a repair that
    passes the static lint; ``{ok:false, error, exhausted}`` on a lint failure or
    once the per-viz round cap is hit. Rejects an unknown session / unknown
    viz_id / a viz_id already past ``visualization.max_repair_rounds``."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"ok": False, "error": "no such session"}
    if not req.viz_id or req.viz_id not in sess.viz_ids:
        return {"ok": False, "error": "unknown viz_id"}
    if not (req.html or "").strip():
        return {"ok": False, "error": "empty html"}

    agent = sess.agent
    cfg = agent.config.visualization
    # v1.12 F3: bound the INPUT before we consume a round or hit the LLM. The 64KB
    # lint only caps the render OUTPUT; without this a 100MB POST would be received
    # in full and fed straight into the repair prompt (cost / DoS). 4× the output
    # cap leaves comfortable headroom for a legitimately large broken artifact.
    if len(req.html.encode("utf-8")) > 4 * int(agent.config.visualization.max_html_bytes):
        return {"ok": False, "error": "html too large", "exhausted": False}
    max_rounds = max(0, int(cfg.max_repair_rounds))
    used = int(sess.viz_repair_rounds.get(req.viz_id, 0))
    if used >= max_rounds:
        return {"ok": False, "error": "repair rounds exhausted", "exhausted": True}
    round_no = used + 1
    sess.note_viz_repair_round(req.viz_id, round_no)  # NICE-3: bounded write
    exhausted = round_no >= max_rounds

    anchor = f"⟦viz:{req.viz_id}⟧"
    err = (req.error or "").strip()
    errors_in = [err] if err else ["runtime error (unspecified)"]

    from sherlock import viz as _viz

    # v1.12 Stage V1: recover the per-job img-src allowlist for this viz_id (stash
    # first, then the ``.allow`` sidecar). MISSING ⇒ () — strict, so a repaired doc
    # that embeds an image but has no recoverable allowlist fails the lint. Thread
    # it into BOTH the repair generation prompt and the re-lint below.
    allowlist = agent._viz_allowlist_for(req.viz_id)
    gen_system = _viz.build_generation_system(allowlist)

    agent._emit(
        "viz.repairing",
        "llm4",
        {
            "viz_id": req.viz_id,
            "anchor": anchor,
            "round": round_no,
            "errors": errors_in,
            "runtime": True,
        },
    )

    loop = asyncio.get_running_loop()

    def _repair() -> str:
        provider = agent._viz_llm()
        raw = agent._viz_chat(provider, gen_system, _viz.build_repair_user(req.html, errors_in))
        return _viz.strip_code_fences(raw)

    try:
        # v1.12 F2: bound the provider call. The round is already consumed, so a
        # hung provider would otherwise make this HTTP request wait forever. Give
        # the render its own timeout + a small margin over the provider budget.
        fixed = await asyncio.wait_for(
            loop.run_in_executor(None, _repair), timeout=float(cfg.timeout_s) + 5.0
        )
    except asyncio.TimeoutError:
        return {"ok": False, "error": "repair timeout", "exhausted": exhausted}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "exhausted": exhausted}

    # Re-lint the repaired HTML. The prior (browser) HTML is the fidelity anchor:
    # the repair may not introduce numbers the failing artifact never contained.
    ok, lint_errors = _viz._viz_static_lint(fixed, req.html, cfg, allowlist)
    if not ok:
        return {
            "ok": False,
            "error": "; ".join(lint_errors)[:500] or "validation failed",
            "exhausted": exhausted,
        }

    validated = _viz.inject_validated_meta(fixed)
    if cfg.save_artifacts:
        try:
            agent._write_viz_artifact(req.viz_id, validated, allowlist)
        except Exception:
            pass
    agent._emit(
        "viz.rendered",
        "llm4",
        {
            "viz_id": req.viz_id,
            "anchor": anchor,
            "html": validated,
            "validated": "runtime",
            "bytes": len(validated.encode("utf-8")),
            # v1.12 F2: same conv namespacing as the agent's static-render emit.
            "conv": agent._viz_conv_component(),
        },
    )
    return {"ok": True, "html": validated, "validated": "runtime"}


@app.get("/api/viz/{viz_id}")
async def api_viz_artifact(viz_id: str, session_id: str):
    """Serve a saved LLM-4 artifact (``<storage>/viz/<conv_id>/<viz_id>.html``) as
    text/html for a reopened session to re-hydrate. The client still renders it
    inside the sandboxed iframe (B4); this endpoint just returns the bytes.
    Path-safe: the id is sanitized to one filename component and the resolved
    path is confined to the session's viz dir.

    v1.12 F2: artifacts are namespaced under a per-conversation subdir, so the
    viz dir is searched (one glob) for ``<viz_id>.html`` under any conversation —
    the endpoint only takes viz_id + session_id and a session may span several
    conversations (switch_session)."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"error": "no such session"}
    # v1.12 F6: only serve ids this session has actually registered (rejects
    # ids from other sessions / never-emitted ids with a 404-style error).
    if viz_id not in sess.viz_ids:
        return {"error": "no such artifact"}
    import re as _re

    safe = _re.sub(r"[^A-Za-z0-9._-]", "_", str(viz_id))
    base = (Path(sess.agent.config.storage.sqlite_path).resolve().parent / "viz").resolve()
    # Locate <base>/<conv>/<safe>.html across conversation subdirs (F2), keeping
    # the legacy flat <base>/<safe>.html working. rglob confines matches under
    # base; the explicit prefix check is defense-in-depth against symlinks.
    path = None
    for cand in base.rglob(f"{safe}.html"):
        rp = cand.resolve()
        if str(rp).startswith(str(base) + os.sep) and rp.is_file():
            path = rp
            break
    if path is None:
        return {"error": "no such artifact"}
    # v1.12 F6: this endpoint is for B4's sandboxed iframe; a direct visit would
    # otherwise run the artifact as first-party HTML in the playground origin. Add
    # a response-level CSP sandbox (defense-in-depth; the artifact's own meta CSP
    # already blocks exfiltration).
    return FileResponse(
        str(path),
        media_type="text/html",
        headers={"Content-Security-Policy": "sandbox allow-scripts"},
    )


# ==================== v1.12 Stage A5: long-term memory ====================
# Live toggles + a small management surface over the A1–A4 library API. The
# chat-level confirm-token flow (A3) still governs CONVERSATIONAL deletion; the
# UI buttons below are "direct" because the browser has its own click-confirm.
#
# v1.12 F7 (posture): these routes (like every endpoint in this file) trust the
# caller — the playground is a localhost-only single-user dev inspector with no
# auth anywhere. This localhost-trust posture is playground-wide, not specific to
# these memory routes; a Host-header allowlist middleware (reject a non-localhost
# Host) is a possible future hardening if this is ever exposed beyond loopback.


class LongTermReq(BaseModel):
    session_id: str
    on: bool


@app.post("/api/long_term")
async def api_long_term(req: LongTermReq):
    """Live-flip long-term memory on/off mid-session. The summarizer's promotion
    gate reads ``config.memory.long_term.enabled`` fresh each cycle, so this
    takes effect on the NEXT turn's compaction."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    try:
        sess.agent.config.memory.long_term.enabled = bool(req.on)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    sess.settings["long_term"] = bool(req.on)
    return {"ok": True, "on": bool(req.on)}


@app.post("/api/incognito")
async def api_incognito(req: LongTermReq):
    """Live-flip incognito: suppress long-term WRITES (promotions) while leaving
    reads/recall intact — a 'pause remembering' switch. Takes effect next turn."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    try:
        sess.agent.config.memory.long_term.incognito = bool(req.on)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    sess.settings["ltm_incognito"] = bool(req.on)
    return {"ok": True, "on": bool(req.on)}


def _ltm_iso(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@app.get("/api/memory/long_term")
async def api_ltm_snapshot(session_id: str):
    """The live cross-conversation long-term memory (sentinel scope), newest
    first — reading is harmless regardless of enabled/incognito."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"error": "no such session"}
    try:
        rows = sess.agent.long_term_memory()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "rows": [
            {
                "id": r.get("id"),
                "category": r.get("category"),
                "content": r.get("content"),
                "confidence": round(float(r.get("confidence") or 0.0), 2),
                "created_at": _ltm_iso(r.get("created_at")),
                "origin": r.get("origin_conversation_id"),
            }
            for r in rows
        ]
    }


class MemDeleteReq(BaseModel):
    session_id: str
    id: str


@app.post("/api/memory/delete")
async def api_memory_delete(req: MemDeleteReq):
    """Hard-delete ONE long-term row by full id. The id MUST belong to the
    sentinel scope — a session/conversation row can never be deleted through
    this endpoint (that path stays behind the conversational confirm-token)."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    from sherlock.memory.entry import LTM_CONVERSATION_ID

    # v1.12 F2: honour the {"error": ...} contract even when the store raises
    # (locked/corrupt DB) instead of leaking an HTTP 500 — same shape as wipe.
    try:
        row = sess.agent.memory.get(req.id)
        if row is None:
            return {"error": "no such memory id"}
        if row.conversation_id != LTM_CONVERSATION_ID:
            return {"error": "not a long-term memory row (refusing to delete)"}
        sess.agent.memory.hard_delete(req.id)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "id": req.id}


class SessionOnlyReq(BaseModel):
    session_id: str


@app.post("/api/memory/wipe")
async def api_memory_wipe(req: SessionOnlyReq):
    """Wipe ALL long-term memory (honours the auto Markdown backup — fail-closed
    if the backup can't be written). Returns ``{removed, backup_path}``."""
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    try:
        return sess.agent.wipe_long_term()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


_EXPORT_CT = {
    "markdown": ("text/markdown", "md"),
    "md": ("text/markdown", "md"),
    "json": ("application/json", "json"),
    "sql": ("application/sql", "sql"),
}


@app.get("/api/memory/export")
async def api_memory_export(session_id: str, fmt: str = "markdown"):
    """Download the long-term memory as markdown / json / sql, with the right
    content-type + a Content-Disposition filename."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"error": "no such session"}
    fmt_norm = (fmt or "markdown").strip().lower()
    if fmt_norm not in _EXPORT_CT:
        return {"error": f"unknown format: {fmt!r} (use markdown/md, json, or sql)"}
    try:
        text = sess.agent.export_memory(fmt=fmt_norm)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    media, ext = _EXPORT_CT[fmt_norm]
    return Response(
        content=text,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="sherlock-ltm-{sess.sid}.{ext}"'},
    )


class MemImportReq(BaseModel):
    session_id: str
    text: str
    fmt: str | None = None


@app.post("/api/memory/import")
async def api_memory_import(req: MemImportReq):
    """Import long-term facts from pasted/uploaded text (json or markdown; auto
    detected when fmt is omitted). Requires long-term memory enabled — every
    fact is re-routed through the store so redaction + dedup re-apply."""
    # v1.12 F5: bound the import body so a runaway paste/upload can't wedge the
    # single-process inspector.
    if len(req.text) > 5_000_000:
        return {"error": "import too large (5 MB max)"}
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        return {"error": "no such session"}
    # v1.12 F1 (security): agent.import_memory treats a SHORT existing string as a
    # filesystem PATH and reads it — over HTTP that turns this endpoint into an
    # arbitrary local-file-read primitive (e.g. slurp a backup, then view via
    # snapshot/export). The browser only ever posts raw export TEXT, so refuse any
    # input that resolves to an existing path before it reaches the library.
    if len(req.text) < 4096 and os.path.exists(req.text):
        return {"error": "raw export text required (path import is not available over HTTP)"}
    try:
        return sess.agent.import_memory(req.text, fmt=req.fmt)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


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

        # v1.12 Stage B4: LLM-4 visualizations rendered this turn (chat OR the DR
        # report). Each ⟦viz:id⟧ placeholder that actually rendered becomes a link
        # to its saved artifact; the description rides the earlier viz.pending event.
        viz_desc = {
            data(e).get("viz_id"): data(e).get("description", "")
            for e in evts
            if e.get("type") == "viz.pending"
        }
        # NOTE: this dedup is PER-TURN (seen_viz is reset for each turn), so a viz
        # that is re-rendered at runtime (e.g. an /api/viz/repair re-render) and
        # re-recorded under a later turn will emit a second link for the same vid.
        # Also the target is RELATIVE: it only resolves when this export is written
        # beside the session's viz storage dir. v1.12 F2: the artifact lives under a
        # per-conversation subdir (viz/<conv>/<vid>.html) — the ``conv`` component
        # rides the viz.rendered event.
        seen_viz: set[str] = set()
        for e in evts:
            if e.get("type") != "viz.rendered":
                continue
            vid = data(e).get("viz_id")
            if not vid or vid in seen_viz:
                continue
            seen_viz.add(vid)
            desc = _one(viz_desc.get(vid)) or "visualization"
            conv = data(e).get("conv")
            sub = f"{conv}/" if conv else ""
            lines.append(f"- [📊 visualization: {desc}](viz/{sub}{vid}.html)")
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
