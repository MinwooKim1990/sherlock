"""Per-session state for the playground: the Sherlock agent + a thread-safe
event bus that forwards core/companion events to the browser over a WebSocket.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    sid: str
    models: dict  # per role: {"provider": "...", "model": "..."} (or legacy bare string)
    loop: Any  # asyncio event loop (captured at session creation)
    queue: Any  # asyncio.Queue bound to that loop
    providers: dict = field(
        default_factory=dict
    )  # {provider: {api_key, base_url}} — server-side only
    agent: Any = None
    storage_dir: str = ""
    turn: int = 0
    settings: dict = field(default_factory=dict)
    system_prompt: str = ""  # the user's LLM-1 persona — reused verbatim by the A/B baseline
    baseline_history: list = field(default_factory=list)  # plain [{role, content}] turns
    baseline_tokens: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    events_log: list = field(default_factory=list)  # every emitted event, for /api/export
    _baseline_engine: Any = None  # lazy search engine for the fair A/B baseline

    EVENTS_LOG_CAP = 20_000

    def emit(self, event: dict) -> None:
        """Push an event onto the loop-bound asyncio queue from ANY thread.

        Called by the wrapped provider callables AND by the Sherlock core's event
        sink (main thread + background companion thread). Cross-thread-safe via
        ``call_soon_threadsafe``; best-effort (never raises into a turn). Every
        event is ALSO appended to ``events_log`` (oldest dropped beyond the cap)
        so /api/export can rebuild the session as a markdown document.
        """
        try:
            self.events_log.append(event)
            if len(self.events_log) > self.EVENTS_LOG_CAP:
                del self.events_log[: -self.EVENTS_LOG_CAP]
        except Exception:
            pass
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except Exception:
            pass


def build_agent(session: Session, system_prompt: str, settings: dict):
    """Construct the Sherlock agent with three provider-backed callables + the
    event sink. Runs in a thread (fastembed model load can be slow on first use).
    """
    from sherlock import Sherlock

    from playground.providers import make_role_callable

    session.settings = settings or {}
    session.system_prompt = system_prompt or "You are a helpful assistant."
    storage = tempfile.mkdtemp(prefix="sherlock_pg_")
    session.storage_dir = storage

    main_cb = make_role_callable("main", session, session.emit)
    summary_cb = make_role_callable("summary", session, session.emit)
    inference_cb = make_role_callable("inference", session, session.emit)

    # Search engine: DuckDuckGo (free, no key) by default; brave/tavily/valyu
    # use the api key the user typed in the UI. "off" disables search. The same
    # engine powers BOTH LLM-1 (search/fetch tags) and LLM-3 (freshness search).
    engine = settings.get("search_engine", "duckduckgo")
    if engine in (None, "", "off", "none"):
        engine = None
    agent = Sherlock.with_callable(
        main_chat=main_cb,
        summary_chat=summary_cb,
        inference_chat=inference_cb,
        system_prompt=system_prompt or "You are a helpful assistant.",
        storage_dir=storage,
        embedding=settings.get("embedding", "local"),
        background=settings.get("background", True),
        redact_secrets=settings.get("redact_secrets", True),
        main_search_engine=engine,
        inference_search_engine=engine,
        search_api_key=settings.get("search_api_key") or None,
        # v1.5 Stage 1: deterministic perception layer ON in the playground so
        # OBSERVED/PRIOR observations surface for human verification.
        perception=settings.get("perception", True),
        # v1.5 Stage 2: evidence-grounded LLM-3 — feed perception to LLM-3, cap
        # uncited hypotheses, and let premise-conflicts trigger a web check.
        evidence_grounding=settings.get("evidence_grounding", True),
        premise_conflict=settings.get("premise_conflict", True),
        # v1.5 Stage 3: LLM-2 memory-consistency — code-first (fast, inline).
        memory_consistency_check=settings.get("memory_consistency_check", "code"),
        # v1.5 Stage 4: recursive inference notebook (background-only, bounded).
        inference_notebook=settings.get("inference_notebook", True),
        notebook_max_rounds=settings.get("notebook_max_rounds", 3),
        # v1.6: dynamic companion gating. "cold_start" (default) = cheap, escalate
        # on signal pressure; "turbo" = the prior all-on; "off" = legacy.
        companions_mode=settings.get("companions_mode", "cold_start"),
    )
    # v1.11: expose the deep-research VERIFY tier (off | faithfulness |
    # faithfulness+web) so the accuracy layer can be A/B'd live in the playground.
    # config.search.deep_research_verify is read fresh per research run; invalid
    # values fall through to the library default ("faithfulness").
    _vt = settings.get("deep_research_verify", "faithfulness")
    if _vt in ("off", "faithfulness", "faithfulness+web"):
        agent.config.search.deep_research_verify = _vt
    agent.set_event_sink(session.emit)
    session.agent = agent
    return agent


def memory_snapshot(agent) -> list[dict]:
    """Full current memory table (already redacted at the store) for the UI."""
    rows: list[dict] = []
    try:
        entries = agent.memory.list(conversation_id=agent.conversation_id)
    except Exception:
        return rows
    # v1.12 F7: pre-conversation (scope None) list() returns every scope,
    # including the long-term sentinel; exclude it so the snapshot only shows
    # the active conversation. The sentinel is a rag_channel-only read door.
    if agent.conversation_id is None:
        from sherlock.memory.entry import LTM_CONVERSATION_ID

        entries = [e for e in entries if e.conversation_id != LTM_CONVERSATION_ID]
    for m in entries:
        rows.append(
            {
                "id": m.id,
                "content": m.content,
                "type": getattr(m.type, "value", str(m.type)),
                "source": getattr(m.source, "value", str(m.source)),
                "state": getattr(m.state, "value", str(m.state)),
                "pinned": bool(m.pinned),
                "confidence": round(float(m.confidence), 2),
                "use_count": m.use_count,
                "last_used_turn": m.last_used_turn_index,
                "tags": m.tags or "",
                "evidence": m.evidence or "",
                "triple": [
                    m.semantic_triple_subject,
                    m.semantic_triple_relation,
                    m.semantic_triple_object,
                ],
            }
        )
    # Pinned first, then most-recently-used.
    rows.sort(key=lambda r: (not r["pinned"], -r["last_used_turn"]))
    return rows


def carry_snapshot(agent) -> dict:
    """The pending hypotheses + freshness results that will seed the NEXT turn."""
    return {
        "hypotheses": list(getattr(agent, "_pending_hypotheses", []) or []),
        "search_results": list(getattr(agent, "_pending_search_results", []) or []),
    }
