"""Memory tool (v0.4.0) — on-demand active recall from Sherlock memory.

LLM-1 can request a lookup by emitting one of:

    <<sherlock-tool: memory lookup "Yujin allergy">>   # semantic + entity
    <<sherlock-tool: memory entity "Yujin">>            # entity-only deterministic
    <<sherlock-tool: memory timeline last 10>>          # recent raw turns
    <<sherlock-tool: memory pinned>>                    # all pinned facts

v1.12 Stage A3 adds natural-language MANAGEMENT of cross-conversation LONG-TERM
memory (the reserved ``LTM_CONVERSATION_ID`` sentinel scope). These verbs are
gated on ``config.memory.long_term.enabled`` — REJECTED (error, no mutation)
when the feature is off — and every deletion is protected by a code-level
single-use confirm token that the model can never forge:

    <<sherlock-tool: memory profile>>                  # what do I remember?
    <<sherlock-tool: memory save "always use metric">> # explicit remember-this
    <<sherlock-tool: memory update ab12 "corrected">>  # supersede one durable fact
    <<sherlock-tool: memory forget allergy>>           # PREVIEW → pending + token
    <<sherlock-tool: memory forget-confirm <token>>>   # execute the frozen delete
    <<sherlock-tool: memory wipe>>                      # PREVIEW ALL → count + token
    <<sherlock-tool: memory wipe-confirm <token>>>     # execute the wipe

The same handlers are exposed as plain callables for users on native
tool-calling, plus schema generators for OpenAI / Anthropic tools.
"""

from __future__ import annotations

import json as _json
import re
import secrets
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sherlock.memory.entry import MemoryEntry
    from sherlock.memory.store import MemoryStore
    from sherlock.rag.hybrid import HybridSearch
    from sherlock.storage import Storage


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

# v1.12 A3: the hyphenated confirm verbs MUST precede their bare prefixes in the
# alternation (regex alternation is leftmost-first) so "forget-confirm" is not
# mis-parsed as "forget" with a "-confirm …" argument.
_PAYLOAD_RE = re.compile(
    r"^(lookup|entity|timeline|pinned|profile|save|update|"
    r"forget-confirm|forget|wipe-confirm|wipe)\b\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_memory_payload(payload: str) -> tuple[str, str]:
    """Split a memory-tool payload into (kind, args_string).

    Returns ``("", "")`` when the payload is malformed.
    """
    payload = (payload or "").strip()
    m = _PAYLOAD_RE.match(payload)
    if not m:
        return "", ""
    return m.group(1).lower(), (m.group(2) or "").strip()


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] == s[0]:
        return s[1:-1].strip()
    return s


def _entry_to_dict(e: "MemoryEntry") -> dict:
    # v1.0: a superseded row may still surface via deliberate recall
    # (entity / list paths) — mark it so LLM-1 never quotes it as current.
    # v1.1 R34 (bi-temporal): when supersede() recorded the invalidation
    # turn, surface it — "(superseded at t7)" lets LLM-1 answer
    # "what was true before turn X?" from deliberate recall.
    content = e.content
    if getattr(e, "superseded_by", None):
        invalid_at = getattr(e, "invalid_at_turn", None)
        if invalid_at is not None:
            content = f"{content} (superseded at t{invalid_at})"
        else:
            content = f"{content} (superseded)"
    return {
        "id": e.id,
        "type": getattr(e.type, "value", str(e.type)),
        "source": getattr(e.source, "value", str(e.source)),
        "content": content,
        "confidence": e.confidence,
        "pinned": e.pinned,
        "tags": e.tags,
        "state": getattr(e.state, "value", str(e.state)),
        "last_used_turn_index": e.last_used_turn_index,
        # v1.1 R31 — structured memory reading: expose the entry's temporal
        # coordinates so LLM-1 can reason about when a fact was learned and
        # last touched. Additive; "last_used_turn" aliases the legacy
        # "last_used_turn_index" key (kept unchanged for compatibility).
        "created_turn": getattr(e, "created_turn_index", 0),
        "last_used_turn": e.last_used_turn_index,
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def memory_lookup(
    query: str,
    *,
    store: "MemoryStore",
    hybrid: "HybridSearch",
    conversation_id: str | None = None,
    top_k: int = 8,
) -> list[dict]:
    """Tier 2 (entity-indexed) → Tier 4 (RAG) fallback.

    Returns up to ``top_k`` memories. Entity matches dominate semantic
    noise because of the entity-precision boost in HybridSearch.
    """
    if not query:
        return []
    hits = hybrid.search(query, conversation_id=conversation_id, top_k=top_k)
    # v1.12 F7: an unscoped lookup (no conversation) runs hybrid's vector tier
    # over EVERY scope, including the long-term sentinel — but the sentinel is
    # only meant to be read through the rag_channel. Drop those rows here so a
    # pre-conversation lookup can't leak durable facts.
    if conversation_id is None:
        from sherlock.memory.entry import LTM_CONVERSATION_ID

        hits = [(e, s) for e, s in hits if e.conversation_id != LTM_CONVERSATION_ID]
    return [_entry_to_dict(e) | {"score": float(score)} for e, score in hits]


def memory_entity(
    entity: str,
    *,
    store: "MemoryStore",
    conversation_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Strictly deterministic — only entries whose entity pool contains
    the queried entity (case-insensitive).
    """
    from sherlock.memory.entry import MemoryState
    from sherlock.rag.hybrid import _entry_entity_pool, extract_entities

    if not entity:
        return []
    targets = extract_entities(entity)
    if not targets:
        targets = {entity.lower()}
    # v0.5.1: hit the persistent entity index instead of scanning every row.
    # find_by_entities is conversation-scoped, so fall back to a full scan only
    # when no conversation is given. We still re-verify the live entity pool
    # (the index may be coarser) and filter FORGOTTEN — same shape hybrid.py uses.
    finder = getattr(store, "find_by_entities", None)
    if conversation_id is not None and callable(finder):
        entries = finder(conversation_id, targets)
    else:
        entries = store.list(conversation_id=conversation_id)
    # v1.12 F7: an unscoped entity scan sees the long-term sentinel scope too;
    # keep it a rag_channel-only door by excluding sentinel rows when unscoped.
    ltm_scope: str | None = None
    if conversation_id is None:
        from sherlock.memory.entry import LTM_CONVERSATION_ID

        ltm_scope = LTM_CONVERSATION_ID
    hits: list["MemoryEntry"] = []
    for e in entries:
        if e.state == MemoryState.FORGOTTEN:
            continue
        if ltm_scope is not None and e.conversation_id == ltm_scope:
            continue
        if _entry_entity_pool(e) & targets:
            hits.append(e)
    hits.sort(key=lambda e: (not e.pinned, -e.last_used_turn_index))
    return [_entry_to_dict(e) for e in hits[:limit]]


def memory_timeline(
    n: int,
    *,
    storage: "Storage",
    conversation_id: str,
) -> list[dict]:
    """Return the last ``n`` raw messages (user + assistant) for the
    given conversation, oldest first.
    """
    msgs = storage.list_messages(conversation_id)
    non_sys = [m for m in msgs if m.role != "system"]
    tail = non_sys[-n:] if n > 0 else non_sys
    return [{"role": m.role, "content": m.content, "created_at": str(m.created_at)} for m in tail]


def memory_pinned(
    *,
    store: "MemoryStore",
    conversation_id: str | None = None,
) -> list[dict]:
    """All pinned memories in the conversation, including persona summary."""
    entries = store.list(conversation_id=conversation_id, pinned=True)
    # v1.0: superseded rows are never "current pinned truth", even if a
    # stale pin flag survives somewhere — exclude them outright.
    entries = [e for e in entries if not getattr(e, "superseded_by", None)]
    # v1.12 F7: an unscoped pinned dump would include the always-pinned
    # long-term sentinel rows; the sentinel is a rag_channel-only door.
    if conversation_id is None:
        from sherlock.memory.entry import LTM_CONVERSATION_ID

        entries = [e for e in entries if e.conversation_id != LTM_CONVERSATION_ID]
    return [_entry_to_dict(e) for e in entries]


# ---------------------------------------------------------------------------
# v1.12 Stage A3: long-term memory MANAGEMENT (write / edit / delete)
# ---------------------------------------------------------------------------


class LTMToolContext:
    """Per-dispatch context for the long-term memory management verbs.

    The memory_tool module is deliberately stateless, so the AGENT owns the
    single-use confirm-token store (a plain ``dict``) and hands a fresh context
    into every ``dispatch_memory`` call. The context carries:

      * the ``enabled`` / ``incognito`` gates (management verbs are rejected when
        the feature is off; WRITES are additionally rejected under incognito);
      * the current ``turn_index`` (used to time-stamp + expire tokens);
      * a reference to the agent-owned ``pending`` dict, so mint/consume mutate
        agent state without the module holding any of its own.

    Token contract (CODE-LEVEL safety — never trusts the model):
      * a token is short random hex minted per PREVIEW;
      * minting a new preview INVALIDATES every prior token (latest wins);
      * a token is SINGLE-USE (popped on consume) and expires after
        ``TOKEN_TTL_TURNS`` turns (``turn_index - minted_turn``);
      * a wrong / expired / reused / wrong-kind token consumes nothing and
        mutates nothing.
    """

    TOKEN_TTL_TURNS = 2

    def __init__(
        self,
        *,
        enabled: bool,
        incognito: bool,
        turn_index: int,
        pending: dict,
    ) -> None:
        self.enabled = bool(enabled)
        self.incognito = bool(incognito)
        self.turn_index = int(turn_index)
        # token -> {"kind": "delete"|"wipe", "ids": list[str]|None, "minted_turn": int}
        self.pending = pending

    def mint(self, kind: str, ids: Optional[list[str]]) -> str:
        """Freeze a pending destructive action and return its confirm token.

        Clears every prior token first — the latest preview always wins, so a
        stale token from an earlier turn can never be confirmed after the user
        re-scopes the request.
        """
        self.pending.clear()
        token = secrets.token_hex(4)
        self.pending[token] = {
            "kind": kind,
            "ids": list(ids) if ids is not None else None,
            "minted_turn": self.turn_index,
        }
        return token

    def consume(self, token: str, kind: str) -> tuple[Optional[dict], Optional[str]]:
        """Validate + single-use-consume ``token`` for ``kind``.

        Returns ``(record, None)`` on success (record already removed) or
        ``(None, error_message)`` — never mutating anything on failure.
        """
        token = (token or "").strip()
        rec = self.pending.get(token)
        if rec is None:
            return None, "invalid or already-used confirm token — run the preview again"
        if rec.get("kind") != kind:
            return None, "confirm token does not match this action"
        if self.turn_index - int(rec.get("minted_turn", self.turn_index)) > self.TOKEN_TTL_TURNS:
            self.pending.pop(token, None)
            return None, "confirm token expired — run the preview again"
        self.pending.pop(token, None)  # single use
        return rec, None


def _require_ltm(
    ltm_ctx: Optional[LTMToolContext], kind: str, *, need_write: bool = False
) -> Optional[dict]:
    """Gate a management verb. Returns an error dict, or ``None`` when allowed."""
    if ltm_ctx is None or not ltm_ctx.enabled:
        return {"tool": "memory", "kind": kind, "error": "long-term memory is disabled"}
    if need_write and ltm_ctx.incognito:
        return {
            "tool": "memory",
            "kind": kind,
            "error": "long-term memory is in incognito mode (durable writes paused)",
        }
    return None


def _live_ltm_rows(store: "MemoryStore") -> list["MemoryEntry"]:
    """Current (non-superseded, non-forgotten) long-term sentinel rows."""
    from sherlock.memory.entry import LTM_CONVERSATION_ID, MemoryState

    return [
        e
        for e in store.list(conversation_id=LTM_CONVERSATION_ID)
        if not getattr(e, "superseded_by", None) and e.state != MemoryState.FORGOTTEN
    ]


def memory_profile(*, store: "MemoryStore", limit: int = 50) -> list[dict]:
    """List the currently-remembered long-term facts (the "what do you remember?"
    view), newest first. Read-only; superseded/forgotten rows are excluded."""
    from sherlock.memory.entry import ltm_category

    rows = _live_ltm_rows(store)
    rows.sort(key=lambda e: e.created_at, reverse=True)
    return [
        {
            "id": e.id[:8],
            "category": ltm_category(e.tags),
            "content": e.content,
            "confidence": e.confidence,
            "created": str(e.created_at),
        }
        for e in rows[:limit]
    ]


def memory_save(
    text: str,
    *,
    store: "MemoryStore",
    conversation_id: str | None,
    ltm_ctx: Optional[LTMToolContext],
) -> dict:
    """EXPLICIT user-directive save: land ``text`` in the long-term sentinel
    scope immediately (category ``user_directive``, pinned). Blocked when the
    feature is disabled or the session is incognito."""
    err = _require_ltm(ltm_ctx, "save", need_write=True)
    if err:
        return err
    text = (text or "").strip()
    if not text:
        return {"tool": "memory", "kind": "save", "error": "nothing to save (empty text)"}
    from sherlock.memory.entry import LTM_CONVERSATION_ID, MemorySource, MemoryType

    row = store.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=text,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        pinned=True,
        last_used_turn_index=ltm_ctx.turn_index,
        tags="ltm,user_directive",
        evidence=_json.dumps([{"quote": text, "turn": ltm_ctx.turn_index}]),
        origin_conversation_id=conversation_id,
        dedup=True,
    )
    return {
        "tool": "memory",
        "kind": "save",
        "saved": True,
        "id": row.id[:8],
        "category": "user_directive",
        "content": row.content,
    }


def memory_update(
    args: str,
    *,
    store: "MemoryStore",
    conversation_id: str | None,
    ltm_ctx: Optional[LTMToolContext],
) -> dict:
    """Correct one durable fact: supersede the row matched by a UNIQUE id-prefix
    with a fresh row carrying ``new text`` (reusing the old row's category)."""
    err = _require_ltm(ltm_ctx, "update")
    if err:
        return err
    args = (args or "").strip()
    parts = args.split(None, 1)
    if len(parts) < 2 or not parts[0].strip():
        return {
            "tool": "memory",
            "kind": "update",
            "error": "usage: memory update <id-prefix> <corrected text>",
        }
    prefix = parts[0].strip()
    new_text = _strip_quotes(parts[1])
    if not new_text:
        return {"tool": "memory", "kind": "update", "error": "corrected text is empty"}
    from sherlock.memory.entry import LTM_CONVERSATION_ID, MemorySource, MemoryType, ltm_category

    live = _live_ltm_rows(store)
    matches = [e for e in live if e.id.startswith(prefix)]
    if not matches:
        return {
            "tool": "memory",
            "kind": "update",
            "error": f"no long-term fact has id starting with '{prefix}'",
        }
    if len(matches) > 1:
        return {
            "tool": "memory",
            "kind": "update",
            "error": "id-prefix is ambiguous; use more characters",
            "candidates": [{"id": e.id[:8], "content": e.content} for e in matches[:8]],
        }
    old = matches[0]
    category = ltm_category(old.tags)
    new_row = store.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=new_text,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=max(0.9, float(old.confidence or 0.0)),
        pinned=True,
        last_used_turn_index=ltm_ctx.turn_index,
        tags=old.tags or f"ltm,{category}",
        evidence=_json.dumps([{"quote": new_text, "turn": ltm_ctx.turn_index}]),
        origin_conversation_id=conversation_id,
        # A correction must NOT dedup-merge back into the row it replaces.
        dedup=False,
    )
    store.supersede(old.id, new_row.id, turn_index=ltm_ctx.turn_index)
    return {
        "tool": "memory",
        "kind": "update",
        "updated": True,
        "old_id": old.id[:8],
        "new_id": new_row.id[:8],
        "content": new_row.content,
    }


def memory_forget(
    query: str,
    *,
    store: "MemoryStore",
    ltm_ctx: Optional[LTMToolContext],
) -> dict:
    """PREVIEW ONLY — never mutates. Find durable facts matching ``query``
    (id-prefix, substring, or entity token; cap 8) and freeze them behind a
    single-use confirm token so ``forget-confirm`` can delete EXACTLY them."""
    err = _require_ltm(ltm_ctx, "forget")
    if err:
        return err
    query = _strip_quotes((query or "").strip())
    if not query:
        return {"tool": "memory", "kind": "forget", "error": "usage: memory forget <what>"}
    # F4 (audit): a 1-char query makes the substring channel ("e" in almost every
    # row) — and a 1-char id-prefix — match nearly everything. Refuse it and ask
    # for something specific rather than freezing an over-broad delete set.
    if len(query) < 2:
        return {
            "tool": "memory",
            "kind": "forget",
            "error": "query too short — name the fact, person, or topic to forget (>= 2 characters)",
        }
    from sherlock.memory.entry import ltm_category
    from sherlock.rag.hybrid import _entry_entity_pool, extract_entities

    ql = query.lower()
    targets = extract_entities(query) or {ql}
    rows = _live_ltm_rows(store)
    matched: list["MemoryEntry"] = []
    # F4 (audit): entity-token / id-prefix matching FIRST — the PRECISE channel
    # (e.g. "유진" hits exactly the 유진 rows, not every row that contains the
    # substring). Only if that finds nothing do we fall back to the broad
    # substring channel (already guarded to >= 2 chars above).
    for e in rows:
        if e.id.startswith(query) or (_entry_entity_pool(e) & targets):
            matched.append(e)
        if len(matched) >= 8:
            break
    if not matched:
        for e in rows:
            if ql in (e.content or "").lower():
                matched.append(e)
            if len(matched) >= 8:
                break
    if not matched:
        return {
            "tool": "memory",
            "kind": "forget",
            "pending": [],
            "count": 0,
            "message": f"no long-term memory matches '{query}'.",
        }
    token = ltm_ctx.mint("delete", [e.id for e in matched])
    return {
        "tool": "memory",
        "kind": "forget",
        "pending": [
            {"id": e.id[:8], "content": e.content, "category": ltm_category(e.tags)}
            for e in matched
        ],
        "count": len(matched),
        "confirm_token": token,
        "instruction": (
            "PREVIEW ONLY — nothing was deleted. Tell the user EXACTLY which fact(s) "
            "above will be erased and ask them to confirm in their language. Only after "
            f"they confirm, emit <<sherlock-tool: memory forget-confirm {token}>> next "
            "turn. Never invent a token."
        ),
    }


def memory_forget_confirm(
    token: str,
    *,
    store: "MemoryStore",
    ltm_ctx: Optional[LTMToolContext],
) -> dict:
    """Execute a frozen forget: hard-delete EXACTLY the ids captured at preview
    time. A wrong / expired / reused token deletes nothing."""
    err = _require_ltm(ltm_ctx, "forget-confirm")
    if err:
        return err
    rec, cerr = ltm_ctx.consume(token, "delete")
    if cerr:
        return {"tool": "memory", "kind": "forget-confirm", "error": cerr}
    ids = rec.get("ids") or []
    deleted = 0
    for mid in ids:
        try:
            if store.get(mid) is not None:
                store.hard_delete(mid)
                deleted += 1
        except Exception:
            pass
    return {"tool": "memory", "kind": "forget-confirm", "deleted": deleted}


def memory_wipe(*, store: "MemoryStore", ltm_ctx: Optional[LTMToolContext]) -> dict:
    """PREVIEW ONLY — count long-term rows and mint a wipe confirm token.

    F6 (audit): the headline ``count`` is LIVE rows only — the exact set the user
    sees via ``memory profile`` — so the preview never claims a bigger number than
    what they can see. ``total`` additionally counts the superseded/forgotten
    tombstone rows that wipe-confirm ALSO purges from the sentinel scope, so the
    two numbers stay reconcilable when they differ.
    """
    err = _require_ltm(ltm_ctx, "wipe")
    if err:
        return err
    from sherlock.memory.entry import LTM_CONVERSATION_ID

    total = len(store.list(conversation_id=LTM_CONVERSATION_ID))
    live = len(_live_ltm_rows(store))
    if total == 0:
        return {
            "tool": "memory",
            "kind": "wipe",
            "count": 0,
            "total": 0,
            "message": "long-term memory is already empty.",
        }
    token = ltm_ctx.mint("wipe", None)
    history = total - live
    history_note = (
        f" (plus {history} superseded/forgotten history row(s) also purged)" if history else ""
    )
    return {
        "tool": "memory",
        "kind": "wipe",
        "count": live,
        "total": total,
        "confirm_token": token,
        "instruction": (
            "PREVIEW ONLY — nothing was deleted. This will PERMANENTLY erase ALL "
            f"{live} remembered long-term fact(s){history_note}. Tell the user and ask them "
            f"to confirm in their language. Only after they confirm, emit "
            f"<<sherlock-tool: memory wipe-confirm {token}>> next turn. Never invent a token."
        ),
    }


def memory_wipe_confirm(
    token: str,
    *,
    store: "MemoryStore",
    ltm_ctx: Optional[LTMToolContext],
) -> dict:
    """Execute a wipe: delete the whole long-term sentinel scope."""
    err = _require_ltm(ltm_ctx, "wipe-confirm")
    if err:
        return err
    rec, cerr = ltm_ctx.consume(token, "wipe")
    if cerr:
        return {"tool": "memory", "kind": "wipe-confirm", "error": cerr}
    from sherlock.memory.entry import LTM_CONVERSATION_ID

    # TODO(Stage A4): when config.memory.long_term.auto_export_on_wipe is set,
    # export the sentinel scope to a portable file BEFORE deleting it here.
    n = store.delete_conversation_memories(LTM_CONVERSATION_ID)
    return {"tool": "memory", "kind": "wipe-confirm", "wiped": n}


# ---------------------------------------------------------------------------
# Unified dispatcher (used by the <<sherlock-tool: memory ...>> tag)
# ---------------------------------------------------------------------------


def dispatch_memory(
    payload: str,
    *,
    store: Optional["MemoryStore"] = None,
    hybrid: Optional["HybridSearch"] = None,
    storage: Optional["Storage"] = None,
    conversation_id: str | None = None,
    ltm_ctx: Optional[LTMToolContext] = None,
) -> dict:
    """Run a parsed memory payload and return a JSON-serialisable dict.

    Errors return ``{"error": "..."}`` instead of raising — the agent
    loop must stay resilient.
    """
    kind, raw_args = parse_memory_payload(payload)
    if not kind:
        return {"tool": "memory", "error": f"unrecognised payload: {payload!r}"}

    args = _strip_quotes(raw_args)

    if kind == "lookup":
        if store is None or hybrid is None:
            return {"tool": "memory", "kind": kind, "error": "memory store not available"}
        return {
            "tool": "memory",
            "kind": "lookup",
            "query": args,
            "results": memory_lookup(
                args,
                store=store,
                hybrid=hybrid,
                conversation_id=conversation_id,
            ),
        }
    if kind == "entity":
        if store is None:
            return {"tool": "memory", "kind": kind, "error": "memory store not available"}
        return {
            "tool": "memory",
            "kind": "entity",
            "entity": args,
            "results": memory_entity(args, store=store, conversation_id=conversation_id),
        }
    if kind == "timeline":
        if storage is None or conversation_id is None:
            return {"tool": "memory", "kind": kind, "error": "storage / conversation unavailable"}
        # Accept "last N", "N", or empty (default 10).
        n = 10
        m = re.search(r"(\d+)", args or "")
        if m:
            try:
                n = max(1, int(m.group(1)))
            except ValueError:
                pass
        return {
            "tool": "memory",
            "kind": "timeline",
            "n": n,
            "results": memory_timeline(n, storage=storage, conversation_id=conversation_id),
        }
    if kind == "pinned":
        if store is None:
            return {"tool": "memory", "kind": kind, "error": "memory store not available"}
        return {
            "tool": "memory",
            "kind": "pinned",
            "results": memory_pinned(store=store, conversation_id=conversation_id),
        }

    # --- v1.12 Stage A3: long-term management verbs (feature-gated) ---------
    if kind in {
        "profile",
        "save",
        "update",
        "forget",
        "forget-confirm",
        "wipe",
        "wipe-confirm",
    }:
        if store is None:
            return {"tool": "memory", "kind": kind, "error": "memory store not available"}
        if kind == "profile":
            gate = _require_ltm(ltm_ctx, "profile")
            if gate:
                return gate
            return {"tool": "memory", "kind": "profile", "results": memory_profile(store=store)}
        if kind == "save":
            return memory_save(args, store=store, conversation_id=conversation_id, ltm_ctx=ltm_ctx)
        if kind == "update":
            # NOTE: pass raw_args (not the quote-stripped `args`) — update splits
            # "<id> <text>" itself and strips quotes around the text component.
            return memory_update(
                raw_args, store=store, conversation_id=conversation_id, ltm_ctx=ltm_ctx
            )
        if kind == "forget":
            return memory_forget(args, store=store, ltm_ctx=ltm_ctx)
        if kind == "forget-confirm":
            return memory_forget_confirm(args, store=store, ltm_ctx=ltm_ctx)
        if kind == "wipe":
            return memory_wipe(store=store, ltm_ctx=ltm_ctx)
        if kind == "wipe-confirm":
            return memory_wipe_confirm(args, store=store, ltm_ctx=ltm_ctx)

    return {"tool": "memory", "error": f"unknown kind: {kind}"}


# ---------------------------------------------------------------------------
# Native tool-calling schema generators (for users not using the tag)
# ---------------------------------------------------------------------------

_MEMORY_DESCRIPTION = (
    "Read AND manage Sherlock's memory. READ verbs recall a specific fact the "
    "user shared before (allergies, names, dates, preferences) that may have "
    "fallen out of the recent K-turn window: lookup (semantic+entity), entity "
    "(deterministic), timeline (last N raw turns), pinned (all pinned facts). "
    "MANAGE verbs act on cross-conversation LONG-TERM memory: profile (list "
    "what is remembered), save (remember this fact permanently), update "
    "(correct one durable fact by id-prefix), forget (PREVIEW facts to delete "
    "+ get a confirm token), forget-confirm (execute the previewed delete only "
    "after the user confirms), wipe / wipe-confirm (all long-term memory). "
    "Deletions ALWAYS require a two-step preview→confirm with the token."
)

# v1.12 A3: the full verb set for the native-tool ``kind`` enum.
_MEMORY_KINDS = [
    "lookup",
    "entity",
    "timeline",
    "pinned",
    "profile",
    "save",
    "update",
    "forget",
    "forget-confirm",
    "wipe",
    "wipe-confirm",
]


def make_openai_memory_tool() -> list[dict]:
    """OpenAI Chat Completions tools= entry for the memory tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_lookup",
                "description": _MEMORY_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": _MEMORY_KINDS,
                            "description": "lookup/entity/timeline/pinned = read; profile = list long-term facts; save = remember permanently; update = correct by id-prefix; forget/wipe = PREVIEW a deletion (returns a confirm token); forget-confirm/wipe-confirm = execute with that token after the user confirms",
                        },
                        "args": {
                            "type": "string",
                            "description": "Query / entity / count / text / id-prefix / confirm token, depending on kind. Empty for 'pinned', 'profile', 'wipe'.",
                        },
                    },
                    "required": ["kind"],
                },
            },
        }
    ]


def make_anthropic_memory_tool() -> list[dict]:
    """Anthropic Messages tools= entry for the memory tool."""
    return [
        {
            "name": "memory_lookup",
            "description": _MEMORY_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": _MEMORY_KINDS,
                    },
                    "args": {
                        "type": "string",
                        "description": "Query / entity / count / text / id-prefix / confirm token, depending on kind.",
                    },
                },
                "required": ["kind"],
            },
        }
    ]
