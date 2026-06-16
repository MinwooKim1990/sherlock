"""Memory tool (v0.4.0) — on-demand active recall from Sherlock memory.

LLM-1 can request a lookup by emitting one of:

    <<sherlock-tool: memory lookup "Yujin allergy">>   # semantic + entity
    <<sherlock-tool: memory entity "Yujin">>            # entity-only deterministic
    <<sherlock-tool: memory timeline last 10>>          # recent raw turns
    <<sherlock-tool: memory pinned>>                    # all pinned facts

The same handlers are exposed as plain callables for users on native
tool-calling, plus schema generators for OpenAI / Anthropic tools.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sherlock.memory.entry import MemoryEntry
    from sherlock.memory.store import MemoryStore
    from sherlock.rag.hybrid import HybridSearch
    from sherlock.storage import Storage


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

_PAYLOAD_RE = re.compile(
    r"^(lookup|entity|timeline|pinned)\b\s*(.*)$",
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
    hits: list["MemoryEntry"] = []
    for e in entries:
        if e.state == MemoryState.FORGOTTEN:
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
    return [_entry_to_dict(e) for e in entries]


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
    return {"tool": "memory", "error": f"unknown kind: {kind}"}


# ---------------------------------------------------------------------------
# Native tool-calling schema generators (for users not using the tag)
# ---------------------------------------------------------------------------

_MEMORY_DESCRIPTION = (
    "Look up information from Sherlock's long-term memory store. Use this "
    "when you need to recall a specific fact the user has shared before "
    "(allergies, names, dates, preferences) that may not be in the recent "
    "K-turn window. Cheaper and more precise than re-asking the user."
)


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
                            "enum": ["lookup", "entity", "timeline", "pinned"],
                            "description": "lookup = semantic+entity search; entity = deterministic entity match; timeline = last N raw turns; pinned = all pinned facts",
                        },
                        "args": {
                            "type": "string",
                            "description": "Query / entity / count, depending on kind. Empty for 'pinned'.",
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
                        "enum": ["lookup", "entity", "timeline", "pinned"],
                    },
                    "args": {
                        "type": "string",
                        "description": "Query / entity / count, depending on kind.",
                    },
                },
                "required": ["kind"],
            },
        }
    ]
