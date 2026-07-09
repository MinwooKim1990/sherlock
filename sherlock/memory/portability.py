"""v1.12 Stage A4 — long-term memory PORTABILITY (export / import).

Pure functions over a :class:`~sherlock.memory.store.MemoryStore`: the store is
passed IN, so this module never imports the agent. Three interchangeable
serialisations of the cross-conversation long-term memory (the reserved
``LTM_CONVERSATION_ID`` sentinel scope) — human-readable Markdown, a
full-fidelity JSON envelope, and executable ``INSERT`` SQL — plus the two
importers that read the first two back.

Design decisions (kept stable across formats):

  * Only LIVE facts are exported — superseded (``superseded_by`` set) and
    ``FORGOTTEN`` tombstones are excluded from every format. Exports are of
    "what is currently remembered", matching ``memory profile``.
  * IMPORT NEVER raw-INSERTs. Every fact is routed through ``store.add`` so the
    redaction choke point AND the 3-scan dedup re-apply. ``id`` /
    ``content_hash`` are therefore regenerated on import (never round-tripped)
    and JSON/Markdown deliberately omit or ignore them.
  * Category is validated against the fixed taxonomy. An unknown category is
    SKIPPED with a per-item warning (never silently remapped) so a foreign or
    corrupt file can't smuggle facts under a bogus bucket.
  * Malformed top-level input → a clear ``{"error": ...}`` dict with NO writes.
    A malformed individual item is skipped (with a warning) while the valid
    items already processed stay written — importers are best-effort per item.
"""

from __future__ import annotations

import json as _json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherlock.memory.entry import MemoryEntry
    from sherlock.memory.store import MemoryStore

# The durable-fact taxonomy (mirrors config LongTermMemoryConfig docs + the
# summarizer's code gate). Markdown H2 sections are emitted in THIS order.
_TAXONOMY_ORDER = (
    "user_directive",
    "identity_health",
    "stable_preference",
    "relationship",
    "long_term_project",
)
_TAXONOMY = frozenset(_TAXONOMY_ORDER)

_MD_HEADER = "# Sherlock long-term memory export"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def live_ltm_rows(store: "MemoryStore") -> list["MemoryEntry"]:
    """Current (non-superseded, non-forgotten) long-term sentinel rows, oldest
    first. This is the exact set every exporter serialises."""
    from sherlock.memory.entry import LTM_CONVERSATION_ID, MemoryState

    rows = [
        e
        for e in store.list(conversation_id=LTM_CONVERSATION_ID)
        if not getattr(e, "superseded_by", None) and e.state != MemoryState.FORGOTTEN
    ]
    rows.sort(key=lambda e: e.created_at)
    return rows


def _iso(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _first_quote(evidence: str) -> str:
    """Best-effort: the first ``quote`` from a row's JSON evidence, or ``""``."""
    try:
        parsed = _json.loads(evidence) if evidence else []
    except Exception:
        return ""
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return str(parsed[0].get("quote", "") or "")
    return ""


# ---------------------------------------------------------------------------
# export: markdown
# ---------------------------------------------------------------------------


def _md_meta_comment(entry: "MemoryEntry") -> str:
    # F3 (audit): normalise whitespace so a quote containing a newline + "- "
    # cannot forge a PHANTOM bullet on re-import, and neutralise "-->" so the
    # quote can't end the HTML comment early (swallowing/exposing later text).
    quote = " ".join(_first_quote(entry.evidence).split()).replace('"', "'").replace("-->", "->")
    origin = entry.origin_conversation_id or "-"
    return (
        f"  <!-- id: {entry.id}, confidence: {entry.confidence}, "
        f'created: {_iso(entry.created_at)}, origin: {origin}, quote: "{quote}" -->'
    )


def export_ltm_markdown(store: "MemoryStore") -> str:
    """Human-readable + re-importable Markdown. Each fact is a bullet with an
    indented HTML-comment metadata line (renders clean in MD viewers but
    survives round-trip through :func:`import_ltm_markdown`)."""
    from sherlock.memory.entry import ltm_category

    rows = live_ltm_rows(store)
    lines = [
        _MD_HEADER,
        "",
        f"<!-- exported_at: {datetime.now(timezone.utc).isoformat()}, count: {len(rows)} -->",
        "",
    ]
    buckets: dict[str, list["MemoryEntry"]] = {}
    for e in rows:
        buckets.setdefault(ltm_category(e.tags), []).append(e)
    # taxonomy order first, then any unexpected categories (defensive) so no
    # live row is ever silently dropped from the export.
    ordered = list(_TAXONOMY_ORDER) + [c for c in buckets if c not in _TAXONOMY]
    for cat in ordered:
        entries = buckets.get(cat)
        if not entries:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        for e in entries:
            # F3 (audit): collapse newlines to spaces so multiline content is not
            # silently TRUNCATED to its first line on re-import (a bullet is a
            # single line). content_hash is whitespace-normalised, so dedup
            # integrity holds despite the collapse.
            lines.append("- " + " ".join(e.content.splitlines()))
            lines.append(_md_meta_comment(e))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# export: json
# ---------------------------------------------------------------------------


def export_ltm_json(store: "MemoryStore") -> str:
    """Versioned envelope with full-fidelity facts (ids/content_hash omitted —
    regenerated on import)."""
    from sherlock.memory.entry import ltm_category

    facts = []
    for e in live_ltm_rows(store):
        try:
            evidence = _json.loads(e.evidence) if e.evidence else []
        except Exception:
            evidence = e.evidence  # keep the raw string if it isn't valid JSON
        facts.append(
            {
                "content": e.content,
                "category": ltm_category(e.tags),
                "confidence": e.confidence,
                "created_at": _iso(e.created_at),
                "origin_conversation_id": e.origin_conversation_id,
                "evidence": evidence,
                "tags": e.tags,
            }
        )
    envelope = {
        "format": "sherlock-ltm",
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "facts": facts,
    }
    return _json.dumps(envelope, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# export: sql
# ---------------------------------------------------------------------------


def _sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):  # before int — bool is an int subclass
        return "1" if value else "0"
    if isinstance(value, Enum):
        value = value.name  # SQLAlchemy Enum columns store the member NAME
    elif isinstance(value, datetime):
        value = value.strftime("%Y-%m-%d %H:%M:%S.%f")  # sqlite dialect storage format
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def export_ltm_sql(store: "MemoryStore") -> str:
    """Executable ``INSERT`` statements for the live sentinel rows, generated
    from the ORM column values (proper escaping, no shell/sqlite3 CLI). The
    table name is read from the model so it can never drift from the schema."""
    from sherlock.memory.entry import MemoryEntry

    table = MemoryEntry.__tablename__
    cols = [c.name for c in MemoryEntry.__table__.columns]
    collist = ", ".join(cols)
    rows = live_ltm_rows(store)
    out = [
        f"-- {_MD_HEADER.lstrip('# ')}",
        f"-- exported_at: {datetime.now(timezone.utc).isoformat()}, count: {len(rows)}",
    ]
    for e in rows:
        values = ", ".join(_sql_literal(getattr(e, c)) for c in cols)
        out.append(f"INSERT INTO {table} ({collist}) VALUES ({values});")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# import: shared ingest
# ---------------------------------------------------------------------------


def _ingest(store: "MemoryStore", items: list[dict]) -> dict:
    """Route normalised fact dicts through ``store.add`` (redaction + dedup).

    Returns ``{imported, skipped, warnings}``. A fact whose returned row id was
    already present (a dedup merge, cross-import or intra-file) counts as
    ``skipped``; an unknown category or empty content is skipped with a warning.
    """
    from sherlock.memory.entry import (
        LTM_CONVERSATION_ID,
        MemorySource,
        MemoryType,
    )

    existing = {e.id for e in store.list(conversation_id=LTM_CONVERSATION_ID)}
    imported = 0
    skipped = 0
    warnings: list[str] = []
    for it in items:
        content = (it.get("content") or "").strip()
        if not content:
            skipped += 1
            warnings.append("skipped an entry with empty content")
            continue
        category = (it.get("category") or "").strip()
        if category not in _TAXONOMY:
            skipped += 1
            warnings.append(f"skipped unknown category {category!r} for: {content[:40]!r}")
            continue
        try:
            confidence = float(it.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        evidence = it.get("evidence") or ""
        if not isinstance(evidence, str):
            evidence = _json.dumps(evidence, ensure_ascii=False)
        row = store.add(
            conversation_id=LTM_CONVERSATION_ID,
            content=content,
            type=MemoryType.FACT,
            source=MemorySource.USER,
            confidence=confidence,
            pinned=True,
            tags=f"ltm,{category}",
            evidence=evidence,
            origin_conversation_id=it.get("origin_conversation_id"),
            dedup=True,
        )
        if row.id in existing:
            skipped += 1
        else:
            imported += 1
            existing.add(row.id)
    return {"imported": imported, "skipped": skipped, "warnings": warnings}


# ---------------------------------------------------------------------------
# import: json
# ---------------------------------------------------------------------------


def import_ltm_json(store: "MemoryStore", text: str) -> dict:
    """Parse a :func:`export_ltm_json` envelope and re-add every fact.

    Top-level malformed input → ``{"error": ...}`` with no writes.

    F7 (audit — accepted, no code change): input size is UNBOUNDED here by
    design — this is a local, user-invoked restore of the user's own export, not
    a network endpoint, so there is no DoS surface to cap. The ``version`` field
    is best-effort/advisory: we key only on the ``format`` marker, so a future
    v2 envelope with extra fields is tolerated (unknown keys ignored) rather than
    rejected on a version bump.
    """
    try:
        data = _json.loads(text)
    except Exception as exc:
        return {"error": f"invalid JSON: {exc}", "imported": 0, "skipped": 0, "warnings": []}
    if not isinstance(data, dict) or data.get("format") != "sherlock-ltm":
        return {
            "error": "not a sherlock-ltm export (missing 'format' marker)",
            "imported": 0,
            "skipped": 0,
            "warnings": [],
        }
    facts = data.get("facts")
    if not isinstance(facts, list):
        return {
            "error": "malformed export: 'facts' must be a list",
            "imported": 0,
            "skipped": 0,
            "warnings": [],
        }
    items = [f for f in facts if isinstance(f, dict)]
    result = _ingest(store, items)
    if len(items) != len(facts):
        result["warnings"].append(f"ignored {len(facts) - len(items)} non-object fact entr(ies)")
    return result


# ---------------------------------------------------------------------------
# import: markdown
# ---------------------------------------------------------------------------

_H2_RE = re.compile(r"^\s*##\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*-\s+(.*)$")
_META_RE = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)


def _parse_meta(inner: str) -> tuple[float, str | None, str | None]:
    confidence = 1.0
    origin: str | None = None
    quote: str | None = None
    cm = re.search(r"confidence:\s*([0-9.]+)", inner)
    if cm:
        try:
            confidence = float(cm.group(1))
        except ValueError:
            confidence = 1.0
    om = re.search(r"origin:\s*([^,]+)", inner)
    if om:
        token = om.group(1).strip()
        origin = None if token in {"-", "", "None"} else token
    qm = re.search(r'quote:\s*"(.*)"\s*$', inner)
    if qm:
        quote = qm.group(1)
    return confidence, origin, quote


def import_ltm_markdown(store: "MemoryStore", text: str) -> dict:
    """Parse Markdown (from :func:`export_ltm_markdown` OR hand-written) and
    re-add every bullet. Category comes from the enclosing ``## <category>``
    header; per-bullet metadata (confidence/origin/quote) is recovered from the
    indented HTML comment when present, else defaults apply."""
    lines = text.splitlines()
    items: list[dict] = []
    current_cat: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        h = _H2_RE.match(line)
        if h:
            current_cat = h.group(1).strip()
            i += 1
            continue
        b = _BULLET_RE.match(line)
        if b and current_cat is not None:
            content = b.group(1).rstrip()
            confidence, origin, quote = 1.0, None, None
            # An indented HTML-comment metadata line may immediately follow.
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("<!--"):
                mm = _META_RE.search(lines[i + 1])
                if mm:
                    confidence, origin, quote = _parse_meta(mm.group(1))
                    i += 1  # consume the comment line
            evidence = _json.dumps([{"quote": quote}], ensure_ascii=False) if quote else ""
            items.append(
                {
                    "content": content,
                    "category": current_cat,
                    "confidence": confidence,
                    "origin_conversation_id": origin,
                    "evidence": evidence,
                }
            )
        i += 1
    if not items:
        return {
            "error": "no importable facts found (expected '## <category>' sections with bullets)",
            "imported": 0,
            "skipped": 0,
            "warnings": [],
        }
    return _ingest(store, items)


# ---------------------------------------------------------------------------
# backup helper (used by wipe)
# ---------------------------------------------------------------------------


def backup_ltm_markdown(store: "MemoryStore", directory: str | Path) -> str | None:
    """Write a Markdown export to ``<directory>/ltm_backup_<UTCtimestamp>.md``
    and return the path. Used by the wipe backup hook.

    F5 (audit): returns ``None`` WITHOUT writing when there are no live rows to
    back up — a double-wipe (or wiping already-empty memory) must not litter the
    storage dir with empty backup files. A ``None`` backup_path is fine to the
    wipe callers (0 rows → nothing to back up). F2's fail-closed abort applies
    only to an actual WRITE failure (this raising), never to an empty scope."""
    if not live_ltm_rows(store):
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"ltm_backup_{stamp}.md"
    path.write_text(export_ltm_markdown(store), encoding="utf-8")
    return str(path)
