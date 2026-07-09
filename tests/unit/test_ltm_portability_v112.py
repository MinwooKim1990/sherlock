"""v1.12 Stage A4 — long-term memory PORTABILITY (export / import).

Pure-store coverage of ``sherlock.memory.portability``: JSON / Markdown / SQL
export, the two importers (routed through ``store.add`` so redaction + dedup
re-apply), taxonomy validation, malformed-input handling, and byte-exact CJK
round-trips across every format.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import (
    LTM_CONVERSATION_ID,
    MemorySource,
    MemoryState,
    MemoryType,
    ltm_category,
)
from sherlock.memory.portability import (
    export_ltm_json,
    export_ltm_markdown,
    export_ltm_sql,
    import_ltm_json,
    import_ltm_markdown,
    live_ltm_rows,
)
from sherlock.storage import Storage


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


def _make_store(tmp_path: Path, name: str, *, redactor=None) -> MemoryStore:
    storage = Storage(tmp_path / f"{name}.db")
    embed = build_embedding_provider(_FakeEmbedConfig())
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=embed,
        vector_path=tmp_path / f"{name}_vectors",
        redactor=redactor,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return _make_store(tmp_path, "src")


def _seed(store, content, category="identity_health", conf=1.0, quote=None, origin=None):
    return store.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=conf,
        pinned=True,
        tags=f"ltm,{category}",
        evidence=json.dumps([{"quote": quote or content, "turn": 1}], ensure_ascii=False),
        origin_conversation_id=origin,
        dedup=False,
    )


def _contents(store) -> set[str]:
    return {e.content for e in live_ltm_rows(store)}


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip_export_wipe_import(store, tmp_path):
    _seed(store, "User is allergic to peanuts", category="identity_health", conf=1.0)
    _seed(store, "Always answer in metric", category="user_directive", conf=0.9, origin="conv-7")
    _seed(store, "User is learning Rust", category="long_term_project", conf=0.75)

    text = export_ltm_json(store)
    before_ids = {e.id for e in live_ltm_rows(store)}

    store.delete_conversation_memories(LTM_CONVERSATION_ID)
    assert live_ltm_rows(store) == []

    result = import_ltm_json(store, text)
    assert result["imported"] == 3 and result["skipped"] == 0

    rows = {e.content: e for e in live_ltm_rows(store)}
    assert set(rows) == {
        "User is allergic to peanuts",
        "Always answer in metric",
        "User is learning Rust",
    }
    # categories + confidence preserved
    assert ltm_category(rows["Always answer in metric"].tags) == "user_directive"
    assert rows["User is learning Rust"].confidence == pytest.approx(0.75)
    # evidence quote survives
    ev = json.loads(rows["User is allergic to peanuts"].evidence)
    assert ev[0]["quote"] == "User is allergic to peanuts"
    # origin preserved
    assert rows["Always answer in metric"].origin_conversation_id == "conv-7"
    # ids are regenerated, not round-tripped
    after_ids = {e.id for e in live_ltm_rows(store)}
    assert before_ids.isdisjoint(after_ids)


def test_json_export_excludes_superseded(store):
    a = _seed(store, "old fact", category="stable_preference")
    b = _seed(store, "new fact", category="stable_preference")
    store.supersede(a.id, b.id, turn_index=5)
    data = json.loads(export_ltm_json(store))
    contents = {f["content"] for f in data["facts"]}
    assert "new fact" in contents and "old fact" not in contents


# ---------------------------------------------------------------------------
# Markdown round-trip
# ---------------------------------------------------------------------------


def test_markdown_round_trip_into_fresh_store(store, tmp_path):
    _seed(store, "User is allergic to peanuts", category="identity_health", conf=0.95, origin="c-1")
    _seed(store, "Always answer in metric", category="user_directive", conf=1.0)

    md = export_ltm_markdown(store)
    assert md.startswith("# Sherlock long-term memory export")

    fresh = _make_store(tmp_path, "fresh_md")
    result = import_ltm_markdown(fresh, md)
    assert result["imported"] == 2 and result["skipped"] == 0

    rows = {e.content: e for e in live_ltm_rows(fresh)}
    assert set(rows) == {"User is allergic to peanuts", "Always answer in metric"}
    # category recovered from the section header
    assert ltm_category(rows["User is allergic to peanuts"].tags) == "identity_health"
    # confidence + origin recovered from the HTML-comment metadata
    assert rows["User is allergic to peanuts"].confidence == pytest.approx(0.95)
    assert rows["User is allergic to peanuts"].origin_conversation_id == "c-1"


def test_markdown_minimal_handwritten_imports(tmp_path):
    fresh = _make_store(tmp_path, "fresh_min")
    md = (
        "# Sherlock long-term memory export\n\n"
        "## identity_health\n\n"
        "- User is allergic to peanuts\n\n"
        "## user_directive\n\n"
        "- Always answer in metric\n"
    )
    result = import_ltm_markdown(fresh, md)
    assert result["imported"] == 2
    rows = {e.content: e for e in live_ltm_rows(fresh)}
    assert ltm_category(rows["Always answer in metric"].tags) == "user_directive"
    # defaults where no metadata comment exists
    assert rows["Always answer in metric"].confidence == pytest.approx(1.0)
    assert rows["Always answer in metric"].evidence == ""


def test_markdown_empty_input_errors(tmp_path):
    fresh = _make_store(tmp_path, "fresh_empty")
    result = import_ltm_markdown(fresh, "# Sherlock long-term memory export\n\nnothing here\n")
    assert "error" in result and result["imported"] == 0


def test_markdown_injection_quote_no_phantom_fact(tmp_path):
    # F3 (audit): an evidence quote containing a newline + "- " (and a "-->")
    # must NOT forge a phantom bullet on re-import; the quote is recovered
    # whitespace-normalised with the comment terminator neutralised.
    src = _make_store(tmp_path, "inj_src")
    inj = "peanut allergy\n- FAKE PHANTOM FACT --> escape"
    _seed(src, "User has a peanut allergy", category="identity_health", quote=inj)

    md = export_ltm_markdown(src)
    fresh = _make_store(tmp_path, "inj_dst")
    result = import_ltm_markdown(fresh, md)

    assert result["imported"] == 1  # exactly the real fact — no phantom bullet
    assert _contents(fresh) == {"User has a peanut allergy"}
    ev = json.loads(live_ltm_rows(fresh)[0].evidence)
    assert ev[0]["quote"] == "peanut allergy - FAKE PHANTOM FACT -> escape"


def test_markdown_multiline_content_survives(tmp_path):
    # F3 (audit): multiline content must not be silently TRUNCATED to its first
    # line on export → re-import; it survives single-line-normalised (whitespace
    # collapsed) with no other loss.
    src = _make_store(tmp_path, "ml_src")
    multi = "line one\nline two\nline three 한국어"
    _seed(src, multi, category="long_term_project")

    md = export_ltm_markdown(src)
    fresh = _make_store(tmp_path, "ml_dst")
    result = import_ltm_markdown(fresh, md)

    assert result["imported"] == 1
    assert _contents(fresh) == {"line one line two line three 한국어"}


# ---------------------------------------------------------------------------
# SQL export
# ---------------------------------------------------------------------------


def test_sql_export_rows_land_in_scratch_db(store, tmp_path):
    tricky = "It's a test\nsecond line 한국어 데이터"
    _seed(store, tricky, category="stable_preference", conf=0.8)
    _seed(store, "plain fact", category="user_directive")

    sql = export_ltm_sql(store)
    assert "INSERT INTO memory_entry" in sql

    # Real schema via a fresh Storage, then load the SQL with raw sqlite3.
    target = Storage(tmp_path / "sql_target.db")
    target.engine.dispose()  # release the engine's handle before the raw write
    conn = sqlite3.connect(str(tmp_path / "sql_target.db"))
    try:
        conn.executescript(sql)
        conn.commit()
        landed = conn.execute(
            "SELECT content, confidence FROM memory_entry WHERE conversation_id = ?",
            (LTM_CONVERSATION_ID,),
        ).fetchall()
    finally:
        conn.close()

    contents = {row[0] for row in landed}
    assert contents == {tricky, "plain fact"}  # quotes / newline / Korean survived
    conf_by_content = {row[0]: row[1] for row in landed}
    assert conf_by_content[tricky] == pytest.approx(0.8)

    # F1 (audit): the restored DB must be RE-OPENABLE by a real MemoryStore.
    # SQLAlchemy Enum columns store the member NAME ('FACT'/'USER'/'FRESH') and
    # sqlite datetimes are space-separated — an export emitting .value / 'T'-iso
    # produces a db that crashes with LookupError on reopen. Reopen and read the
    # rows back through the ORM to prove state/type/created_at survived.
    src_created = {e.content: e.created_at for e in live_ltm_rows(store)}
    reopened = _make_store(tmp_path, "sql_target")
    restored = {e.content: e for e in live_ltm_rows(reopened)}
    assert set(restored) == {tricky, "plain fact"}
    tr = restored[tricky]
    assert tr.type == MemoryType.FACT  # enum coerced from stored NAME, no LookupError
    assert tr.source == MemorySource.USER
    assert tr.state == MemoryState.FRESH
    assert ltm_category(tr.tags) == "stable_preference"
    # created_at round-trips to the same wall-clock instant (sqlite drops tz).
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    assert tr.created_at.strftime(fmt) == src_created[tricky].strftime(fmt)


# ---------------------------------------------------------------------------
# dedup + redaction on import
# ---------------------------------------------------------------------------


def test_import_twice_dedups(store, tmp_path):
    _seed(store, "fact a", category="user_directive")
    _seed(store, "fact b", category="identity_health")
    text = export_ltm_json(store)

    fresh = _make_store(tmp_path, "dedup")
    first = import_ltm_json(fresh, text)
    assert first["imported"] == 2 and first["skipped"] == 0
    second = import_ltm_json(fresh, text)
    assert second["imported"] == 0 and second["skipped"] == 2
    assert len(live_ltm_rows(fresh)) == 2  # no duplicate rows


def test_import_applies_redaction(tmp_path):
    redactor = lambda s: s.replace("SECRET-KEY", "[REDACTED]")  # noqa: E731
    fresh = _make_store(tmp_path, "redact", redactor=redactor)
    envelope = {
        "format": "sherlock-ltm",
        "version": 1,
        "facts": [
            {"content": "my SECRET-KEY is here", "category": "user_directive", "confidence": 1.0}
        ],
    }
    result = import_ltm_json(fresh, json.dumps(envelope))
    assert result["imported"] == 1
    stored = live_ltm_rows(fresh)[0].content
    assert "SECRET-KEY" not in stored and "[REDACTED]" in stored


# ---------------------------------------------------------------------------
# validation / malformed input
# ---------------------------------------------------------------------------


def test_unknown_category_skipped_with_warning(store, tmp_path):
    fresh = _make_store(tmp_path, "unknown_cat")
    envelope = {
        "format": "sherlock-ltm",
        "version": 1,
        "facts": [
            {"content": "keep me", "category": "user_directive"},
            {"content": "drop me", "category": "bogus_bucket"},
        ],
    }
    result = import_ltm_json(fresh, json.dumps(envelope))
    assert result["imported"] == 1 and result["skipped"] == 1
    assert any("bogus_bucket" in w for w in result["warnings"])
    assert _contents(fresh) == {"keep me"}


def test_malformed_json_errors_no_writes(tmp_path):
    fresh = _make_store(tmp_path, "malformed")
    result = import_ltm_json(fresh, "{not valid json")
    assert "error" in result and result["imported"] == 0
    assert live_ltm_rows(fresh) == []


def test_wrong_envelope_marker_errors(tmp_path):
    fresh = _make_store(tmp_path, "wrong_marker")
    result = import_ltm_json(fresh, json.dumps({"format": "something-else", "facts": []}))
    assert "error" in result and result["imported"] == 0


# ---------------------------------------------------------------------------
# CJK byte-exact across every format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "사용자는 소바 알레르기가 있습니다",
        "ユーザーはメートル法を好む",
        "用户正在学习 Rust 编程",
    ],
)
def test_cjk_byte_exact_all_formats(store, tmp_path, content):
    _seed(store, content, category="identity_health", conf=0.9)

    # JSON
    j = _make_store(tmp_path, f"cjk_j_{abs(hash(content))}")
    import_ltm_json(j, export_ltm_json(store))
    assert _contents(j) == {content}

    # Markdown
    m = _make_store(tmp_path, f"cjk_m_{abs(hash(content))}")
    import_ltm_markdown(m, export_ltm_markdown(store))
    assert _contents(m) == {content}

    # SQL
    target = Storage(tmp_path / f"cjk_sql_{abs(hash(content))}.db")
    target.engine.dispose()
    conn = sqlite3.connect(str(tmp_path / f"cjk_sql_{abs(hash(content))}.db"))
    try:
        conn.executescript(export_ltm_sql(store))
        conn.commit()
        got = conn.execute(
            "SELECT content FROM memory_entry WHERE conversation_id = ?",
            (LTM_CONVERSATION_ID,),
        ).fetchall()
    finally:
        conn.close()
    assert {row[0] for row in got} == {content}
