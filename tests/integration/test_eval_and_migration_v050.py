"""v0.5.0 — LLM-judge plumbing (Phase 6) + DB migration (Phase 7.1)."""

from __future__ import annotations


from sqlalchemy import create_engine, text

# ---- LLM judge -----------------------------------------------------------


def test_llm_judge_assertion_consults_judge():
    from evaluation.ralph_v2 import AssertionContext, _assert_llm_judge

    seen = {"prompt": None}

    def stub_judge(prompt: str) -> str:
        seen["prompt"] = prompt
        return "PASS: the response acknowledges the underlying ask."

    ctx = AssertionContext(
        response="I hear that you're really asking whether it's okay to feel worried.",
        captured={},
        agent=None,
        probe=None,
        judge=stub_judge,
    )
    ok, msg = _assert_llm_judge(ctx, {"criterion": "acknowledges the emotional subtext"})
    assert ok is True
    assert seen["prompt"] is not None and "CRITERION" in seen["prompt"]
    assert "RESPONSE" in seen["prompt"]


def test_llm_judge_fail_verdict():
    from evaluation.ralph_v2 import AssertionContext, _assert_llm_judge

    ctx = AssertionContext(
        response="yes.",
        captured={},
        agent=None,
        probe=None,
        judge=lambda p: "FAIL: flat yes/no, ignores the subtext.",
    )
    ok, msg = _assert_llm_judge(ctx, {"criterion": "must not be a flat yes/no"})
    assert ok is False
    assert "FAIL" in msg


def test_llm_judge_skips_without_judge():
    from evaluation.ralph_v2 import AssertionContext, _assert_llm_judge

    ctx = AssertionContext(response="anything", captured={}, agent=None, probe=None, judge=None)
    ok, msg = _assert_llm_judge(ctx, {"criterion": "x"})
    assert ok is True  # skip = pass (rule assertions still gate)
    assert "SKIP" in msg.upper()


def test_make_judge_with_fake_provider():
    from evaluation.ralph_v2 import make_judge

    judge = make_judge("fake:echo")
    assert callable(judge)
    out = judge("PASS or FAIL: is the sky blue?")
    assert isinstance(out, str)


def test_make_judge_none_for_empty_spec():
    from evaluation.ralph_v2 import make_judge

    assert make_judge(None) is None
    assert make_judge("") is None


# ---- DB migration --------------------------------------------------------


def test_run_migrations_adds_missing_column(tmp_path):
    """Simulate a pre-v0.5 DB whose memory_entry lacks content_hash, then
    confirm run_migrations adds it (and the store can use it).
    """
    import sherlock.memory.entry  # noqa: F401 — register memory models in metadata
    from sherlock.storage.db import run_migrations

    db = tmp_path / "old.sqlite"
    eng = create_engine(f"sqlite:///{db}")
    # Create a memory_entry table WITHOUT content_hash (old schema subset).
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE memory_entry (id TEXT PRIMARY KEY, conversation_id TEXT, "
                "content TEXT, type TEXT, source TEXT)"
            )
        )
    added = run_migrations(eng)
    assert any("memory_entry.content_hash" == a for a in added), f"added={added}"
    # Column now exists.
    with eng.begin() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(memory_entry)"))]
    assert "content_hash" in cols


def test_store_works_on_migrated_db(tmp_path):
    """End-to-end: a store built on a stale DB should self-migrate and work."""
    from sherlock.memory import MemoryStore, build_embedding_provider
    from sherlock.memory.entry import MemoryType, MemorySource

    class _Cfg:
        provider = "fake"
        model = "fake-embedding"
        api_key_env = None

    db = tmp_path / "stale.sqlite"
    eng = create_engine(f"sqlite:///{db}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE memory_entry (id TEXT PRIMARY KEY, conversation_id TEXT, "
                "content TEXT, type TEXT, source TEXT)"
            )
        )
    # Building the store should migrate + create memory_entity, then add works.
    store = MemoryStore(
        engine=eng,
        embedding_provider=build_embedding_provider(_Cfg()),
        vector_path=tmp_path / "vec",
    )
    e = store.add(
        conversation_id="c", content="hello", type=MemoryType.FACT, source=MemorySource.USER
    )
    assert e.content_hash  # column usable
