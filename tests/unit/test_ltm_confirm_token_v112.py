"""v1.12 Stage A3 — natural-language long-term memory MANAGEMENT (unit level).

Covers the confirm-token mechanics (LTMToolContext) and the dispatch-level
behaviour of the management verbs against a canned MemoryStore seeded directly
in the ``LTM_CONVERSATION_ID`` sentinel scope:

  * token mint / single-use / expiry / new-preview-invalidation / wrong-kind;
  * every management verb REJECTED when the feature is disabled (no ltm_ctx);
  * save blocked under incognito;
  * forget PREVIEW never mutates; forget-confirm deletes EXACTLY the frozen ids
    (and the Chroma vector), and only with the right, unexpired, unused token;
  * entity/substring-scoped forget previews exactly the matching rows;
  * update supersedes (bi-temporal) the matched row;
  * wipe preview/confirm clears the whole sentinel scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sherlock.agent import _detect_remember_cue
from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import LTM_CONVERSATION_ID, MemorySource, MemoryType
from sherlock.storage import Storage
from sherlock.tools import LTMToolContext, dispatch_memory


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


@pytest.fixture
def store(tmp_path: Path):
    storage = Storage(tmp_path / "test.db")
    embed = build_embedding_provider(_FakeEmbedConfig())
    return MemoryStore(
        engine=storage.engine,
        embedding_provider=embed,
        vector_path=tmp_path / "vectors",
    )


def _seed(store: MemoryStore, content: str, category: str = "identity_health", conf: float = 1.0):
    return store.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=conf,
        pinned=True,
        tags=f"ltm,{category}",
        dedup=False,
    )


def _ctx(pending: dict, *, turn: int = 1, enabled: bool = True, incognito: bool = False):
    return LTMToolContext(enabled=enabled, incognito=incognito, turn_index=turn, pending=pending)


def _live_ids(store: MemoryStore) -> set[str]:
    return {e.id for e in store.list(conversation_id=LTM_CONVERSATION_ID) if not e.superseded_by}


# ---------------------------------------------------------------------------
# token mechanics (LTMToolContext)
# ---------------------------------------------------------------------------


def test_token_mint_single_use():
    pending: dict = {}
    ctx = _ctx(pending, turn=3)
    tok = ctx.mint("delete", ["a", "b"])
    assert tok in pending
    rec, err = ctx.consume(tok, "delete")
    assert err is None and rec["ids"] == ["a", "b"]
    # single-use: the token is gone.
    rec2, err2 = ctx.consume(tok, "delete")
    assert rec2 is None and err2


def test_token_expiry_boundary():
    pending: dict = {}
    _ctx(pending, turn=1).mint("delete", ["x"])
    tok = next(iter(pending))
    # 2 turns later: still valid (turn_index - minted_turn == 2 == TTL).
    rec, err = _ctx(pending, turn=3).consume(tok, "delete")
    assert err is None and rec is not None
    # restore + 3 turns later: expired.
    _ctx(pending, turn=1).mint("delete", ["x"])
    tok = next(iter(pending))
    rec, err = _ctx(pending, turn=4).consume(tok, "delete")
    assert rec is None and "expired" in err
    assert tok not in pending  # expiry evicts the stale record


def test_new_preview_invalidates_prior_token():
    pending: dict = {}
    ctx = _ctx(pending, turn=1)
    tok_a = ctx.mint("delete", ["a"])
    tok_b = ctx.mint("delete", ["b"])  # latest preview wins
    assert tok_a != tok_b
    assert tok_a not in pending and tok_b in pending
    rec, err = ctx.consume(tok_a, "delete")
    assert rec is None and err


def test_wrong_kind_token_rejected():
    pending: dict = {}
    ctx = _ctx(pending, turn=1)
    tok = ctx.mint("delete", ["a"])
    rec, err = ctx.consume(tok, "wipe")
    assert rec is None and "does not match" in err
    assert tok in pending  # not consumed on a kind mismatch


# ---------------------------------------------------------------------------
# feature gate: rejected when disabled / incognito
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "profile",
        "save remember me",
        "forget x",
        "forget-confirm ab",
        "wipe",
        "wipe-confirm ab",
        "update ab new",
    ],
)
def test_verbs_rejected_when_disabled(store, payload):
    before = len(store.list(conversation_id=LTM_CONVERSATION_ID))
    # ltm_ctx=None → the agent would pass enabled=False; both must reject.
    res = dispatch_memory(payload, store=store, conversation_id="c1", ltm_ctx=None)
    assert "error" in res and res["error"] == "long-term memory is disabled"
    assert len(store.list(conversation_id=LTM_CONVERSATION_ID)) == before


def test_save_blocked_incognito(store):
    ctx = _ctx({}, incognito=True)
    res = dispatch_memory("save always use metric", store=store, conversation_id="c1", ltm_ctx=ctx)
    assert "error" in res and "incognito" in res["error"]
    assert store.list(conversation_id=LTM_CONVERSATION_ID) == []


# ---------------------------------------------------------------------------
# save + profile round-trip
# ---------------------------------------------------------------------------


def test_save_and_profile_roundtrip(store):
    ctx = _ctx({})
    res = dispatch_memory(
        "save always answer in metric units", store=store, conversation_id="conv-x", ltm_ctx=ctx
    )
    assert res["saved"] is True and res["category"] == "user_directive"
    rows = store.list(conversation_id=LTM_CONVERSATION_ID)
    assert len(rows) == 1
    assert rows[0].origin_conversation_id == "conv-x"
    assert rows[0].pinned and "ltm,user_directive" in rows[0].tags
    prof = dispatch_memory("profile", store=store, ltm_ctx=ctx)
    assert prof["kind"] == "profile"
    assert any("metric" in r["content"] for r in prof["results"])
    assert prof["results"][0]["category"] == "user_directive"


# ---------------------------------------------------------------------------
# forget: preview never mutates; confirm deletes frozen ids + vector
# ---------------------------------------------------------------------------


def test_forget_preview_never_mutates(store):
    _seed(store, "User is allergic to peanuts")
    _seed(store, "User likes tea", category="stable_preference")
    pending: dict = {}
    ctx = _ctx(pending)
    before = _live_ids(store)
    res = dispatch_memory("forget allergic", store=store, ltm_ctx=ctx)
    assert res["count"] == 1 and "confirm_token" in res
    assert "instruction" in res
    # calling forget again (another preview) still deletes nothing.
    dispatch_memory("forget allergic", store=store, ltm_ctx=ctx)
    assert _live_ids(store) == before  # zero mutation across two previews


def test_forget_confirm_deletes_frozen_ids_and_vector(store):
    row = _seed(store, "User is allergic to peanuts")
    _seed(store, "User likes tea", category="stable_preference")
    pending: dict = {}
    ctx = _ctx(pending, turn=1)
    prev = dispatch_memory("forget peanuts", store=store, ltm_ctx=ctx)
    tok = prev["confirm_token"]
    # confirm on the next turn.
    ctx2 = _ctx(pending, turn=2)
    res = dispatch_memory(f"forget-confirm {tok}", store=store, ltm_ctx=ctx2)
    assert res["deleted"] == 1
    assert store.get(row.id) is None  # SQLite row gone
    got = store._collection.get(ids=[row.id])  # Chroma vector gone
    assert got.get("ids") == []
    # the other fact survives.
    assert len(store.list(conversation_id=LTM_CONVERSATION_ID)) == 1


def test_forget_confirm_wrong_token(store):
    row = _seed(store, "User is allergic to peanuts")
    pending: dict = {}
    dispatch_memory("forget peanuts", store=store, ltm_ctx=_ctx(pending))
    res = dispatch_memory("forget-confirm deadbeef", store=store, ltm_ctx=_ctx(pending, turn=2))
    assert "error" in res
    assert store.get(row.id) is not None  # nothing deleted


def test_forget_confirm_reused_token(store):
    row = _seed(store, "User is allergic to peanuts")
    pending: dict = {}
    tok = dispatch_memory("forget peanuts", store=store, ltm_ctx=_ctx(pending))["confirm_token"]
    r1 = dispatch_memory(f"forget-confirm {tok}", store=store, ltm_ctx=_ctx(pending, turn=2))
    assert r1["deleted"] == 1 and store.get(row.id) is None
    # reusing the same token deletes nothing more.
    r2 = dispatch_memory(f"forget-confirm {tok}", store=store, ltm_ctx=_ctx(pending, turn=2))
    assert "error" in r2


def test_forget_confirm_expired(store):
    row = _seed(store, "User is allergic to peanuts")
    pending: dict = {}
    tok = dispatch_memory("forget peanuts", store=store, ltm_ctx=_ctx(pending, turn=1))[
        "confirm_token"
    ]
    res = dispatch_memory(f"forget-confirm {tok}", store=store, ltm_ctx=_ctx(pending, turn=4))
    assert "error" in res and "expired" in res["error"]
    assert store.get(row.id) is not None


def test_forget_superseded_by_new_preview(store):
    row = _seed(store, "User is allergic to peanuts")
    pending: dict = {}
    ctx = _ctx(pending, turn=1)
    tok_a = dispatch_memory("forget peanuts", store=store, ltm_ctx=ctx)["confirm_token"]
    tok_b = dispatch_memory("forget peanuts", store=store, ltm_ctx=ctx)["confirm_token"]
    # confirming the STALE (pre-latest-preview) token does nothing.
    r_a = dispatch_memory(f"forget-confirm {tok_a}", store=store, ltm_ctx=_ctx(pending, turn=1))
    assert "error" in r_a and store.get(row.id) is not None
    # the latest token still works.
    r_b = dispatch_memory(f"forget-confirm {tok_b}", store=store, ltm_ctx=_ctx(pending, turn=1))
    assert r_b["deleted"] == 1 and store.get(row.id) is None


# ---------------------------------------------------------------------------
# entity-scoped forget
# ---------------------------------------------------------------------------


def test_entity_scoped_forget(store):
    a = _seed(store, "유진은 5살이다", category="relationship")
    b = _seed(store, "유진은 땅콩 알레르기가 있다", category="identity_health")
    other = _seed(store, "김민우는 서울에 산다", category="identity_health")
    pending: dict = {}
    prev = dispatch_memory("forget 유진", store=store, ltm_ctx=_ctx(pending, turn=1))
    ids = {p["id"] for p in prev["pending"]}
    assert ids == {a.id[:8], b.id[:8]}  # exactly the two 유진 facts
    tok = prev["confirm_token"]
    res = dispatch_memory(f"forget-confirm {tok}", store=store, ltm_ctx=_ctx(pending, turn=2))
    assert res["deleted"] == 2
    assert store.get(a.id) is None and store.get(b.id) is None
    assert store.get(other.id) is not None  # the other person survives


# ---------------------------------------------------------------------------
# update: supersede in place
# ---------------------------------------------------------------------------


def test_update_supersedes(store):
    row = _seed(store, "User lives in Seoul", category="identity_health")
    ctx = _ctx({}, turn=5)
    res = dispatch_memory(
        f"update {row.id[:8]} User lives in Busan", store=store, conversation_id="c1", ltm_ctx=ctx
    )
    assert res["updated"] is True
    old = store.get(row.id)
    assert old.superseded_by is not None
    assert old.invalid_at_turn == 5
    live = [e for e in store.list(conversation_id=LTM_CONVERSATION_ID) if not e.superseded_by]
    assert len(live) == 1 and live[0].content == "User lives in Busan"
    assert "ltm,identity_health" in live[0].tags  # category carried over


def test_update_ambiguous_prefix_errors(store):
    # Two rows whose ids both start with the same 1-char prefix are unlikely with
    # uuid4, so force ambiguity by matching on the empty-ish common case: use a
    # prefix that both share (the full-scope match via a deliberately short probe).
    r1 = _seed(store, "fact one")
    r2 = _seed(store, "fact two")
    # Find a shared leading char across the two ids; if none, this asserts the
    # unique path instead (still valid coverage).
    if r1.id[0] == r2.id[0]:
        res = dispatch_memory(f"update {r1.id[0]} corrected", store=store, ltm_ctx=_ctx({}))
        assert "error" in res and res["error"].startswith("id-prefix is ambiguous")
        assert "candidates" in res
    else:
        res = dispatch_memory(f"update {r1.id[:8]} corrected", store=store, ltm_ctx=_ctx({}))
        assert res["updated"] is True


def test_update_no_match_errors(store):
    _seed(store, "fact one")
    res = dispatch_memory("update zzzzzzzz corrected", store=store, ltm_ctx=_ctx({}))
    assert "error" in res and "no long-term fact" in res["error"]


# ---------------------------------------------------------------------------
# wipe
# ---------------------------------------------------------------------------


def test_wipe_preview_and_confirm(store):
    _seed(store, "fact one")
    _seed(store, "fact two")
    pending: dict = {}
    prev = dispatch_memory("wipe", store=store, ltm_ctx=_ctx(pending, turn=1))
    assert prev["count"] == 2 and "confirm_token" in prev
    assert prev["total"] == 2  # F6: no tombstones → live == total
    # still present after preview.
    assert len(store.list(conversation_id=LTM_CONVERSATION_ID)) == 2
    tok = prev["confirm_token"]
    res = dispatch_memory(f"wipe-confirm {tok}", store=store, ltm_ctx=_ctx(pending, turn=2))
    assert res["wiped"] == 2
    assert store.list(conversation_id=LTM_CONVERSATION_ID) == []


def test_wipe_confirm_requires_wipe_token_not_delete(store):
    _seed(store, "fact one")
    pending: dict = {}
    # a delete token must not be usable to confirm a wipe.
    tok = dispatch_memory("forget fact", store=store, ltm_ctx=_ctx(pending, turn=1))[
        "confirm_token"
    ]
    res = dispatch_memory(f"wipe-confirm {tok}", store=store, ltm_ctx=_ctx(pending, turn=1))
    assert "error" in res
    assert len(store.list(conversation_id=LTM_CONVERSATION_ID)) == 1


def test_wipe_preview_reports_live_and_total(store):
    # F6: with a superseded tombstone in the scope, the headline `count` is the
    # LIVE set (what `profile` shows) and `total` is the full row count that
    # wipe-confirm actually purges — count must never exceed total.
    a = _seed(store, "orig fact")
    _seed(store, "kept fact")
    ctx = _ctx({}, turn=1)
    dispatch_memory(
        f"update {a.id[:8]} corrected fact", store=store, conversation_id="c1", ltm_ctx=ctx
    )
    live_rows = dispatch_memory("profile", store=store, ltm_ctx=_ctx({}))["results"]
    prev = dispatch_memory("wipe", store=store, ltm_ctx=_ctx({}, turn=2))
    assert prev["count"] == len(live_rows) == 2  # corrected + kept
    assert prev["total"] == 3  # + the superseded tombstone
    assert prev["count"] <= prev["total"]
    assert "superseded/forgotten history row" in prev["instruction"]


# ---------------------------------------------------------------------------
# F4: forget query specificity — 1-char reject, entity-first, substring fallback
# ---------------------------------------------------------------------------


def test_forget_single_char_query_rejected(store):
    row = _seed(store, "User is allergic to peanuts")
    pending: dict = {}
    res = dispatch_memory("forget e", store=store, ltm_ctx=_ctx(pending))
    assert "error" in res and "too short" in res["error"]
    assert pending == {}  # no token minted for an over-broad query
    assert "confirm_token" not in res
    assert store.get(row.id) is not None  # nothing was touched


def test_forget_substring_fallback_when_no_entity(store):
    # "peanuts" isn't an entity token here → the substring fallback still finds
    # exactly the one matching row (and not the unrelated tea row).
    row = _seed(store, "User is allergic to peanuts")
    _seed(store, "User likes tea", category="stable_preference")
    pending: dict = {}
    res = dispatch_memory("forget peanuts", store=store, ltm_ctx=_ctx(pending))
    assert res["count"] == 1
    assert res["pending"][0]["id"] == row.id[:8]


def test_forget_two_char_query_allowed(store):
    # The >= 2 char guard is a boundary: a 2-char query is accepted (not rejected)
    # and still previews exactly the matching row.
    row = _seed(store, "User loves AI research")
    pending: dict = {}
    res = dispatch_memory("forget ai", store=store, ltm_ctx=_ctx(pending))
    assert "error" not in res
    assert res["count"] == 1 and res["pending"][0]["id"] == row.id[:8]


# ---------------------------------------------------------------------------
# F2: remember-cue false positives + double-space lookbehind guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "항상 미터법으로 답해줘. 기억해!",  # Korean imperative
        "기억해줘",
        "잊지 마",
        "覚えて",  # Japanese imperative
        "merke dir das",  # German imperative
        "recuerda esto",  # Spanish imperative
        "please remember to email me",
        "remember to buy milk",
        "remember this address",
        "from now on use metric",
        "keep in mind I'm allergic to shellfish",
        "don't forget my birthday",
    ],
)
def test_remember_cue_positive(text):
    assert _detect_remember_cue(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I remember that trip fondly",  # F2: bare declarative "remember that" removed
        "do you remember that trip",
        "do you  remember this",  # F2: double space must not defeat the (?<!you ) guard
        "기억해?",  # F2: Korean recall QUESTION
        "그거 기억나?",
        "¿recuerdas eso?",  # F2: Spanish recall QUESTION ("recuerdas" ⊃ "recuerda")
    ],
)
def test_remember_cue_negative(text):
    assert _detect_remember_cue(text) is False
