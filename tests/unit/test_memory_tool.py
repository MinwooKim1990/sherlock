"""Memory-tool unit tests (v0.4.0).

Exercises ``parse_memory_payload``, ``dispatch_memory``, and the four
handler kinds (lookup / entity / timeline / pinned) against a canned
in-memory MemoryStore.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sherlock.memory import MemoryStore, build_embedding_provider
from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.rag.hybrid import HybridSearch
from sherlock.storage import Storage
from sherlock.tools import (
    dispatch_memory,
    memory_entity,
    memory_lookup,
    memory_pinned,
    memory_timeline,
    parse_memory_payload,
)

# ---------- payload parser ------------------------------------------------


def test_parse_payload_lookup_quoted():
    kind, args = parse_memory_payload('lookup "Yujin 알레르기"')
    assert kind == "lookup"
    assert '"Yujin 알레르기"' in args


def test_parse_payload_entity_bare():
    kind, args = parse_memory_payload("entity Yujin")
    assert kind == "entity"
    assert "Yujin" in args


def test_parse_payload_timeline():
    kind, args = parse_memory_payload("timeline last 10")
    assert kind == "timeline"
    assert "10" in args


def test_parse_payload_pinned_no_args():
    kind, args = parse_memory_payload("pinned")
    assert kind == "pinned"
    assert args == ""


def test_parse_payload_malformed():
    kind, args = parse_memory_payload("nonsense")
    assert kind == ""


# ---------- fixtures ------------------------------------------------------


@pytest.fixture
def stores(tmp_path: Path):
    storage = Storage(tmp_path / "test.db")
    embed = build_embedding_provider(_FakeEmbedConfig())
    store = MemoryStore(
        engine=storage.engine,
        embedding_provider=embed,
        vector_path=tmp_path / "vectors",
    )
    hybrid = HybridSearch(store=store)
    conv = storage.create_conversation(project="test")
    # Seed entries:
    store.add(
        conversation_id=conv.id,
        content="Yujin은 5살이고 땅콩 알레르기 있음",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        pinned=True,
        tags="yujin,allergy",
        semantic_triple=("Yujin", "has_allergy", "peanut"),
    )
    store.add(
        conversation_id=conv.id,
        content="사용자는 Nimbus 대시보드 작업 중",
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=1.0,
        pinned=False,
        tags="nimbus,work",
    )
    store.add(
        conversation_id=conv.id,
        content="이름은 Minwoo",
        type=MemoryType.FACT,
        source=MemorySource.SYSTEM,
        confidence=1.0,
        pinned=True,
        tags="persona",
        semantic_triple=("user", "named", "Minwoo"),
    )
    # Some raw messages for timeline:
    for i in range(5):
        storage.add_message(conv.id, role="user", content=f"user turn {i}")
        storage.add_message(conv.id, role="assistant", content=f"reply {i}")
    return storage, store, hybrid, conv.id


class _FakeEmbedConfig:
    provider = "fake"
    model = "fake-embedding"
    api_key_env = None


# ---------- handler tests -------------------------------------------------


def test_memory_pinned_lists_pinned_only(stores):
    _, store, _, conv_id = stores
    out = memory_pinned(store=store, conversation_id=conv_id)
    contents = [r["content"] for r in out]
    assert any("Yujin" in c for c in contents)
    assert any("Minwoo" in c for c in contents)
    # Nimbus is unpinned — should NOT appear
    assert not any("Nimbus" in c for c in contents)


def test_memory_entity_deterministic_match(stores):
    _, store, _, conv_id = stores
    out = memory_entity("Yujin", store=store, conversation_id=conv_id)
    assert any("Yujin" in r["content"] for r in out)
    # Should NOT include the Minwoo fact (different entity)
    assert not any("Minwoo" in r["content"] for r in out)


def test_memory_entity_indexed_lookup_correctness(stores):
    """Regression (v0.5.1 review): memory_entity now uses the persistent entity
    index (find_by_entities) instead of scanning all rows. Correctness must be
    preserved even with many distractor rows: only entity matches come back.
    """
    _, store, _, conv_id = stores
    for i in range(25):
        store.add(
            conversation_id=conv_id,
            content=f"날씨 잡담 {i}",
            type=MemoryType.FACT,
            source=MemorySource.USER,
        )
    out = memory_entity("Yujin", store=store, conversation_id=conv_id)
    contents = [r["content"] for r in out]
    assert any("Yujin" in c for c in contents), "entity hit must be returned"
    assert not any("잡담" in c for c in contents), "distractors must be excluded"
    assert not any("Minwoo" in c for c in contents), "other-entity rows excluded"


def test_memory_timeline_returns_last_n(stores):
    storage, _, _, conv_id = stores
    out = memory_timeline(4, storage=storage, conversation_id=conv_id)
    assert len(out) == 4
    # Should be most-recent 4 (mix of user/assistant)
    assert out[-1]["content"] == "reply 4"


def test_memory_lookup_entity_match_dominates(stores):
    _, store, hybrid, conv_id = stores
    out = memory_lookup("Yujin 알레르기", store=store, hybrid=hybrid, conversation_id=conv_id)
    # Yujin fact should rank first (entity boost)
    assert out and "Yujin" in out[0]["content"]


# ---------- unified dispatcher --------------------------------------------


def test_dispatch_memory_lookup(stores):
    storage, store, hybrid, conv_id = stores
    res = dispatch_memory(
        'lookup "Yujin 알레르기"',
        store=store,
        hybrid=hybrid,
        storage=storage,
        conversation_id=conv_id,
    )
    assert res["tool"] == "memory"
    assert res["kind"] == "lookup"
    assert res["results"]


def test_dispatch_memory_entity(stores):
    storage, store, hybrid, conv_id = stores
    res = dispatch_memory(
        "entity Yujin",
        store=store,
        hybrid=hybrid,
        storage=storage,
        conversation_id=conv_id,
    )
    assert res["kind"] == "entity"
    assert any("Yujin" in r["content"] for r in res["results"])


def test_dispatch_memory_timeline(stores):
    storage, store, hybrid, conv_id = stores
    res = dispatch_memory(
        "timeline last 3",
        store=store,
        hybrid=hybrid,
        storage=storage,
        conversation_id=conv_id,
    )
    assert res["kind"] == "timeline"
    assert len(res["results"]) == 3


def test_dispatch_memory_pinned(stores):
    storage, store, hybrid, conv_id = stores
    res = dispatch_memory(
        "pinned",
        store=store,
        hybrid=hybrid,
        storage=storage,
        conversation_id=conv_id,
    )
    assert res["kind"] == "pinned"
    assert len(res["results"]) >= 2  # Yujin + Minwoo


def test_dispatch_memory_unknown_kind_error():
    res = dispatch_memory("bogus arguments here")
    assert "error" in res


def test_dispatch_memory_no_store_returns_error():
    res = dispatch_memory("lookup foo")
    assert "error" in res


# ---------- NICE-1: native-tool schema gates the LTM verbs on long_term -------

_LTM_VERBS = {"profile", "save", "update", "forget", "forget-confirm", "wipe", "wipe-confirm"}
_READ_VERBS = {"lookup", "entity", "timeline", "pinned"}


def _openai_kinds(long_term):
    from sherlock.tools import make_openai_memory_tool

    tool = make_openai_memory_tool(long_term=long_term)
    return set(tool[0]["function"]["parameters"]["properties"]["kind"]["enum"])


def _anthropic_kinds(long_term):
    from sherlock.tools import make_anthropic_memory_tool

    tool = make_anthropic_memory_tool(long_term=long_term)
    return set(tool[0]["input_schema"]["properties"]["kind"]["enum"])


def test_native_tool_schema_full_surface_when_long_term_on():
    # default AND explicit-True expose read + the seven manage verbs
    for kinds in (_openai_kinds(True), _anthropic_kinds(True)):
        assert kinds == _READ_VERBS | _LTM_VERBS
    from sherlock.tools import make_anthropic_memory_tool, make_openai_memory_tool

    assert (
        set(make_openai_memory_tool()[0]["function"]["parameters"]["properties"]["kind"]["enum"])
        == _READ_VERBS | _LTM_VERBS
    )
    assert (
        set(make_anthropic_memory_tool()[0]["input_schema"]["properties"]["kind"]["enum"])
        == _READ_VERBS | _LTM_VERBS
    )


def test_native_tool_schema_hides_ltm_verbs_when_off():
    # LTM off → only the read verbs, byte-identical to the pre-LTM surface
    for kinds in (_openai_kinds(False), _anthropic_kinds(False)):
        assert kinds == _READ_VERBS
        assert not (kinds & _LTM_VERBS)


def test_native_tool_off_description_has_no_manage_language():
    from sherlock.tools import make_openai_memory_tool

    desc = make_openai_memory_tool(long_term=False)[0]["function"]["description"]
    assert "MANAGE" not in desc and "long-term" not in desc.lower()
