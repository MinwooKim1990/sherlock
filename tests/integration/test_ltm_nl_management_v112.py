"""v1.12 Stage A3 — natural-language long-term memory MANAGEMENT (end-to-end).

Drives the management verbs through ``Sherlock.chat()`` with a stateful fake
LLM-1 that emits ``<<sherlock-tool: memory ...>>`` tags exactly as a real model
would, and reads the confirm token back out of the rendered tool-result block:

  * profile / save round-trip through chat();
  * save blocked when disabled + when incognito (error, no row);
  * the two-turn forget→confirm flow deletes the row AND its Chroma vector;
    a lone forget preview mutates nothing;
  * management events fire (memory.saved / delete_pending / deleted / wiped);
  * the deterministic "remember this" cue → memory.remember_cue event + a nudge
    line in the assembled prompt + the next compaction promoting the fact as
    user_directive;
  * OFF (default) → verbs rejected, no nudge, prompts byte-identical.

Hermetic: fake embeddings + canned callables, background=False (companions
inline so state is settled the moment chat() returns).
"""

from __future__ import annotations

import json
import re

from sherlock import Sherlock
from sherlock.agent import _LTM_REMEMBER_NUDGE, LTM_TOOL_GUIDANCE
from sherlock.memory.entry import LTM_CONVERSATION_ID, MemorySource, MemoryType

_TOKEN_RE = re.compile(r"CONFIRM TOKEN:\s*([0-9a-f]+)")


def _last_user(messages) -> str:
    for m in reversed(messages):
        role = m["role"] if isinstance(m, dict) else m.role
        if role == "user":
            return m["content"] if isinstance(m, dict) else m.content
    return ""


def _first_system(messages) -> str:
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        if role == "system":
            return m["content"] if isinstance(m, dict) else m.content
    return ""


class ToolMain:
    """Stateful fake LLM-1. On a REAL user turn it emits whatever tag was queued
    via ``next_tag``; on the tool-result finalisation call it records the block
    (and parses out any confirm token) and returns a plain reply."""

    def __init__(self):
        self.next_tag: str | None = None
        self.tool_blocks: list[str] = []
        self.token: str | None = None
        self.system_msgs: list[str] = []

    def __call__(self, messages):
        content = _last_user(messages)
        self.system_msgs.append(_first_system(messages))
        if "SHERLOCK TOOL RESULTS" in content:
            self.tool_blocks.append(content)
            m = _TOKEN_RE.search(content)
            if m:
                self.token = m.group(1)
            return "완료."
        tag = self.next_tag
        self.next_tag = None
        return tag if tag else "네."


def _agent(tmp_path, main, *, long_term, summary_chat=None, companions="off"):
    return Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary_chat,
        system_prompt="You are a helpful assistant.",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        companions_mode=companions,
        long_term=long_term,
    )


def _seed(agent, content, category="identity_health", conf=1.0):
    return agent.memory.add(
        conversation_id=LTM_CONVERSATION_ID,
        content=content,
        type=MemoryType.FACT,
        source=MemorySource.USER,
        confidence=conf,
        pinned=True,
        tags=f"ltm,{category}",
        dedup=False,
    )


def _events_of(events, typ):
    return [e for e in events if e["type"] == typ]


# ---------------------------------------------------------------------------
# profile / save
# ---------------------------------------------------------------------------


def test_profile_via_chat(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=True)
    _seed(agent, "User is allergic to peanuts")
    _seed(agent, "Always answer in metric", category="user_directive")

    main.next_tag = "<<sherlock-tool: memory profile>>"
    agent.chat("뭘 기억하고 있어?")

    assert main.tool_blocks, "profile finalisation never happened"
    block = main.tool_blocks[-1]
    assert "allergic to peanuts" in block
    assert "Always answer in metric" in block


def test_save_via_chat_creates_row_and_event(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)

    main.next_tag = "<<sherlock-tool: memory save 항상 미터법으로 답해줘>>"
    agent.chat("이거 저장해줘")

    rows = agent.memory.list(conversation_id=LTM_CONVERSATION_ID)
    assert len(rows) == 1
    assert "미터법" in rows[0].content
    assert rows[0].origin_conversation_id == agent.conversation_id
    assert _events_of(events, "memory.saved")


def test_save_blocked_when_disabled(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=None)
    main.next_tag = "<<sherlock-tool: memory save remember this>>"
    agent.chat("save it")
    assert agent.memory.list(conversation_id=LTM_CONVERSATION_ID) == []
    assert "disabled" in main.tool_blocks[-1]


def test_save_blocked_when_incognito(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term={"enabled": True, "incognito": True})
    main.next_tag = "<<sherlock-tool: memory save remember this>>"
    agent.chat("save it")
    assert agent.memory.list(conversation_id=LTM_CONVERSATION_ID) == []
    assert "incognito" in main.tool_blocks[-1]


# ---------------------------------------------------------------------------
# forget: two-turn confirm flow + vector deletion
# ---------------------------------------------------------------------------


def test_forget_two_turn_flow_deletes_row_and_vector(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)
    row = _seed(agent, "User is allergic to peanuts")
    _seed(agent, "User likes tea", category="stable_preference")

    # Turn 1: forget PREVIEW — mutates nothing, surfaces a confirm token.
    main.next_tag = "<<sherlock-tool: memory forget peanuts>>"
    agent.chat("내 땅콩 알레르기 정보 지워줘")
    assert agent.memory.get(row.id) is not None  # NOT deleted on preview
    assert main.token is not None  # LLM-1 read the token from the block
    assert _events_of(events, "memory.delete_pending")
    assert not _events_of(events, "memory.deleted")

    # Turn 2: the user confirms → LLM-1 emits forget-confirm with that token.
    main.next_tag = f"<<sherlock-tool: memory forget-confirm {main.token}>>"
    agent.chat("응 지워도 돼")

    assert agent.memory.get(row.id) is None  # SQLite row gone
    got = agent.memory._collection.get(ids=[row.id])
    assert got.get("ids") == []  # Chroma vector gone too
    assert len(agent.memory.list(conversation_id=LTM_CONVERSATION_ID)) == 1  # tea survives
    deleted = _events_of(events, "memory.deleted")
    assert deleted and deleted[-1]["data"]["count"] == 1


def test_forget_without_confirm_never_deletes(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=True)
    row = _seed(agent, "User is allergic to peanuts")

    main.next_tag = "<<sherlock-tool: memory forget peanuts>>"
    agent.chat("지워줘")
    # A second, independent preview turn — still no deletion.
    main.next_tag = "<<sherlock-tool: memory forget peanuts>>"
    agent.chat("아니 다시 봐줘")

    assert agent.memory.get(row.id) is not None


def test_wipe_two_turn_flow(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)
    _seed(agent, "fact one")
    _seed(agent, "fact two")

    main.next_tag = "<<sherlock-tool: memory wipe>>"
    agent.chat("내 장기기억 다 지워")
    assert len(agent.memory.list(conversation_id=LTM_CONVERSATION_ID)) == 2  # preview only
    assert main.token is not None

    main.next_tag = f"<<sherlock-tool: memory wipe-confirm {main.token}>>"
    agent.chat("응 전부 지워")
    assert agent.memory.list(conversation_id=LTM_CONVERSATION_ID) == []
    wiped = _events_of(events, "memory.wiped")
    assert wiped and wiped[-1]["data"]["count"] == 2


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_via_chat_supersedes(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)
    row = _seed(agent, "User lives in Seoul", category="identity_health")

    main.next_tag = f"<<sherlock-tool: memory update {row.id[:8]} User lives in Busan>>"
    agent.chat("고쳐줘")

    old = agent.memory.get(row.id)
    assert old.superseded_by is not None and old.invalid_at_turn is not None
    live = agent.long_term_memory()
    assert len(live) == 1 and live[0]["content"] == "User lives in Busan"
    assert _events_of(events, "memory.updated")


# ---------------------------------------------------------------------------
# remember-this cue
# ---------------------------------------------------------------------------


def _llm2_fact_payload(_messages):
    # No "quote" field → the ungrounded-quote guard is skipped, so the ALWAYS
    # user_directive promotion (forced by the belt-and-braces flag) goes through.
    return json.dumps(
        {
            "summary": "user shared a durable preference",
            "facts": [
                {
                    "content": "User wants answers in metric units",
                    "type": "fact",
                    "source": "user",
                    "confidence": 0.8,
                    "pin_recommended": False,
                    "let_fade": False,
                    "category": "none",
                    "long_term": False,
                }
            ],
            "topic_label": "prefs",
            "topic_changed_from_previous": False,
            "retrieval_keywords": [],
        }
    )


def test_remember_cue_nudges_and_promotes(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(
        tmp_path, main, long_term=True, summary_chat=_llm2_fact_payload, companions="turbo"
    )
    agent.set_event_sink(events.append)

    # Korean imperative "기억해" + the LLM-1 reply also asks for a compaction, so
    # the belt-and-braces promotion runs this same turn.
    main.next_tag = "알겠습니다.\n<<sherlock-companions: compact>>"
    agent.chat("항상 미터법으로 답해줘. 기억해!")

    # (a) the cue event fired.
    assert _events_of(events, "memory.remember_cue")
    # (b) the nudge reached the assembled prompt (slot.assembled.final_user_message).
    slot = _events_of(events, "slot.assembled")[-1]
    assert _LTM_REMEMBER_NUDGE in slot["data"]["final_user_message"]
    # (c) the compaction promoted the fact as user_directive despite the model's
    #     "category": "none".
    ltm = agent.long_term_memory()
    assert ltm and any(r["category"] == "user_directive" for r in ltm)


def test_off_no_cue_no_nudge_byte_identical(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=None)
    agent.set_event_sink(events.append)

    agent.chat("항상 미터법으로 답해줘. 기억해!")

    # No cue event, no nudge, no long-term guidance in the prompt.
    assert not _events_of(events, "memory.remember_cue")
    slot = _events_of(events, "slot.assembled")[-1]
    assert _LTM_REMEMBER_NUDGE not in slot["data"]["final_user_message"]
    sys = main.system_msgs[-1]
    assert LTM_TOOL_GUIDANCE not in sys
    assert not agent._ltm_remember_promote_pending


def test_off_guidance_absent_on_present_when_enabled(tmp_path):
    # Companion to the OFF case: enabling the feature DOES inject the guidance.
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=True)
    agent.chat("hello")
    assert LTM_TOOL_GUIDANCE in main.system_msgs[-1]


# ---------------------------------------------------------------------------
# F5(1): ACHAT round-trip parity — the async path must handle both the two-turn
# token confirm AND the remember-cue promotion (this bites the F1 latch bug).
# ---------------------------------------------------------------------------


async def test_achat_forget_confirm_roundtrip(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(tmp_path, main, long_term=True)
    agent.set_event_sink(events.append)
    row = _seed(agent, "User is allergic to peanuts")
    _seed(agent, "User likes tea", category="stable_preference")

    # Turn 1: forget PREVIEW via achat — mutates nothing, surfaces a token.
    main.next_tag = "<<sherlock-tool: memory forget peanuts>>"
    await agent.achat("내 땅콩 알레르기 정보 지워줘")
    assert agent.memory.get(row.id) is not None
    assert main.token is not None
    assert _events_of(events, "memory.delete_pending")

    # Turn 2: confirm via achat → the exact frozen id is deleted.
    main.next_tag = f"<<sherlock-tool: memory forget-confirm {main.token}>>"
    await agent.achat("응 지워도 돼")
    assert agent.memory.get(row.id) is None
    got = agent.memory._collection.get(ids=[row.id])
    assert got.get("ids") == []  # Chroma vector gone too
    assert len(agent.memory.list(conversation_id=LTM_CONVERSATION_ID)) == 1  # tea survives
    assert _events_of(events, "memory.deleted")


async def test_achat_remember_cue_promotes_and_clears_latch(tmp_path):
    main = ToolMain()
    events: list[dict] = []
    agent = _agent(
        tmp_path, main, long_term=True, summary_chat=_llm2_fact_payload, companions="turbo"
    )
    agent.set_event_sink(events.append)

    main.next_tag = "알겠습니다.\n<<sherlock-companions: compact>>"
    await agent.achat("항상 미터법으로 답해줘. 기억해!")

    # F1: the async compaction CONSUMED the latch this turn (so it can't leak
    # forward and force-promote an unrelated window on a later compaction) ...
    assert agent._ltm_remember_promote_pending is False
    # ... and promoted the covered fact as user_directive despite the model's
    # "category": "none" — the belt-and-braces path fired on the async turn too.
    ltm = agent.long_term_memory()
    assert ltm and any(r["category"] == "user_directive" for r in ltm)
    assert _events_of(events, "memory.remember_cue")


# ---------------------------------------------------------------------------
# F5(2): a confirm token is session-local — it must not survive a session switch.
# ---------------------------------------------------------------------------


def test_confirm_token_cleared_on_new_session(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=True)
    row = _seed(agent, "User is allergic to peanuts")

    # Conversation A: mint a delete token via a forget preview.
    main.next_tag = "<<sherlock-tool: memory forget peanuts>>"
    agent.chat("지워줘")
    tok = main.token
    assert tok is not None and agent._ltm_pending  # a token is pending

    # New session clears the agent-owned pending-token store.
    agent.new_session()
    assert agent._ltm_pending == {}

    # Confirming the stale token in the new session mutates nothing.
    main.next_tag = f"<<sherlock-tool: memory forget-confirm {tok}>>"
    agent.chat("응 지워")
    assert agent.memory.get(row.id) is not None


# ---------------------------------------------------------------------------
# F5(3): a -confirm with NO prior preview is a code-level no-op (never trusts the
# model's token), at the integration level.
# ---------------------------------------------------------------------------


def test_forget_confirm_without_preview_never_deletes(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=True)
    row = _seed(agent, "User is allergic to peanuts")
    main.next_tag = "<<sherlock-tool: memory forget-confirm deadbeef>>"
    agent.chat("그냥 지워")
    assert agent.memory.get(row.id) is not None  # no preview → nothing deleted
    assert "error" in main.tool_blocks[-1].lower()


def test_wipe_confirm_without_preview_never_wipes(tmp_path):
    main = ToolMain()
    agent = _agent(tmp_path, main, long_term=True)
    _seed(agent, "fact one")
    _seed(agent, "fact two")
    main.next_tag = "<<sherlock-tool: memory wipe-confirm deadbeef>>"
    agent.chat("전부 지워")
    assert len(agent.memory.list(conversation_id=LTM_CONVERSATION_ID)) == 2  # not wiped
    assert "error" in main.tool_blocks[-1].lower()
