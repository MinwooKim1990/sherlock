"""v1.0 P2 — one-shot JSON retry with error feedback for LLM-2/LLM-3."""

from __future__ import annotations

import json

from sherlock.jsonish import RETRY_STATS, chat_json_with_retry, safe_parse_json
from sherlock.providers.base import ChatMessage, ChatResponse


class ScriptedProvider:
    """Returns queued replies; records every prompt it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages, **kw):
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else "{}"
        return ChatResponse(text=text, model="scripted")


def _msgs():
    return [ChatMessage(role="user", content="produce json")]


def test_garbage_then_valid_is_rescued():
    p = ScriptedProvider(["sorry, I cannot do JSON", '{"facts": [1, 2]}'])
    parsed, resp = chat_json_with_retry(p, _msgs(), want=dict)
    assert parsed == {"facts": [1, 2]}
    assert len(p.calls) == 2
    # the retry prompt carries the parse error + the strict instruction
    retry_user = p.calls[1][-1]
    assert retry_user.role == "user"
    assert "not parseable" in retry_user.content
    assert "ONLY the JSON" in retry_user.content
    # ...and the failed attempt rides along as assistant context
    assert any(m.role == "assistant" and "sorry" in m.content for m in p.calls[1])


def test_garbage_twice_returns_none_after_exactly_two_calls():
    p = ScriptedProvider(["nope", "still nope"])
    parsed, resp = chat_json_with_retry(p, _msgs(), want=dict)
    assert parsed is None
    assert len(p.calls) == 2
    assert resp.text == "still nope"  # callers' fallbacks read the LAST attempt


def test_valid_first_try_makes_one_call():
    p = ScriptedProvider(['{"ok": true}'])
    parsed, _ = chat_json_with_retry(p, _msgs(), want=dict)
    assert parsed == {"ok": True}
    assert len(p.calls) == 1


def test_want_dict_rejects_bare_list_and_retries():
    p = ScriptedProvider(['["a", "b"]', '{"a": 1}'])
    parsed, _ = chat_json_with_retry(p, _msgs(), want=dict)
    assert parsed == {"a": 1}
    assert len(p.calls) == 2


def test_on_usage_fires_per_attempt():
    seen = []
    p = ScriptedProvider(["junk", '{"a": 1}'])
    chat_json_with_retry(p, _msgs(), want=dict, on_usage=lambda r: seen.append(r.text))
    assert seen == ["junk", '{"a": 1}']


def test_summarizer_rescued_by_retry(tmp_path):
    """LLM-2 returns garbage once → retry parses → facts persist (the call
    is no longer wasted)."""
    from sherlock import Sherlock
    from sherlock.memory.entry import MemoryType

    state = {"n": 0}

    def llm2(messages):
        last = messages[-1].get("content", "")
        if "TRANSCRIPT" in last or "not parseable" in last:
            state["n"] += 1
            if state["n"] == 1:
                return "I summarized it like this: the user likes tea!"
            return json.dumps(
                {
                    "summary": "user likes tea",
                    "facts": [
                        {
                            "content": "user likes tea",
                            "type": "fact",
                            "source": "user",
                            "confidence": 0.9,
                        }
                    ],
                    "topic_label": "tea",
                }
            )
        return "{}"

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.\n<<sherlock-companions: compact>>",
        summary_chat=llm2,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=None,
        inference_search_engine=None,
    )
    agent.chat("i like tea")
    assert state["n"] == 2, "expected garbage + one retry"
    summaries = [
        m
        for m in agent.memory.list(conversation_id=agent.conversation_id)
        if m.type == MemoryType.SUMMARY
    ]
    assert any("tea" in m.content for m in summaries), "retried JSON must persist"


def test_retry_stats_accumulate():
    before = dict(RETRY_STATS)
    p = ScriptedProvider(["junk", '{"a": 1}'])
    chat_json_with_retry(p, _msgs(), want=dict)
    assert RETRY_STATS["retries"] == before["retries"] + 1
    assert RETRY_STATS["rescued"] == before["rescued"] + 1


def test_safe_parse_json_reexports_match():
    from sherlock.inference.engine import _safe_parse_json

    assert _safe_parse_json is safe_parse_json
