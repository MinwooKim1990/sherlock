"""v1.0 P8 — prompt-cache plumbing: hint marking, litellm block conversion,
BYO-callable byte identity, cache telemetry."""

from __future__ import annotations

from sherlock.providers.base import ChatMessage, TokenUsage
from sherlock.providers.callable_provider import CallableProvider
from sherlock.providers.litellm_provider import LiteLLMProvider


def test_to_litellm_messages_splits_on_hint():
    msgs = [
        ChatMessage(
            role="system", content="STABLE-PART|volatile-part", cache_stable_prefix_chars=11
        ),
        ChatMessage(role="user", content="hi"),
    ]
    out = LiteLLMProvider._to_litellm_messages(msgs)
    blocks = out[0]["content"]
    assert blocks[0]["text"] == "STABLE-PART"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "|volatile-part"
    assert "cache_control" not in blocks[1]
    assert out[1] == {"role": "user", "content": "hi"}  # plain stays plain


def test_to_litellm_messages_whole_message_hint():
    msgs = [ChatMessage(role="system", content="ALL STABLE", cache_stable_prefix_chars=10)]
    out = LiteLLMProvider._to_litellm_messages(msgs)
    blocks = out[0]["content"]
    assert len(blocks) == 1 and blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_no_hint_is_byte_identical_legacy_shape():
    msgs = [ChatMessage(role="system", content="x"), ChatMessage(role="user", content="y")]
    assert LiteLLMProvider._to_litellm_messages(msgs) == [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
    ]


def test_cache_usage_extraction_anthropic_and_openai_shapes():
    class A:  # Anthropic-style (litellm passthrough)
        cache_read_input_tokens = 120
        cache_creation_input_tokens = 30

    class Details:
        cached_tokens = 99

    class OpenAIUsage:  # OpenAI-style
        cache_read_input_tokens = 0
        prompt_tokens_details = Details()

    assert LiteLLMProvider._cache_usage(A()) == (120, 30)
    assert LiteLLMProvider._cache_usage(OpenAIUsage()) == (99, 0)
    assert LiteLLMProvider._cache_usage(None) == (0, 0)


def test_plain_callable_payload_is_byte_identical():
    """The flagship BYO contract: f(list[dict]) receives EXACTLY the legacy
    payload even when cache hints are set on the messages."""
    seen = {}

    def fn(messages):
        seen["payload"] = messages
        return "ok"

    p = CallableProvider(fn)
    msgs = [
        ChatMessage(role="system", content="stable|volatile", cache_stable_prefix_chars=6),
        ChatMessage(role="user", content="hi"),
    ]
    p.chat(msgs)
    assert seen["payload"] == [
        {"role": "system", "content": "stable|volatile"},
        {"role": "user", "content": "hi"},
    ]


def test_callable_declaring_cache_hints_receives_them():
    seen = {}

    def fn(messages, cache_hints=None):
        seen["hints"] = cache_hints
        return "ok"

    p = CallableProvider(fn)
    msgs = [ChatMessage(role="system", content="stable|volatile", cache_stable_prefix_chars=6)]
    p.chat(msgs)
    assert seen["hints"] == {"stable_prefix_chars": {0: 6}}


def test_callable_without_hints_gets_no_kwarg_when_no_hints_present():
    seen = {"called": False}

    def fn(messages, cache_hints=None):
        seen["called"] = True
        assert cache_hints is None
        return "ok"

    p = CallableProvider(fn)
    p.chat([ChatMessage(role="user", content="hi")])
    assert seen["called"]


def test_token_usage_cache_fields_default_zero():
    u = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert u.cache_read_tokens == 0 and u.cache_creation_tokens == 0


def test_assembled_system_message_carries_breakpoint(tmp_path):
    """TIER 1+2 = stable prefix; the marker lands where TIER 3 begins and the
    stable prefix is byte-stable across turns without compaction."""
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="persona",
        storage_dir=tmp_path,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
    )
    prefixes = []
    for i in range(3):
        agent.chat(f"hello {i}")
        msgs = agent.inspect_last_turn().messages_passed_to_llm1
        sys_msg = msgs[0]
        split = sys_msg.cache_stable_prefix_chars
        if split:
            assert "TIER 3" not in sys_msg.content[:split]
            prefixes.append(sys_msg.content[:split])
    if len(prefixes) >= 2:
        assert prefixes[-1] == prefixes[-2], "stable prefix must be byte-stable across turns"
