"""Provider abstraction tests (no network calls)."""
from __future__ import annotations

import asyncio

import pytest

from sherlock.config import ModelConfig
from sherlock.providers import FakeProvider, build_provider
from sherlock.providers.base import ChatMessage


def test_fake_provider_echoes_last_user() -> None:
    p = FakeProvider(model_id="fake/echo-test")
    resp = p.chat(
        [
            ChatMessage(role="system", content="You are X."),
            ChatMessage(role="user", content="hello sherlock"),
        ]
    )
    assert "hello sherlock" in resp.text
    assert resp.model == "fake/echo-test"
    assert resp.usage.total_tokens > 0
    assert resp.cost_usd == 0.0


def test_fake_provider_canned_reply() -> None:
    p = FakeProvider(canned_reply="canned-output")
    resp = p.chat([ChatMessage(role="user", content="anything")])
    assert resp.text == "canned-output"


def test_build_provider_dispatches_on_provider_field() -> None:
    cfg = ModelConfig(provider="fake", model="echo")
    p = build_provider(cfg)
    assert isinstance(p, FakeProvider)


def test_async_path_falls_back_to_thread() -> None:
    p = FakeProvider()

    async def _run() -> str:
        resp = await p.achat([ChatMessage(role="user", content="async hi")])
        return resp.text

    out = asyncio.run(_run())
    assert "async hi" in out
