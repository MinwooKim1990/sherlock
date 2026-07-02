"""Open-source-model aggregator providers (DeepInfra / Together / OpenRouter).

These are OpenAI-compatible hosts for open-weight models, wired two ways:
  * package path  — ModelConfig.litellm_model_id() + LiteLLMProvider env mirror
  * playground    — providers.list_models() (key → live model list) + resolve_model_spec()

All three route through litellm's native prefixes (deepinfra/ together_ai/
openrouter/), so the only per-provider logic is the /models response shape and
chat filter — covered here against the shapes verified live in research.
"""

from __future__ import annotations

import pytest

from sherlock.config import ModelConfig
from sherlock.providers.litellm_provider import LiteLLMProvider


# ---------- package path -----------------------------------------------------
def test_litellm_model_id_oss_prefixes() -> None:
    assert (
        ModelConfig(provider="deepinfra", model="meta-llama/Llama-3.3-70B").litellm_model_id()
        == "deepinfra/meta-llama/Llama-3.3-70B"
    )
    assert (
        ModelConfig(provider="openrouter", model="anthropic/claude-3.5").litellm_model_id()
        == "openrouter/anthropic/claude-3.5"
    )


def test_together_alias_normalizes_to_together_ai() -> None:
    # users reach for "together"; litellm's slug is "together_ai"
    assert (
        ModelConfig(provider="together", model="openai/gpt-oss-20b").litellm_model_id()
        == "together_ai/openai/gpt-oss-20b"
    )
    assert ModelConfig(provider="together_ai", model="x/y").litellm_model_id() == "together_ai/x/y"


def test_canonical_env_var_for_oss_providers() -> None:
    assert LiteLLMProvider._canonical_env_var("deepinfra") == "DEEPINFRA_API_KEY"
    assert LiteLLMProvider._canonical_env_var("together") == "TOGETHERAI_API_KEY"
    assert LiteLLMProvider._canonical_env_var("together_ai") == "TOGETHERAI_API_KEY"
    assert LiteLLMProvider._canonical_env_var("openrouter") == "OPENROUTER_API_KEY"


def test_provider_init_mirrors_key_to_litellm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # the package path relies on env mirroring: the user's api_key_env is copied
    # into the name litellm reads, so litellm.completion auths without extra wiring
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setenv("MY_DI_KEY", "sk-di-xyz")
    cfg = ModelConfig(provider="deepinfra", model="meta-llama/x", api_key_env="MY_DI_KEY")
    LiteLLMProvider(cfg)
    import os

    assert os.environ["DEEPINFRA_API_KEY"] == "sk-di-xyz"
    # The provider mirrors the key via a DIRECT os.environ write (not monkeypatch),
    # so clean it up explicitly to not leak "sk-di-xyz" into later tests.
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)


# ---------- playground path --------------------------------------------------
class _FakeResp:
    def __init__(self, body):
        self._b = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._b


def _patch_httpx(monkeypatch, body_by_url, capture):
    def fake_get(url, headers=None, timeout=None, params=None):
        capture["url"] = url
        capture["headers"] = headers or {}
        return _FakeResp(body_by_url[url])

    import playground.providers as P

    monkeypatch.setattr(P.httpx, "get", fake_get)


def test_list_deepinfra_filters_to_chat_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    import playground.providers as P

    cap: dict = {}
    body = {
        "object": "list",
        "data": [
            {"id": "meta-llama/Llama-3.3-70B", "metadata": {"tags": ["chat", "reasoning"]}},
            {"id": "BAAI/bge-m3", "metadata": {"tags": ["embed"]}},
            {"id": "bfl/FLUX", "metadata": {"tags": ["image-gen"]}},
        ],
    }
    _patch_httpx(monkeypatch, {P.OPENAI_COMPAT["deepinfra"]["models_url"]: body}, cap)
    out = P.list_models("deepinfra", api_key="ignored-for-listing")
    assert [m["id"] for m in out] == ["meta-llama/Llama-3.3-70B"]
    # public list — a (possibly invalid) key must NOT be sent: DeepInfra 401s on it
    assert "Authorization" not in cap["headers"]


def test_list_together_handles_bare_array_and_type_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    import playground.providers as P

    cap: dict = {}
    # Together returns a TOP-LEVEL array (no {"data": ...} envelope)
    body = [
        {"id": "meta-llama/Llama-3.3-70B-Turbo", "type": "chat", "created": 100},
        {"id": "togethercomputer/m2-bert", "type": "embedding", "created": 50},
        {"id": "openai/gpt-oss-20b", "type": "chat", "created": 200},
    ]
    _patch_httpx(monkeypatch, {P.OPENAI_COMPAT["together"]["models_url"]: body}, cap)
    out = P.list_models("together", api_key="tg-key")
    assert [m["id"] for m in out] == ["openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Turbo"]
    # Together's list DOES need the key
    assert cap["headers"].get("Authorization") == "Bearer tg-key"


def test_list_openrouter_drops_non_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import playground.providers as P

    cap: dict = {}
    body = {
        "data": [
            {
                "id": "anthropic/claude-3.5",
                "architecture": {"output_modalities": ["text"]},
                "created": 300,
            },
            {
                "id": "openai/dall-e-3",
                "architecture": {"output_modalities": ["image"]},
                "created": 250,
            },
            {
                "id": "meta-llama/llama-3.3-70b",
                "architecture": {"output_modalities": ["text"]},
                "created": 400,
            },
        ]
    }
    _patch_httpx(monkeypatch, {P.OPENAI_COMPAT["openrouter"]["models_url"]: body}, cap)
    out = P.list_models("openrouter", api_key="")  # public list works keyless
    assert [m["id"] for m in out] == ["meta-llama/llama-3.3-70b", "anthropic/claude-3.5"]


def test_resolve_model_spec_oss_prefixes() -> None:
    import playground.providers as P

    creds = {
        "deepinfra": {"api_key": "di"},
        "together": {"api_key": "tg"},
        "openrouter": {"api_key": "or"},
    }
    mid, extra = P.resolve_model_spec({"provider": "deepinfra", "model": "meta-llama/X"}, creds)
    assert mid == "deepinfra/meta-llama/X" and extra == {"api_key": "di"}

    mid, extra = P.resolve_model_spec(
        {"provider": "together", "model": "openai/gpt-oss-20b"}, creds
    )
    assert mid == "together_ai/openai/gpt-oss-20b" and extra == {"api_key": "tg"}

    mid, extra = P.resolve_model_spec({"provider": "openrouter", "model": "anthropic/c"}, creds)
    assert mid == "openrouter/anthropic/c"
    assert extra["api_key"] == "or"
    assert extra["extra_headers"] == {"X-Title": "Sherlock"}  # cosmetic attribution
