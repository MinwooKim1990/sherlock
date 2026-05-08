"""Config schema + YAML loader tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sherlock.config import Config, ModelConfig


def test_loads_minimal_yaml(fake_yaml: Path) -> None:
    cfg = Config.from_yaml(fake_yaml)
    assert cfg.project == "sherlock_test"
    assert cfg.models.main.provider == "fake"
    assert cfg.models.main.model == "echo"


def test_resolves_relative_paths(tmp_path: Path) -> None:
    prompt = tmp_path / "p.md"
    prompt.write_text("hi", encoding="utf-8")
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        """
project: rel
main_system_prompt:
  path: p.md
models:
  main:
    provider: fake
    model: echo
""",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(yaml_file)
    assert cfg.main_system_prompt.path.exists()
    assert cfg.main_system_prompt.path.is_absolute()


def test_missing_prompt_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        """
project: bad
main_system_prompt:
  path: /no/such/file.md
models:
  main:
    provider: fake
    model: echo
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        Config.from_yaml(yaml_file)


def test_litellm_model_id_routing() -> None:
    assert ModelConfig(provider="anthropic", model="claude-haiku").litellm_model_id() == "anthropic/claude-haiku"
    assert ModelConfig(provider="openai", model="gpt-5").litellm_model_id() == "gpt-5"
    assert ModelConfig(provider="gemini", model="gemini-3.1").litellm_model_id() == "gemini/gemini-3.1"
    assert ModelConfig(provider="xai", model="grok-4").litellm_model_id() == "xai/grok-4"
    assert ModelConfig(provider="ollama", model="llama3").litellm_model_id() == "ollama/llama3"


def test_resolved_api_key_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "sk-xxxxx")
    cfg = ModelConfig(provider="anthropic", model="claude", api_key_env="MY_TEST_KEY")
    assert cfg.resolved_api_key() == "sk-xxxxx"


def test_resolved_api_key_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNSET_TEST_KEY", raising=False)
    cfg = ModelConfig(provider="anthropic", model="claude", api_key_env="UNSET_TEST_KEY")
    assert cfg.resolved_api_key() is None
