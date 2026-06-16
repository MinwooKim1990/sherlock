"""M1 exit-criterion integration test.

Exit (SPEC.md § 9 M1): `sherlock chat` produces conversation; provider can
be switched via config without code change.

This test exercises both halves with the FakeProvider (hermetic) and a
flag-gated live provider call when API keys exist in env.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sherlock import Sherlock


def _write_yaml(
    yaml_path: Path, prompt_path: Path, db_path: Path, *, provider: str, model: str
) -> None:
    yaml_path.write_text(
        f"""
project: m1_test
main_system_prompt:
  path: {prompt_path}
models:
  main:
    provider: {provider}
    model: {model}
storage:
  sqlite_path: {db_path}
""",
        encoding="utf-8",
    )


def test_chat_persists_messages_with_fake_provider(tmp_path: Path) -> None:
    prompt = tmp_path / "p.md"
    prompt.write_text("system prompt", encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    db_path = tmp_path / "m1.db"
    _write_yaml(yaml_path, prompt, db_path, provider="fake", model="echo")

    agent = Sherlock.from_yaml(yaml_path)
    reply = agent.chat("hi sherlock")
    assert "hi sherlock" in reply  # FakeProvider echoes
    msgs = agent.messages()
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1].content == "hi sherlock"
    assert msgs[2].content == reply


def test_provider_switch_is_config_only(tmp_path: Path) -> None:
    """Switching providers requires editing YAML, not code. Verify by loading two configs."""
    prompt = tmp_path / "p.md"
    prompt.write_text("system prompt", encoding="utf-8")

    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    _write_yaml(cfg_a, prompt, db_a, provider="fake", model="model-A")
    _write_yaml(cfg_b, prompt, db_b, provider="fake", model="model-B")

    agent_a = Sherlock.from_yaml(cfg_a)
    agent_b = Sherlock.from_yaml(cfg_b)

    assert agent_a.provider.model_id == "model-A"
    assert agent_b.provider.model_id == "model-B"

    # And both work end-to-end on the same code path:
    rep_a = agent_a.chat("ping a")
    rep_b = agent_b.chat("ping b")
    assert "ping a" in rep_a
    assert "ping b" in rep_b


def test_multi_turn_history_is_preserved(tmp_path: Path) -> None:
    prompt = tmp_path / "p.md"
    prompt.write_text("you remember", encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    db_path = tmp_path / "multi.db"
    _write_yaml(yaml_path, prompt, db_path, provider="fake", model="echo")

    agent = Sherlock.from_yaml(yaml_path)
    agent.chat("first")
    agent.chat("second")
    agent.chat("third")

    msgs = agent.messages()
    assert len(msgs) == 1 + 3 * 2  # system + 3 (user, assistant) pairs
    assert msgs[1].content == "first"
    assert msgs[3].content == "second"
    assert msgs[5].content == "third"


def test_inspect_last_turn_returns_state(tmp_path: Path) -> None:
    prompt = tmp_path / "p.md"
    prompt.write_text("sys", encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    db_path = tmp_path / "inspect.db"
    _write_yaml(yaml_path, prompt, db_path, provider="fake", model="echo")

    agent = Sherlock.from_yaml(yaml_path)
    assert agent.inspect_last_turn() is None
    agent.chat("anything")
    state = agent.inspect_last_turn()
    assert state is not None
    assert state.user_text == "anything"
    assert state.response.text


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY for a live provider check.",
)
def test_live_anthropic_smoke(tmp_path: Path) -> None:
    """Optional live check: only runs when ANTHROPIC_API_KEY is set in env."""
    prompt = tmp_path / "p.md"
    prompt.write_text(
        "Reply with exactly: SHERLOCK_M1_LIVE_OK",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "cfg.yaml"
    db_path = tmp_path / "live.db"
    yaml_path.write_text(
        f"""
project: m1_live
main_system_prompt:
  path: {prompt}
models:
  main:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
storage:
  sqlite_path: {db_path}
""",
        encoding="utf-8",
    )
    agent = Sherlock.from_yaml(yaml_path)
    reply = agent.chat("Output the marker.")
    assert reply  # any non-empty reply means the wire path works
