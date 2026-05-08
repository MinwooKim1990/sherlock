"""Pytest fixtures shared across the suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the repo root is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_prompt(tmp_path: Path) -> Path:
    p = tmp_path / "main.md"
    p.write_text("You are a helpful test assistant.", encoding="utf-8")
    return p


@pytest.fixture
def fake_yaml(tmp_path: Path, tmp_prompt: Path) -> Path:
    yaml_path = tmp_path / "sherlock.yaml"
    yaml_path.write_text(
        f"""
project: sherlock_test
main_system_prompt:
  path: {tmp_prompt}
models:
  main:
    provider: fake
    model: echo
storage:
  sqlite_path: {tmp_path / "test.db"}
""",
        encoding="utf-8",
    )
    return yaml_path
