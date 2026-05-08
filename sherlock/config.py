"""Pydantic config schema and YAML loader.

M1-relevant subset of SPEC.md § 8.3. M2+ fields are declared with sane
defaults so downstream code can read them without breaking when omitted.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class MainPromptConfig(BaseModel):
    path: Path
    domain_hints: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key_env: str | None = None
    api_base: str | None = None  # for Ollama / LM Studio / proxies

    def resolved_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    def litellm_model_id(self) -> str:
        """Return the model id in litellm's expected format.

        litellm expects "anthropic/claude-...", "openai/gpt-...",
        "gemini/gemini-...", "ollama/...", "openrouter/...", etc.
        Some providers (openai) don't need the prefix. We normalise here.
        """
        prov = self.provider.lower()
        if prov in {"openai"}:
            return self.model
        return f"{prov}/{self.model}"


class ModelsConfig(BaseModel):
    main: ModelConfig
    background_summary: ModelConfig | None = None
    background_inference: ModelConfig | None = None


class StorageConfig(BaseModel):
    sqlite_path: Path = Path("./sherlock.db")
    vector_db: Literal["chroma", "lancedb"] = "chroma"
    vector_path: Path = Path("./sherlock_vectors")


class MemoryConfig(BaseModel):
    k_turn_min: int = 3
    k_turn_max_adaptive: bool = True


class ExecutionConfig(BaseModel):
    parallel_when_possible: bool = True
    max_concurrent_background_tasks: int = 3
    cost_cap_per_turn_usd: float = 0.50
    fallback_to_sequential_on_local: bool = True


class Config(BaseModel):
    """Root config. M1-relevant fields are required; M2+ fields default to spec values."""

    project: str = "sherlock_default"
    main_system_prompt: MainPromptConfig
    models: ModelsConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("main_system_prompt", mode="after")
    @classmethod
    def _check_prompt_exists(cls, v: MainPromptConfig) -> MainPromptConfig:
        if not v.path.exists():
            raise ValueError(f"main_system_prompt.path does not exist: {v.path}")
        return v

    def read_main_system_prompt(self) -> str:
        return self.main_system_prompt.path.read_text(encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp) or {}
        # Resolve relative paths against the YAML file's directory so configs
        # stay portable.
        base = path.parent
        if isinstance(raw.get("main_system_prompt"), dict):
            p = raw["main_system_prompt"].get("path")
            if p and not Path(p).is_absolute():
                raw["main_system_prompt"]["path"] = str((base / p).resolve())
        if isinstance(raw.get("storage"), dict):
            for key in ("sqlite_path", "vector_path"):
                p = raw["storage"].get(key)
                if p and not Path(p).is_absolute():
                    raw["storage"][key] = str((base / p).resolve())
        return cls.model_validate(raw)
