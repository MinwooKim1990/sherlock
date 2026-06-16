"""Bootstrap engine. Calls LLM-1 to author LLM-2 and LLM-3 system prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sherlock.bootstrap.meta_context import META_CONTEXT
from sherlock.providers.base import BaseProvider, ChatMessage


@dataclass
class BootstrapResult:
    llm2_system_prompt: str
    llm3_system_prompt: str
    rationale: str


REQUIRED_KEYS = {"llm2_system_prompt", "llm3_system_prompt", "rationale"}


class BootstrapEngine:
    def __init__(
        self,
        main_provider: BaseProvider,
        main_system_prompt: str,
        domain_hints: list[str] | None = None,
    ) -> None:
        self._provider = main_provider
        self._main_prompt = main_system_prompt
        self._hints = domain_hints or []

    def _user_message(self) -> str:
        hints = "\n".join(f"- {h}" for h in self._hints) if self._hints else "(none provided)"
        return (
            "You are LLM 1. Below is YOUR main system prompt (the role the "
            "user gave you), followed by domain hints. After that, the "
            "Sherlock meta-context tells you what to author.\n\n"
            "--- YOUR MAIN SYSTEM PROMPT ---\n"
            f"{self._main_prompt}\n--- END ---\n\n"
            "--- DOMAIN HINTS ---\n"
            f"{hints}\n--- END ---\n\n"
            "--- SHERLOCK META-CONTEXT ---\n"
            f"{META_CONTEXT}\n--- END ---\n\n"
            "Now produce the JSON output described in the meta-context. "
            "Output JSON only."
        )

    def run(self) -> tuple[str, str]:
        result = self.run_with_rationale()
        return result.llm2_system_prompt, result.llm3_system_prompt

    def run_with_rationale(self) -> BootstrapResult:
        msgs = [
            ChatMessage(role="system", content="You author companion prompts. Output JSON only."),
            ChatMessage(role="user", content=self._user_message()),
        ]
        resp = self._provider.chat(msgs)
        parsed = _parse_json_strict(resp.text)
        if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed.keys()):
            raise ValueError(
                f"Bootstrap output missing required keys. Got keys={list(parsed) if isinstance(parsed, dict) else type(parsed).__name__}"
            )
        for key in ("llm2_system_prompt", "llm3_system_prompt"):
            v = parsed[key]
            if not isinstance(v, str) or len(v) < 100:
                raise ValueError(
                    f"Bootstrap field {key} too short: {len(v) if isinstance(v, str) else type(v).__name__}"
                )
        return BootstrapResult(
            llm2_system_prompt=parsed["llm2_system_prompt"],
            llm3_system_prompt=parsed["llm3_system_prompt"],
            rationale=parsed.get("rationale", ""),
        )


def _parse_json_strict(text: str) -> object:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        body = text.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            try:
                return json.loads(inner.strip())
            except Exception:
                pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None
