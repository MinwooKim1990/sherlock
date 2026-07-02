"""v1.11 observability: silent failures made visible + redaction fail-closed.

- a redactor CRASH must not leak the raw (possibly-secret) text into memory
  (fail-closed) and must emit `memory.redaction_failed`.
- the v1.10 verify chain silently no-ops when there's no raw to check against
  (e.g. deep_research_reconstruct_from_raw=False); it now emits
  `deep_research.verify_skipped {stage, reason}` so users can see WHY.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.tools.web_search import SearchEngine


def test_redaction_failure_is_fail_closed(tmp_path, monkeypatch):
    import sherlock.security.redaction as red

    def boom(_text):
        raise RuntimeError("redactor down")

    monkeypatch.setattr(red, "redact", boom)
    events: list[tuple] = []
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        redact_secrets=True,
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    out = agent._redact_for_memory("my api key is sk-SECRET-123")
    assert "sk-SECRET-123" not in out, "raw secret must NOT survive a redactor crash"
    assert out == "[redaction unavailable — content withheld]"
    assert any(t == "memory.redaction_failed" for t, _ in events), "failure not surfaced"


def test_redaction_disabled_is_untouched(tmp_path, monkeypatch):
    # redact_secrets off → exact passthrough, redactor never even imported/called
    import sherlock.security.redaction as red

    monkeypatch.setattr(red, "redact", lambda _t: (_ for _ in ()).throw(AssertionError("called")))
    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
    )
    assert agent._redact_for_memory("plain text sk-XYZ") == "plain text sk-XYZ"


class _E(SearchEngine):
    def search(self, q, *, max_results=5):
        return [{"title": "alpha", "url": "https://e/1", "content": "alpha detail"}]

    def fetch(self, url, *, raw=False, timeout=10.0):
        return {"url": url, "status": 200, "text": f"page {url}"}


def _dr_main(messages):
    c = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    if "RESEARCH STRATEGY" in c:
        return json.dumps(
            {
                "objective": "o",
                "sub_topics": ["alpha"],
                "scope": {"include": [], "exclude": []},
                "clarifying_questions": [],
            }
        )
    if "Answer these meta-questions" in c:
        return json.dumps(
            {
                "facts": [{"fact": "alpha detail", "sources": ["https://e/1"]}],
                "key_finding": "k",
                "summary": "s",
                "gaps": [],
                "sufficient": True,
                "next_queries": [],
            }
        )
    if "FAITHFULNESS-checking" in c or "CONSISTENCY checker" in c:
        return json.dumps({"fixes": []})
    return "## R\nalpha detail https://e/1"


def test_verify_skipped_emitted_when_no_raw(tmp_path):
    events: list[tuple] = []
    agent = Sherlock.with_callable(
        main_chat=_dr_main,
        system_prompt="x",
        storage_dir=tmp_path,
        embedding="fake",
        background=False,
        main_search_engine=_E(),
        inference_search_engine="disabled",
    )
    # no raw kept → faithfulness has nothing to check → must SAY so, not run silently
    agent.config.search.deep_research_reconstruct_from_raw = False
    prompts: list[str] = []
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))

    orig = agent._provider.chat

    def spy(messages, *a, **k):
        prompts.append(messages[-1].content if messages else "")
        return orig(messages, *a, **k)

    agent._provider.chat = spy  # capture prompts to prove faithfulness didn't run
    agent._run_deep_research(agent._ensure_conversation().id, "topic", 1, "sk")

    skipped = [d for (t, d) in events if t == "deep_research.verify_skipped"]
    assert any(
        d.get("stage") == "faithfulness" and d.get("reason") == "no_raw" for d in skipped
    ), f"expected a faithfulness/no_raw verify_skipped event, got {skipped}"
    assert not any("FAITHFULNESS-checking" in p for p in prompts), "faithfulness ran despite no raw"
