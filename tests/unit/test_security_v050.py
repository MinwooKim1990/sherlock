"""v0.5.0 Phase 4 — SSRF guard + redaction."""

from __future__ import annotations

import pytest

from sherlock.security.redaction import redact, redact_findings
from sherlock.security.urlguard import is_safe_url

# ---- SSRF guard ----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://evil/x",
    ],
)
def test_unsafe_urls_blocked(url):
    ok, reason = is_safe_url(url)
    assert ok is False, f"{url} should be blocked, got ok (reason={reason})"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://duckduckgo.com/?q=test",
        "http://example.com/page",
    ],
)
def test_public_urls_allowed(url):
    ok, reason = is_safe_url(url)
    if not ok and "dns" in reason.lower():
        pytest.skip(f"no network for DNS resolution: {reason}")
    assert ok is True, f"{url} should be allowed, reason={reason}"


def test_fetch_blocks_metadata_endpoint():
    from sherlock.tools.web_search import _default_fetch

    out = _default_fetch("http://169.254.169.254/latest/meta-data/")
    assert "error" in out
    assert "blocked" in out["error"].lower()


def test_dns_rebinding_blocked_offline():
    """A public hostname that resolves to a private IP must be blocked.
    Uses an injected resolver so the rebinding defense is verified with NO
    network (the prior test relied on real DNS and skipped when offline).
    """
    # Hostname looks public, but "resolves" to an internal address.
    ok, reason = is_safe_url("https://totally-public.example", resolver=lambda h, p: ["10.0.0.5"])
    assert ok is False
    assert "blocked" in reason.lower()
    # Same hostname resolving to a real public IP is allowed — proves the
    # block is about the resolved IP, not the name.
    ok2, _ = is_safe_url("https://totally-public.example", resolver=lambda h, p: ["93.184.216.34"])
    assert ok2 is True


def test_fetch_rejects_rebinding_via_injected_resolver():
    """End-to-end: _default_fetch refuses to even open the client when the
    (injected) resolver maps the host to a private IP."""
    from sherlock.tools.web_search import _default_fetch

    out = _default_fetch("https://internal.example", resolver=lambda h, p: ["192.168.1.10"])
    assert "error" in out and "blocked" in out["error"].lower()


# ---- redaction -----------------------------------------------------------


def test_redact_openai_key():
    s = "my key is sk-abcdefghijklmnopqrstuvwxyz123456 ok"
    out = redact(s)
    assert "sk-abcdefghijklmnop" not in out
    assert "[REDACTED:api_key]" in out


def test_redact_anthropic_key():
    s = "ANTHROPIC_API_KEY=sk-ant-api03-aaaaaaaaaaaaaaaaaaaa"
    out = redact(s)
    assert "sk-ant-api03" not in out


def test_redact_email():
    out = redact("contact me at jiwon.park@example.com please")
    assert "jiwon.park@example.com" not in out
    assert "[REDACTED:email]" in out


def test_redact_bearer_and_jwt():
    out = redact("Authorization: Bearer abcdef0123456789abcdef")
    assert "[REDACTED:bearer]" in out


def test_redact_findings_reports_labels():
    _, labels = redact_findings("sk-abcdefghijklmnopqrstuvwx and foo@bar.com")
    assert "api_key" in labels
    assert "email" in labels


def test_redact_noop_on_clean_text():
    s = "just a normal sentence about the weather"
    assert redact(s) == s


# ---- integration: redaction on memory write path -------------------------


def test_secret_not_persisted_to_memory(tmp_path):
    from sherlock import Sherlock

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
        redact_secrets=True,
    )
    agent.chat("here is my key sk-abcdefghijklmnopqrstuvwxyz123456 keep it safe")
    # The raw transcript keeps it (faithful), but memory must be redacted.
    mems = agent.memory.list(conversation_id=agent.conversation_id)
    joined = " ".join(m.content for m in mems)
    assert "sk-abcdefghijklmnop" not in joined, "secret leaked into memory/RAG"


def test_store_level_redaction_covers_all_writes(tmp_path):
    """The redactor injected into MemoryStore must scrub content regardless of
    which path calls add() — so a secret cannot leak via LLM-2/LLM-3/freshness.
    This is the single-choke-point guarantee; it's tested directly on the store
    so it holds even if callers forget to pre-redact.
    """
    from sherlock.memory import MemoryStore, build_embedding_provider
    from sherlock.memory.entry import MemorySource, MemoryType
    from sherlock.security.redaction import redact
    from sherlock.storage import Storage

    class _FakeEmbedConfig:
        provider = "fake"
        model = "fake-embedding"
        api_key_env = None

    storage = Storage(tmp_path / "db.sqlite")
    embed = build_embedding_provider(_FakeEmbedConfig())
    store = MemoryStore(
        engine=storage.engine,
        embedding_provider=embed,
        vector_path=tmp_path / "vec",
        redactor=redact,
    )
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    e = store.add(
        conversation_id="c",
        content=f"the deploy key is {secret} btw",
        type=MemoryType.FACT,
        source=MemorySource.SEARCH,
    )
    # Stored row content is redacted ...
    assert secret not in e.content
    assert "[REDACTED:api_key]" in e.content
    # ... and so is the Chroma-side document (what RAG returns).
    got = store.search("deploy key", conversation_id="c", top_k=1)
    assert got and secret not in got[0][0].content


def test_secret_in_llm2_fact_not_persisted(tmp_path):
    """Regression (v0.5.0 external review): a secret re-emitted by the LLM-2
    summarizer as an extracted FACT bypassed the user-utterance-only redaction
    and was persisted verbatim. The store-level redactor must catch it.
    """
    import json

    from sherlock import Sherlock

    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"

    def main(messages):
        return "noted.\n<<sherlock-companions: compact>>"

    def summary(messages):
        return json.dumps(
            {
                "summary": f"User shared a deploy key: {secret}",
                "facts": [
                    {
                        "content": f"User's API key is {secret}",
                        "type": "fact",
                        "source": "user",
                        "confidence": 1.0,
                        "pin_recommended": True,
                    },
                ],
                "topic_label": "secrets",
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
                "persona_summary": f"User who pasted {secret} once.",
                "predicted_directions": [],
                "worth_digging": [],
            }
        )

    agent = Sherlock.with_callable(
        main_chat=main,
        summary_chat=summary,
        system_prompt="x",
        storage_dir=tmp_path,
        redact_secrets=True,
    )
    agent._turn_index = 10  # bypass cold-start so compact fires
    agent.chat("here, write this down")
    agent.drain()

    mems = agent.memory.list(conversation_id=agent.conversation_id)
    joined = " ".join(m.content for m in mems)
    assert secret not in joined, f"secret leaked via LLM-2 fact/summary/persona path: {joined!r}"
    # And it must not surface in the assembled system prompt either.
    assert secret not in agent._format_persona_summary_block(agent.conversation_id)


def test_store_redacts_all_string_fields(tmp_path):
    """Regression (v0.5.1 review): redaction must cover evidence, tags, and
    semantic_triple — not just content. A secret in any of those previously
    persisted verbatim (and leaked via the eval ledger / memory tool / entity
    index). The evidence JSON must remain parseable after redaction.
    """
    import json

    from sherlock.memory import MemoryStore, build_embedding_provider
    from sherlock.memory.entry import MemoryEntity, MemorySource, MemoryType
    from sherlock.security.redaction import redact
    from sherlock.storage import Storage
    from sqlmodel import Session, select

    class _FakeEmbedConfig:
        provider = "fake"
        model = "fake-embedding"
        api_key_env = None

    storage = Storage(tmp_path / "db.sqlite")
    embed = build_embedding_provider(_FakeEmbedConfig())
    store = MemoryStore(
        engine=storage.engine,
        embedding_provider=embed,
        vector_path=tmp_path / "vec",
        redactor=redact,
    )
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    e = store.add(
        conversation_id="c",
        content=f"deploy key {secret}",
        type=MemoryType.FACT,
        source=MemorySource.SEARCH,
        evidence=json.dumps([f"user said {secret}", "ctx"]),
        tags=f"security,{secret}",
        semantic_triple=("user", "has_api_key", secret),
    )
    # Every string field on the row is scrubbed.
    assert secret not in e.content
    assert secret not in (e.evidence or "")
    assert secret not in (e.tags or "")
    assert secret not in (e.semantic_triple_object or "")
    # Evidence stays valid JSON (placeholder has no quotes/commas/braces).
    assert json.loads(e.evidence) and secret not in " ".join(json.loads(e.evidence))
    # The persistent entity index never holds the raw secret either.
    with Session(store._engine) as s:
        rows = list(s.exec(select(MemoryEntity).where(MemoryEntity.memory_id == e.id)))
    assert all(secret not in r.entity for r in rows)


def test_builtin_url_fetch_blocks_metadata():
    """Builtin _url_fetch (defense-in-depth) must refuse the cloud-metadata
    endpoint just like the agent-facing fetch does."""
    from sherlock.tools.builtin import _url_fetch

    out = _url_fetch("http://169.254.169.254/latest/meta-data/")
    assert "error" in out and "block" in out["error"].lower()


def test_builtin_url_fetch_rebinding_blocked_offline():
    """A public-looking host that resolves to a private IP must be blocked —
    verified offline via an injected resolver (no DNS)."""
    from sherlock.tools.builtin import _url_fetch

    out = _url_fetch("https://internal.example", resolver=lambda h, p: ["10.0.0.5"])
    assert "error" in out and "block" in out["error"].lower()


def test_per_conversation_tool_cap(tmp_path):
    from sherlock import Sherlock
    from sherlock.tools.web_search import StubSearch

    agent = Sherlock.with_callable(
        main_chat=lambda m: "ok",
        system_prompt="x",
        storage_dir=tmp_path,
        main_search_engine=StubSearch(),
    )
    agent.config.execution.max_tool_calls_per_conversation = 2
    r1 = agent._execute_tool_call("search", "a")
    r2 = agent._execute_tool_call("search", "b")
    r3 = agent._execute_tool_call("search", "c")
    assert "results" in r1 and "results" in r2
    assert "error" in r3 and "cap" in r3["error"].lower()
