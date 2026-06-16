"""Interactive Sherlock v0.4.0 test CLI.

Run from a venv:

    python test_sherlock.py

Or with explicit provider override:

    python test_sherlock.py --provider anthropic

Fill in the API key for the provider you want to use, either via env var
(recommended — see :data:`API_KEY_ENV`) or by editing the constants
below. **Do NOT commit this file with a hardcoded key.**

The CLI is a REPL. Type messages to chat with Sherlock. Type `/help` for
the full command list, or `/quit` to exit.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Optional

import httpx

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION — fill these in, or set the corresponding env vars
# ════════════════════════════════════════════════════════════════════════

# Pick one: "openrouter" | "openai" | "anthropic" | "google" | "xai"
PROVIDER = "openrouter"

# API keys. Either fill them here OR (recommended) set the env var. Env
# var wins when set. Leaving both empty for the chosen provider will
# fail fast at startup with a clear error.
OPENROUTER_API_KEY = ""
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GOOGLE_API_KEY = ""
XAI_API_KEY = ""

# Env-var names checked when the constant above is empty.
API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}

# Default model per provider. Override per-provider as you like.
DEFAULT_MODEL = {
    "openrouter": "anthropic/claude-haiku-4.5",
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-2.5-flash-lite",
    "xai": "grok-4",
}

# Override the model used (None → DEFAULT_MODEL[PROVIDER]).
MODEL = None

# Optional explicit context-window size. Leave None to auto-detect:
#   - OpenRouter: fetched from /api/v1/models at startup
#   - Anthropic / OpenAI / Google: looked up in Sherlock's registry
#   - xAI / Grok: defaults to 256K (registry doesn't have entries yet)
# Set explicitly if auto-detect fails (e.g., a model that's not in either).
CONTEXT_WINDOW_OVERRIDE: Optional[int] = None

# Your role/persona prompt. Sherlock's internal protocol (companion
# tags + tool tags + cross-verify discipline) rides alongside this.
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a candid, casual assistant who is also Sherlock-style observant.
    Speak naturally in whatever language the user uses. When you're not
    sure of a fact, hedge or use the search/memory tools rather than
    guessing.
""").strip()

# Where to persist sessions / memory. ./test_sherlock_state/ is local
# and easy to delete; switch to a permanent path if you want to keep
# the conversation across reboots.
STORAGE_DIR = Path("./test_sherlock_state")

# Sherlock's web-search tool. True → DuckDuckGo (free, no key).
# False → search tool disabled for both LLM-1 and LLM-3.
ENABLE_WEB_SEARCH = True

# v0.5.0 — embeddings (semantic memory). "local" = real multilingual
# embeddings via fastembed (no API key; `pip install sherlock[embeddings]`).
# "fake" = hash vectors (no real recall). "openai"/"voyage"/"cohere" = keyed.
EMBEDDING = "local"
EMBEDDING_MODEL: Optional[str] = None  # None → built-in multilingual default

# Redact secrets/PII (API keys, tokens, emails…) before they enter
# long-term memory/RAG. The raw transcript is never redacted.
REDACT_SECRETS = True

# True → main reply returns immediately; companions/decay run in a
# background worker (production feel). False → inline (simpler to trace).
BACKGROUND = True

# Safety guards. Set MAX_USER_INPUT_TOKENS to a positive int to refuse
# pastes larger than that. None disables the guard.
MAX_USER_INPUT_TOKENS: Optional[int] = 20_000

# httpx timeout for provider calls (seconds). Sherlock's own background
# timeouts apply on top.
HTTP_TIMEOUT_SECONDS = 60.0


# ════════════════════════════════════════════════════════════════════════
# Pricing table (approximate; for /cost rough estimate). USD per 1M tokens.
# Update as your provider's prices change.
# ════════════════════════════════════════════════════════════════════════

PRICING_PER_MTOKEN: dict[str, dict[str, float]] = {
    # Pattern → {"in": $, "out": $}.  Glob-matched against MODEL.
    "*claude-haiku-4-5*": {"in": 1.0, "out": 5.0},
    "*claude-sonnet-4*": {"in": 3.0, "out": 15.0},
    "*claude-opus-4-7*": {"in": 15.0, "out": 75.0},
    "*gpt-5*": {"in": 1.0, "out": 5.0},
    "*gpt-4o-mini*": {"in": 0.15, "out": 0.60},
    "*gpt-4o*": {"in": 2.5, "out": 10.0},
    "*gemini-2.5*": {"in": 0.10, "out": 0.40},
    "*gemini-3.0*": {"in": 0.30, "out": 1.20},
    "*gemini-3.1*": {"in": 1.25, "out": 5.0},
    "*grok-2*": {"in": 2.0, "out": 10.0},
    "*grok-4*": {"in": 5.0, "out": 15.0},
}


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def resolve_api_key(provider: str) -> str:
    """Pick the API key for ``provider``: env var first, hardcoded second."""
    env_name = API_KEY_ENV[provider]
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return env_val
    hardcoded = {
        "openrouter": OPENROUTER_API_KEY,
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "google": GOOGLE_API_KEY,
        "xai": XAI_API_KEY,
    }[provider].strip()
    return hardcoded


def lookup_price(model: str) -> Optional[dict]:
    """Return {"in": ..., "out": ...} for a model, or None if unknown."""
    import fnmatch

    for pat, prices in PRICING_PER_MTOKEN.items():
        if fnmatch.fnmatchcase(model, pat):
            return prices
    return None


# ── Shared HTTP client (reused across calls — avoids per-turn TLS handshake) ──

_HTTP_CLIENT: Optional[httpx.Client] = None


def _http() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.Client(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS, connect=10.0))
    return _HTTP_CLIENT


# ── Provider call shapes ───────────────────────────────────────────────


def _extract_system_and_user_msgs(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split system messages (joined) from the rest. Used by Anthropic
    and Gemini, which both treat system as a separate field.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]
    return ("\n\n".join(system_parts), body)


def _openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    extra_headers: Optional[dict] = None,
    max_tokens: int = 4096,
) -> str:
    """Hits an OpenAI-compatible /chat/completions endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    r = _http().post(f"{base_url}/chat/completions", headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"] or ""


def _anthropic_chat(
    *, api_key: str, model: str, messages: list[dict], max_tokens: int = 4096
) -> str:
    system, body = _extract_system_and_user_msgs(messages)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": m["role"], "content": m["content"]} for m in body],
    }
    if system:
        payload["system"] = system
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    r = _http().post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    parts = data.get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _gemini_chat(*, api_key: str, model: str, messages: list[dict], max_tokens: int = 4096) -> str:
    system, body = _extract_system_and_user_msgs(messages)
    contents = []
    for m in body:
        # Gemini uses "model" instead of "assistant".
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload: dict = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    r = _http().post(url, json=payload, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def build_chat_callable(provider: str, api_key: str, model: str):
    """Return a ``chat(messages) -> str`` function for the chosen provider."""

    def chat(messages: list[dict]) -> str:
        # Normalise into role/content dicts (drop unknown keys).
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        try:
            if provider == "openrouter":
                return _openai_compatible_chat(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=model,
                    messages=msgs,
                    extra_headers={
                        # OpenRouter likes a referrer + title for cost dashboards
                        "HTTP-Referer": "https://github.com/MinwooKim1990/sherlock-test-cli",
                        "X-Title": "Sherlock test CLI",
                    },
                )
            if provider == "openai":
                return _openai_compatible_chat(
                    base_url="https://api.openai.com/v1",
                    api_key=api_key,
                    model=model,
                    messages=msgs,
                )
            if provider == "anthropic":
                return _anthropic_chat(api_key=api_key, model=model, messages=msgs)
            if provider == "google":
                return _gemini_chat(api_key=api_key, model=model, messages=msgs)
            if provider == "xai":
                return _openai_compatible_chat(
                    base_url="https://api.x.ai/v1",
                    api_key=api_key,
                    model=model,
                    messages=msgs,
                )
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            return f"[provider error {e.response.status_code}: {body}]"
        except httpx.TimeoutException:
            return "[timeout — provider did not respond within 60s]"
        except httpx.HTTPError as e:
            return f"[provider network error: {type(e).__name__}: {e}]"
        return f"[unknown provider: {provider}]"

    return chat


# ── Context-window resolution ──────────────────────────────────────────


def fetch_openrouter_context_length(model: str) -> Optional[int]:
    """One-off GET against OpenRouter's /models. Returns None on failure."""
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get("https://openrouter.ai/api/v1/models")
            r.raise_for_status()
            data = r.json()
        for item in data.get("data") or []:
            if item.get("id") == model:
                ctx = item.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
    except Exception:
        return None
    return None


def resolve_ctx_window(provider: str, model: str) -> int:
    """Resolve the context window we should budget against."""
    if CONTEXT_WINDOW_OVERRIDE is not None and CONTEXT_WINDOW_OVERRIDE > 0:
        return CONTEXT_WINDOW_OVERRIDE
    if provider == "openrouter":
        live = fetch_openrouter_context_length(model)
        if live is not None:
            return live
    if provider == "xai":
        # Sherlock's registry doesn't have Grok entries (yet).
        # Grok 4 / grok-2 both publish 256K. Default to that.
        return 256_000
    from sherlock.budget import resolve_context_window

    return resolve_context_window(model)


def patch_agent_budget(agent, ctx_window: int) -> None:
    """Inject a custom context window AFTER construction.

    `agent._ctx_window` feeds the per-turn `k_turn_budget` math, but
    `agent._slot_budget` is the profile snapshot picked at init time —
    we have to refresh both.
    """
    from sherlock.budget import (
        DEFAULT_PROFILE,
        SMALL_MODEL_PROFILE,
        apply_overrides,
        select_profile_for_window,
    )

    agent._ctx_window = ctx_window
    profile_choice = getattr(agent.config.memory, "slot_budget_profile", "auto")
    if profile_choice == "off":
        agent._slot_budget = None
        return
    if profile_choice == "default":
        base = DEFAULT_PROFILE
    elif profile_choice == "small":
        base = SMALL_MODEL_PROFILE
    else:
        base = select_profile_for_window(ctx_window)
    agent._slot_budget = apply_overrides(
        base, getattr(agent.config.memory, "slot_budget_overrides", {}) or {}
    )


# ── Cost tracking ──────────────────────────────────────────────────────


class CostMeter:
    """Approximate per-turn cost based on token usage + pricing table."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.price = lookup_price(model)
        self.in_tokens = 0
        self.out_tokens = 0
        self.calls = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.in_tokens += max(0, int(prompt_tokens or 0))
        self.out_tokens += max(0, int(completion_tokens or 0))
        self.calls += 1

    def estimate_usd(self) -> float:
        if self.price is None:
            return 0.0
        return (
            self.in_tokens * self.price["in"] / 1_000_000
            + self.out_tokens * self.price["out"] / 1_000_000
        )

    def report(self) -> str:
        if self.price is None:
            return (
                f"calls={self.calls} in_tokens={self.in_tokens} out_tokens={self.out_tokens} "
                f"(no pricing entry for {self.model})"
            )
        return (
            f"calls={self.calls} in={self.in_tokens} out={self.out_tokens} "
            f"≈ ${self.estimate_usd():.4f} (USD, approx)"
        )


# ════════════════════════════════════════════════════════════════════════
# REPL
# ════════════════════════════════════════════════════════════════════════

HELP_TEXT = textwrap.dedent("""\
    Commands:
      /help              show this help
      /quit              exit (saves session)
      /inspect           dump the last TurnState (slot budget, hypotheses, etc.)
      /budget            show resolved slot budget + context window
      /sessions          list all sessions (id, turn count, persona summary)
      /switch <id>       switch active session to a previous one
      /new               start a fresh session
      /delete <id>       delete a session (cascade)
      /pinned            list pinned memory facts
      /memory <query>    manual memory lookup (Tier 2 entity → Tier 4 RAG)
      /search <query>    manual web search via DuckDuckGo
      /hypotheses        last turn's LLM-3 hypotheses (after `infer` tag)
      /predictions       persisted LLM-2 forward predictions (≥0.6 confidence)
      /tools             tool calls executed on the last turn
      /decay             last turn's decay-engine counts (fresh→warm→cold→forgotten)
      /persona           current persona summary (LLM-2 maintained)
      /system <prompt>   replace the user system prompt (next turn onward)
      /cost              running cumulative cost estimate
      free text          chat with the agent
""")


def print_banner(provider: str, model: str, ctx_window: int) -> None:
    print("=" * 72)
    print("  Sherlock v0.5.0 — interactive test CLI")
    print(f"  provider={provider}  model={model}  ctx_window={ctx_window:,}")
    print(f"  storage={STORAGE_DIR.resolve()}")
    print(f"  web_search={'on' if ENABLE_WEB_SEARCH else 'off'}")
    print("=" * 72)
    print("Type /help for commands. Type /quit to exit.")
    print()
    # Git-commit foot-gun warning.
    keys_in_source = any(
        [
            OPENROUTER_API_KEY,
            OPENAI_API_KEY,
            ANTHROPIC_API_KEY,
            GOOGLE_API_KEY,
            XAI_API_KEY,
        ]
    )
    if keys_in_source:
        print("⚠️  An API key is hardcoded in test_sherlock.py — DO NOT commit this file.")
        print(
            "    Prefer setting the env var (e.g. ANTHROPIC_API_KEY=... python test_sherlock.py)."
        )
        print()


def show_inspect(agent) -> None:
    # Drain background work so the snapshot reflects completed companions.
    try:
        agent.drain()
    except Exception:
        pass
    state = agent.inspect_last_turn()
    if state is None:
        print("(no turns yet)")
        return
    print("--- Last TurnState ---")
    print(f"user           : {state.user_text!r}")
    print(f"tokens_used    : {state.tokens_used}")
    print(f"k_turn_turns   : {state.k_turn_turns_used}")
    print(f"k_turn_tokens  : {state.k_turn_tokens_used}")
    print(f"summary_run    : {state.summary_run}")
    print(f"decay_counts   : {state.decay_counts}")
    print(f"slot_budget    : {state.slot_budget}")
    print(f"hypotheses     : {len(state.hypotheses)} entries")
    for i, h in enumerate(state.hypotheses[:3], 1):
        print(f"  {i}. p={h.get('probability')} — {h.get('intent','')[:80]}")
    print(f"search_results : {len(state.search_results)} entries")


def show_predictions(agent) -> None:
    try:
        agent.drain()
    except Exception:
        pass
    conv_id = agent.conversation_id
    if conv_id is None:
        print("(no session yet)")
        return
    preds = agent._fetch_recent_llm2_predictions(conv_id, limit=10)
    if not preds:
        print("(no LLM-2 predictions persisted yet — they're emitted when `compact` runs)")
        return
    print(f"--- LLM-2 predictions ({len(preds)}) ---")
    for p in preds:
        print(f"  turn={p['turn_index']} conf={p['confidence']:.2f}  {p['direction']}")
        for ev in (p.get("evidence") or [])[:3]:
            print(f"      • {ev}")


def show_tools_history(agent) -> None:
    state = agent.inspect_last_turn()
    history = getattr(agent, "_tool_call_history", []) or []
    print(f"--- LLM-3 tool recommendations across {len(history)} turns ---")
    for h in history[-10:]:
        rec = h.get("tools_recommended") or []
        fresh = h.get("freshness_required") or []
        if not (rec or fresh):
            continue
        print(f"  turn={h['turn_index']}  recommended={rec}  freshness={fresh}")
    if state and state.search_results:
        print(f"\nLast turn web-search results ({len(state.search_results)}):")
        for r in state.search_results[:5]:
            print(f"  • {r.get('title','')[:60]} — {r.get('url','')}")


def show_pinned(agent) -> None:
    try:
        agent.drain()
    except Exception:
        pass
    conv_id = agent.conversation_id
    if conv_id is None:
        print("(no session yet)")
        return
    pinned = agent.memory.list(conversation_id=conv_id, pinned=True)
    if not pinned:
        print("(no pinned facts yet)")
        return
    print(f"--- Pinned facts ({len(pinned)}) ---")
    for p in pinned:
        tag = "persona" if "persona_summary" in (p.tags or "") else p.source.value
        print(f"  ({tag}) {p.content[:160]}")


def show_persona(agent) -> None:
    try:
        agent.drain()
    except Exception:
        pass
    conv_id = agent.conversation_id
    if conv_id is None:
        print("(no session yet)")
        return
    block = agent._format_persona_summary_block(conv_id)
    if not block:
        print("(no persona summary yet — emitted by LLM-2 when `compact` runs)")
    else:
        print(block)


def show_sessions(agent) -> None:
    sessions = agent.list_sessions()
    if not sessions:
        print("(no sessions)")
        return
    print(f"--- Sessions ({len(sessions)}) ---")
    for s in sessions:
        active = "  ← active" if agent.conversation_id == s.id else ""
        summary = (s.persona_summary or "(no persona)")[:80]
        print(f"  {s.id}  turns={s.turn_count}  created={s.created_at[:19]}{active}")
        print(f"    {summary}")


def manual_memory_lookup(agent, query: str) -> None:
    conv_id = agent.conversation_id
    if conv_id is None:
        print("(no session yet)")
        return
    from sherlock.tools.memory_tool import dispatch_memory

    out = dispatch_memory(
        f'lookup "{query}"',
        store=agent.memory,
        hybrid=agent._hybrid,
        storage=agent._storage,
        conversation_id=conv_id,
    )
    results = out.get("results") or []
    if not results:
        print("(no matches)")
        return
    print(f"--- {len(results)} matches for {query!r} ---")
    for r in results[:8]:
        tag = r.get("source", "")
        score = r.get("score")
        score_str = f" score={score:.3f}" if isinstance(score, float) else ""
        print(f"  ({tag}{score_str}) {r.get('content','')[:160]}")


def manual_web_search(query: str) -> None:
    if not ENABLE_WEB_SEARCH:
        print("(web search disabled — set ENABLE_WEB_SEARCH=True to enable)")
        return
    from sherlock.tools.web_search import DuckDuckGoSearch

    eng = DuckDuckGoSearch()
    results = eng.search(query, max_results=5)
    if not results:
        print("(no results)")
        return
    print(f"--- {len(results)} results for {query!r} ---")
    for r in results:
        print(
            f"  • {r.get('title','')[:60]}\n    {r.get('url','')}\n    {r.get('content','')[:200]}"
        )


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default=PROVIDER,
        choices=["openrouter", "openai", "anthropic", "google", "xai"],
        help=f"LLM provider (default: {PROVIDER})",
    )
    parser.add_argument("--model", default=None, help="Override MODEL constant")
    parser.add_argument(
        "--storage-dir",
        default=str(STORAGE_DIR),
        help=f"Storage directory (default: {STORAGE_DIR})",
    )
    args = parser.parse_args()

    provider = args.provider
    model = args.model or MODEL or DEFAULT_MODEL[provider]

    # 1. Resolve API key.
    api_key = resolve_api_key(provider)
    if not api_key:
        env_name = API_KEY_ENV[provider]
        print(f"error: no API key for provider={provider}.", file=sys.stderr)
        print(
            f"  Set the env var {env_name}=... or edit the constant in test_sherlock.py",
            file=sys.stderr,
        )
        return 2

    # 2. Resolve context window.
    ctx_window = resolve_ctx_window(provider, model)
    print(f"[resolved context_window={ctx_window:,} for {provider}/{model}]")

    # 3. Build chat callable.
    chat_callable = build_chat_callable(provider, api_key, model)

    # 4. Build Sherlock agent.
    from sherlock import Sherlock

    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    agent = Sherlock.with_callable(
        main_chat=chat_callable,
        system_prompt=SYSTEM_PROMPT,
        storage_dir=storage_dir,
        main_search_engine="duckduckgo" if ENABLE_WEB_SEARCH else None,
        inference_search_engine="duckduckgo" if ENABLE_WEB_SEARCH else None,
        embedding=EMBEDDING,
        embedding_model=EMBEDDING_MODEL,
        redact_secrets=REDACT_SECRETS,
        background=BACKGROUND,
    )
    patch_agent_budget(agent, ctx_window)

    print_banner(provider, model, ctx_window)
    cost = CostMeter(model)

    # 5. REPL.
    while True:
        try:
            try:
                line = input("you> ")
            except EOFError:
                print()
                break

            line = line.strip()
            if not line:
                continue

            # --- slash commands ---
            if line.startswith("/"):
                parts = line.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in {"/quit", "/exit"}:
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/inspect":
                    show_inspect(agent)
                elif cmd == "/budget":
                    sb = agent._slot_budget.as_dict() if agent._slot_budget else {}
                    print(f"context_window = {agent._ctx_window:,}")
                    print(f"slot_budget    = {sb}")
                elif cmd == "/sessions":
                    show_sessions(agent)
                elif cmd == "/switch":
                    if not arg:
                        print("usage: /switch <session-id>")
                    else:
                        try:
                            agent.switch_session(arg)
                            print(f"switched to {arg}")
                        except ValueError as e:
                            print(f"error: {e}")
                elif cmd == "/new":
                    sid = agent.new_session()
                    print(f"new session: {sid}")
                elif cmd == "/delete":
                    if not arg:
                        print("usage: /delete <session-id>")
                    else:
                        info = agent.delete_session(arg)
                        print(f"deleted {arg}: {info}")
                elif cmd == "/pinned":
                    show_pinned(agent)
                elif cmd == "/memory":
                    if not arg:
                        print("usage: /memory <query>")
                    else:
                        manual_memory_lookup(agent, arg)
                elif cmd == "/search":
                    if not arg:
                        print("usage: /search <query>")
                    else:
                        manual_web_search(arg)
                elif cmd == "/hypotheses":
                    state = agent.inspect_last_turn()
                    if not state or not state.hypotheses:
                        print(
                            "(no hypotheses on last turn — LLM-3 only runs when `infer` tag is emitted)"
                        )
                    else:
                        print(f"--- {len(state.hypotheses)} hypotheses ---")
                        for i, h in enumerate(state.hypotheses, 1):
                            print(
                                f"  {i}. p={h.get('probability')} type={h.get('reasoning_type','')}"
                            )
                            print(f"     intent : {h.get('intent','')}")
                            for ev in (h.get("evidence") or [])[:3]:
                                print(f"     evidence: {ev}")
                elif cmd == "/predictions":
                    show_predictions(agent)
                elif cmd == "/tools":
                    show_tools_history(agent)
                elif cmd == "/decay":
                    state = agent.inspect_last_turn()
                    if state:
                        print(f"last turn decay_counts: {state.decay_counts}")
                    else:
                        print("(no turns yet)")
                elif cmd == "/persona":
                    show_persona(agent)
                elif cmd == "/system":
                    if not arg:
                        print("usage: /system <new system prompt>")
                    else:
                        agent._user_system_prompt = arg
                        # Re-compose with the extension.
                        ext = agent._sherlock_extension or ""
                        agent._system_prompt = (
                            f"{arg.rstrip()}\n\n{ext.lstrip()}" if ext.strip() else arg
                        )
                        print("system prompt updated (effective next turn)")
                elif cmd == "/cost":
                    print(f"cost: {cost.report()}")
                else:
                    print(f"unknown command: {cmd}.  Try /help.")
                continue

            # --- free-text chat ---

            # Big-paste guard.
            if MAX_USER_INPUT_TOKENS is not None:
                from sherlock.budget import count_tokens

                n = count_tokens(line)
                if n > MAX_USER_INPUT_TOKENS:
                    print(
                        f"refusing: input is ~{n} tokens, MAX_USER_INPUT_TOKENS={MAX_USER_INPUT_TOKENS}. "
                        f"Edit the constant in test_sherlock.py if you really want to send it."
                    )
                    continue

            t0 = time.time()
            try:
                reply = agent.chat(line)
            except KeyboardInterrupt:
                print("\n(interrupted — last user turn was persisted but no reply was generated)")
                continue
            dt = time.time() - t0
            print(f"\nagent> {reply}\n")
            # Cost accounting from the last turn's TurnState.
            state = agent.inspect_last_turn()
            if state and state.response and state.response.usage:
                cost.record(
                    state.response.usage.prompt_tokens, state.response.usage.completion_tokens
                )
                print(
                    f"[turn took {dt:.1f}s | "
                    f"k_turn={state.k_turn_turns_used}t/{state.k_turn_tokens_used}tok | "
                    f"in={state.response.usage.prompt_tokens} out={state.response.usage.completion_tokens} | "
                    f"cumulative {cost.report()}]"
                )

        except KeyboardInterrupt:
            # Top-level Ctrl+C at the input prompt → graceful exit.
            print()
            break
        except Exception:
            traceback.print_exc()
            print("[continuing — REPL kept alive]")

    # Let any in-flight background companion work finish before summarising.
    try:
        agent.drain()
    except Exception:
        pass
    print("\n=== session summary ===")
    print(f"sessions on disk : {len(agent.list_sessions())}")
    print(f"final cost       : {cost.report()}")
    print(f"storage          : {storage_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
