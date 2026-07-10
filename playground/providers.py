"""Provider glue for the playground: live model listing + per-role chat callables.

Supported providers (all through litellm, so Sherlock itself stays BYO-LLM):

  gemini     Google AI Studio key            -> litellm "gemini/<model>"
  openai     OpenAI API key                  -> litellm "openai/<model>"
  anthropic  Anthropic API key               -> litellm "anthropic/<model>"
  local      any OpenAI-compatible server    -> litellm "openai/<model>" + api_base
             (Ollama, LM Studio, vLLM, llama.cpp server, ...)

  Open-source-model aggregators (OpenAI-compatible, descriptor-driven via
  ``OPENAI_COMPAT`` below — adding the next one is a one-row dict entry):
  deepinfra  DeepInfra key                   -> litellm "deepinfra/<org/model>"
  together   Together AI key                 -> litellm "together_ai/<org/model>"
  openrouter OpenRouter key                  -> litellm "openrouter/<org/model>"

API keys live ONLY in the server-side Session — they are sent once from the
browser to /api/models and /api/session and never echoed back.
"""

from __future__ import annotations

import re
import time

import httpx

from sherlock.providers.base import ChatMessage, ChatResponse, TokenUsage

_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_ROLE_ACTOR = {"main": "llm1", "summary": "llm2", "inference": "llm3", "viz": "llm4"}

# OpenAI /v1/models lists every modality; keep only chat-capable families.
_OPENAI_CHAT_RE = re.compile(r"^(gpt-[45o]|gpt-oss|o[134](-|$)|chatgpt-)")
_OPENAI_NON_CHAT_RE = re.compile(
    r"(embed|whisper|tts|audio|realtime|image|dall-e|moderation|transcribe|search|davinci|babbage|instruct)"
)

# Open-source-model aggregators. All three are the SAME OpenAI-compatible API
# behind three base URLs — the only differences (base URL, litellm route prefix,
# whether /models needs the key, and the /models JSON shape) are DATA, not logic.
# A descriptor table keeps adding the next aggregator (Fireworks, Novita, ...) a
# one-row change instead of another if-ladder. Verified live 2026-06-19.
OPENAI_COMPAT = {
    "deepinfra": {
        "label": "DeepInfra",
        "litellm_prefix": "deepinfra/",
        "models_url": "https://api.deepinfra.com/v1/openai/models",
        "models_need_key": False,  # public; a NON-EMPTY invalid key 401s → never send it to list
        "list_shape": "data",  # {"data": [{id, metadata:{tags, context_length}}]}
        "chat_filter": "deepinfra",  # keep metadata.tags ∋ "chat"
        "extra_headers": {},
    },
    "together": {
        "label": "Together AI",
        "litellm_prefix": "together_ai/",
        "models_url": "https://api.together.ai/v1/models",
        "models_need_key": True,
        "list_shape": "bare_array",  # TOP-LEVEL [ {...} ] — NO {"data": ...} envelope
        "chat_filter": "together",  # keep type == "chat"
        "extra_headers": {},
    },
    "openrouter": {
        "label": "OpenRouter",
        "litellm_prefix": "openrouter/",
        "models_url": "https://openrouter.ai/api/v1/models",
        "models_need_key": False,  # public list
        "list_shape": "data",  # {"data": [{id, architecture:{output_modalities}}]}
        "chat_filter": "openrouter",  # keep text-output models
        # X-Title shows up in the user's OpenRouter dashboard; purely cosmetic,
        # never required, no fake referrer URL shipped.
        "extra_headers": {"X-Title": "Sherlock"},
    },
}


def _normalize_local_base(base_url: str) -> str:
    """'http://localhost:11434' -> 'http://localhost:11434/v1' (Ollama/LM Studio
    both serve the OpenAI-compatible surface under /v1)."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("local provider needs a base URL (e.g. http://localhost:11434/v1)")
    if not base.startswith("http"):
        base = "http://" + base
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def list_models(provider: str, api_key: str = "", base_url: str = "") -> list[dict]:
    """Live model list for one provider: ``[{id, display, ...}]``, newest-ish
    first. Raises on HTTP/auth failure so the caller can surface the error."""
    provider = (provider or "gemini").lower()
    if provider in OPENAI_COMPAT:
        return _list_openai_compat(provider, api_key)
    if provider == "gemini":
        return _list_gemini(api_key)
    if provider == "openai":
        return _list_openai(api_key)
    if provider == "anthropic":
        return _list_anthropic(api_key)
    if provider == "local":
        return _list_local(base_url, api_key)
    raise ValueError(f"unknown provider: {provider}")


def _list_gemini(api_key: str) -> list[dict]:
    r = httpx.get(_GEMINI_MODELS_URL, params={"key": api_key, "pageSize": 1000}, timeout=20.0)
    r.raise_for_status()
    out: list[dict] = []
    for m in r.json().get("models", []):
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        mid = (m.get("name") or "").removeprefix("models/")
        if not mid:
            continue
        out.append(
            {
                "id": mid,
                "display": m.get("displayName") or mid,
                "input_limit": m.get("inputTokenLimit"),
                "output_limit": m.get("outputTokenLimit"),
            }
        )
    out.sort(key=lambda x: x["id"], reverse=True)
    return out


def _list_openai(api_key: str) -> list[dict]:
    r = httpx.get(_OPENAI_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20.0)
    r.raise_for_status()
    out: list[dict] = []
    for m in r.json().get("data", []):
        mid = m.get("id") or ""
        if not _OPENAI_CHAT_RE.search(mid) or _OPENAI_NON_CHAT_RE.search(mid):
            continue
        out.append({"id": mid, "display": mid, "created": m.get("created") or 0})
    out.sort(key=lambda x: (-(x.get("created") or 0), x["id"]))
    return out


def _list_anthropic(api_key: str) -> list[dict]:
    r = httpx.get(
        _ANTHROPIC_MODELS_URL,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        params={"limit": 100},
        timeout=20.0,
    )
    r.raise_for_status()
    out = [
        {"id": m.get("id"), "display": m.get("display_name") or m.get("id")}
        for m in r.json().get("data", [])
        if m.get("id")
    ]
    return out  # API already returns newest first


def _list_local(base_url: str, api_key: str = "") -> list[dict]:
    base = _normalize_local_base(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.get(f"{base}/models", headers=headers, timeout=10.0)
    r.raise_for_status()
    data = r.json().get("data", [])
    out = [{"id": m.get("id"), "display": m.get("id")} for m in data if m.get("id")]
    out.sort(key=lambda x: x["id"])
    return out


def _chat_models_deepinfra(items: list[dict]) -> list[dict]:
    """DeepInfra /models mixes chat with embed/image/tts/stt — keep tags ∋ 'chat'."""
    out = []
    for m in items:
        meta = m.get("metadata") or {}
        if "chat" not in (meta.get("tags") or []):
            continue
        mid = m.get("id") or ""
        if mid:
            out.append({"id": mid, "display": mid, "input_limit": meta.get("context_length")})
    out.sort(key=lambda x: x["id"])
    return out


def _chat_models_together(items: list[dict]) -> list[dict]:
    """Together model objects carry a ``type`` enum (chat|language|code|image|
    embedding|...) — keep only chat."""
    out = []
    for m in items:
        if (m.get("type") or "") != "chat":
            continue
        mid = m.get("id") or ""
        if mid:
            out.append(
                {
                    "id": mid,
                    "display": m.get("display_name") or mid,
                    "created": m.get("created") or 0,
                }
            )
    out.sort(key=lambda x: (-(x.get("created") or 0), x["id"]))
    return out


def _chat_models_openrouter(items: list[dict]) -> list[dict]:
    """OpenRouter lists a few non-text-output models — drop anything whose
    architecture can't emit text."""
    out = []
    for m in items:
        outs = (m.get("architecture") or {}).get("output_modalities") or []
        if outs and "text" not in outs:
            continue
        mid = m.get("id") or ""
        if mid:
            out.append(
                {"id": mid, "display": m.get("name") or mid, "created": m.get("created") or 0}
            )
    out.sort(key=lambda x: (-(x.get("created") or 0), x["id"]))
    return out


_OSS_CHAT_FILTERS = {
    "deepinfra": _chat_models_deepinfra,
    "together": _chat_models_together,
    "openrouter": _chat_models_openrouter,
}


def _list_openai_compat(name: str, api_key: str = "") -> list[dict]:
    """Generic model lister for the OpenAI-compatible aggregators in
    ``OPENAI_COMPAT``. Normalizes the three /models response shapes into one
    ``[{id, display, ...}]`` list and applies the per-aggregator chat filter.
    Auth is sent ONLY when the endpoint requires it (DeepInfra/OpenRouter list
    publicly, and DeepInfra 401s on a non-empty invalid key — keyless listing is
    the robust path)."""
    d = OPENAI_COMPAT[name]
    headers = dict(d.get("extra_headers") or {})
    if d["models_need_key"] and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = httpx.get(d["models_url"], headers=headers, timeout=20.0)
    r.raise_for_status()
    body = r.json()
    if d["list_shape"] == "bare_array":
        items = body if isinstance(body, list) else body.get("data", [])
    else:  # "data": {"data": [...]} (with or without an "object":"list" wrapper)
        items = body.get("data", []) if isinstance(body, dict) else (body or [])
    return _OSS_CHAT_FILTERS[d["chat_filter"]](items)


def resolve_model_spec(spec, providers: dict) -> tuple[str, dict]:
    """Turn a role's model spec into (litellm_model_id, extra litellm kwargs).

    ``spec`` is ``{"provider": ..., "model": ...}`` from the UI, or a bare
    string (legacy sessions / tests) which is treated as a Gemini model id.
    ``providers`` is the session's credential map {provider: {api_key, base_url}}.
    """
    if isinstance(spec, str):
        provider, model = "gemini", spec
    else:
        provider = (spec or {}).get("provider", "gemini")
        model = (spec or {}).get("model", "")
    creds = (providers or {}).get(provider, {})
    key = creds.get("api_key", "")
    if provider in OPENAI_COMPAT:
        d = OPENAI_COMPAT[provider]
        # litellm knows each prefix's base URL + cost map natively; the key is
        # passed explicitly so no env-var mirroring is needed on this path.
        extra = {"api_key": key}
        if d.get("extra_headers"):
            extra["extra_headers"] = dict(d["extra_headers"])
        return f"{d['litellm_prefix']}{model}", extra
    if provider == "gemini":
        return f"gemini/{model}", {"api_key": key}
    if provider == "openai":
        return f"openai/{model}", {"api_key": key}
    if provider == "anthropic":
        return f"anthropic/{model}", {"api_key": key}
    if provider == "local":
        base = _normalize_local_base(creds.get("base_url", ""))
        # litellm requires SOME api_key for the openai route; local servers ignore it.
        return f"openai/{model}", {"api_base": base, "api_key": key or "local"}
    raise ValueError(f"unknown provider: {provider}")


def _call_litellm(model: str, messages: list[dict], **extra):
    """Single litellm entry point for every playground call (role callables AND
    the A/B baseline). Module-level so tests can monkeypatch it."""
    import litellm

    litellm.suppress_debug_info = True
    return litellm.completion(model=model, messages=messages, **extra)


def _call_litellm_image(model: str, prompt: str, **extra):
    """litellm image-generation entry point (v1.12 Stage V3). Module-level so
    tests can monkeypatch it like _call_litellm."""
    import litellm

    litellm.suppress_debug_info = True
    return litellm.image_generation(prompt=prompt, model=model, **extra)


def _viz_flatten_system(messages: list[dict]) -> list[dict]:
    """Merge system content into the first user turn — image-capable omni
    models (Gemini omni previews) 400 on developer/system instructions."""
    sys_txt = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    if not sys_txt:
        return rest
    if rest and rest[0].get("role") == "user":
        return [{"role": "user", "content": sys_txt + "\n\n" + rest[0].get("content", "")}] + rest[
            1:
        ]
    return [{"role": "user", "content": sys_txt}] + rest


def _call_litellm_viz(model_id: str, send_messages: list[dict], **extra):
    """LLM-4 call with OMNI adaptation (v1.12 fix, rev2): some image-capable
    models reject the standard shape with a 400. We do NOT guess error wording —
    ANY 400 gets the progressive ladder: ① normal shape → ② system merged into
    the user turn → ③ merged + [image, text] modalities. Non-400 errors
    (429/5xx/auth) re-raise immediately at every step, so a rate limit is never
    retried into a bigger burn and the original error is never masked."""

    def _is_400(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        try:
            return status is not None and int(status) == 400
        except Exception:
            return False

    try:
        return _call_litellm(model_id, send_messages, **extra)
    except Exception as exc:
        if not _is_400(exc):
            raise
        first_err = exc
    flat = _viz_flatten_system(send_messages)
    try:
        return _call_litellm(model_id, flat, **extra)
    except Exception as exc:
        if not _is_400(exc):
            raise
    try:
        return _call_litellm(model_id, flat, modalities=["image", "text"], **extra)
    except Exception as exc:
        # every rung 400'd — surface the ORIGINAL shape error (most diagnostic)
        raise first_err if _is_400(exc) else exc


def _image_from_completion_response(resp):
    """Extract a data:/http image from a chat-completion response (litellm
    normalises omni image output to message.images[{image_url:{url}}]; some
    routes put a data URI straight into content). Returns the URI/URL string or
    None."""
    msg = resp.choices[0].message if getattr(resp, "choices", None) else None
    if msg is None:
        return None
    imgs = getattr(msg, "images", None) or (msg.get("images") if isinstance(msg, dict) else None)
    for first in imgs or []:
        u = first.get("image_url") if isinstance(first, dict) else getattr(first, "image_url", None)
        if isinstance(u, dict):
            u = u.get("url")
        elif u is not None and not isinstance(u, str):
            u = getattr(u, "url", None)
        if isinstance(u, str) and (u.startswith("data:image") or u.startswith("http")):
            return u
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if isinstance(content, str) and content.strip().startswith("data:image"):
        return content.strip()
    return None


def make_omni_image_callable(session, role: str = "viz"):  # noqa: ANN001
    """v1.12 omni: text→image via the CHAT model itself (completion +
    [image, text] modalities) — no dedicated image model needed. Used for
    ``image:`` markers when settings.image_model is empty; a text-only model
    raises here and the library falls back to drawing the visual as HTML/SVG."""

    def _generate(prompt: str):
        spec = session.models.get(role) or session.models.get("main")
        model_id, extra = resolve_model_spec(spec, getattr(session, "providers", {}))
        t0 = time.time()
        resp = _call_litellm(
            model_id,
            [{"role": "user", "content": "Generate ONE image (no text reply): " + prompt}],
            modalities=["image", "text"],
            **extra,
        )
        session.emit(
            {
                "type": "llm.call",
                "actor": "llm4",
                "turn": session.turn,
                "data": {
                    "role": "viz_image_omni",
                    "model": model_id,
                    "latency_ms": int((time.time() - t0) * 1000),
                },
            }
        )
        out = _image_from_completion_response(resp)
        if not out:
            raise ValueError("model returned no image (text-only model?)")
        return out

    return _generate


def make_image_callable(session, spec):  # noqa: ANN001
    """v1.12 Stage V3: a text→image callable for ``image:`` viz markers.

    ``spec`` is the same ``{"provider", "model"}`` shape as the role model specs
    (or a bare litellm model-id string, treated per resolve_model_spec).
    Credentials resolve from the session's SERVER-SIDE provider creds at CALL
    time — a key edited later still applies, and nothing key-shaped ever reaches
    the browser. Returns ``{"b64": ..., "url": ...}`` for the library adapter
    (sherlock.agent._viz_image_generate) to normalise."""

    def _generate(prompt: str) -> dict:
        model, extra = resolve_model_spec(spec, session.providers)
        t0 = time.time()
        resp = _call_litellm_image(model, prompt, **extra)
        data = getattr(resp, "data", None) or []
        first = data[0] if data else None
        b64 = getattr(first, "b64_json", None) or (
            first.get("b64_json") if isinstance(first, dict) else None
        )
        url = getattr(first, "url", None) or (first.get("url") if isinstance(first, dict) else None)
        session.emit(
            {
                "type": "llm.call",
                "actor": "llm4",
                "turn": session.turn,
                "data": {
                    "role": "viz_image",
                    "model": model,
                    "latency_ms": int((time.time() - t0) * 1000),
                },
            }
        )
        return {"b64": b64, "url": url}

    return _generate


def _stopped(session) -> bool:
    """True when the user pressed Stop for this session's current turn. Safe if
    the agent / stop event isn't wired yet (returns False)."""
    ev = getattr(getattr(session, "agent", None), "_stop_event", None)
    return bool(ev is not None and ev.is_set())


def _call_litellm_stream(model, messages, on_delta, should_stop, on_reasoning=None, **extra):
    """Streaming variant of _call_litellm for the USER-VISIBLE main reply.

    Calls ``on_delta(chunk_text)`` per answer token and ``on_reasoning(piece)``
    per reasoning/"thinking" token (litellm normalizes provider reasoning into
    ``delta.reasoning_content`` — DeepSeek-R1, GLM, o-series, Gemini/Anthropic
    thinking). Breaks early if ``should_stop()`` flips. Returns a
    ModelResponse-shaped object (full text + usage) rebuilt from the chunks, so
    the caller's text/usage extraction is IDENTICAL to the non-streaming path.
    The initial ``completion(stream=True)`` is intentionally outside the try
    block: if it fails before any token, the exception propagates so the caller
    can cleanly fall back to non-streaming; once tokens have streamed, a
    mid-stream error keeps whatever arrived."""
    import litellm
    from types import SimpleNamespace

    litellm.suppress_debug_info = True
    stream = litellm.completion(model=model, messages=messages, stream=True, **extra)
    chunks, parts = [], []
    try:
        for chunk in stream:
            chunks.append(chunk)
            try:
                _delta = chunk.choices[0].delta
            except Exception:
                _delta = None
            piece = (getattr(_delta, "content", None) or "") if _delta is not None else ""
            if piece:
                parts.append(piece)
                on_delta(piece)
            if on_reasoning is not None and _delta is not None:
                rc = getattr(_delta, "reasoning_content", None) or ""
                if rc:
                    on_reasoning(rc)
            if should_stop():
                break
    except Exception:
        pass  # mid-stream error → keep the text we already streamed
    try:
        built = litellm.stream_chunk_builder(chunks, messages=messages)
        if (
            built
            and getattr(built, "choices", None)
            and built.choices[0].message.content is not None
        ):
            return built
    except Exception:
        pass
    # Fallback shape so the caller reads .choices[0].message.content / .usage uniformly.
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="".join(parts)))],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def _spec_provider(spec) -> str:
    """Provider name for a role's model spec (legacy bare string = gemini)."""
    if isinstance(spec, str):
        return "gemini"
    return ((spec or {}).get("provider") or "gemini").lower()


def _apply_cache_hints(messages: list[dict], cache_hints, provider: str) -> list[dict]:
    """v1.3: turn sherlock's CallableProvider ``cache_hints`` into Anthropic
    prompt-cache blocks. Hints are ``{"stable_prefix_chars": {msg_idx: chars}}``;
    for the anthropic provider we REUSE sherlock's converter (ChatMessage +
    LiteLLMProvider._to_litellm_messages) so each hinted message becomes
    OpenAI-format content blocks with ``cache_control`` on the stable prefix.
    Other providers (gemini/openai/local) cache implicitly server-side, so the
    hints are ignored and the payload stays byte-identical."""
    prefixes = (cache_hints or {}).get("stable_prefix_chars") or {}
    if not prefixes or provider != "anthropic":
        return messages
    try:  # deferred, mirroring _call_litellm — litellm import is heavy/optional
        from sherlock.providers.litellm_provider import LiteLLMProvider
    except Exception:
        return messages
    converted: list[ChatMessage] = []
    for i, m in enumerate(messages):
        content = m.get("content")
        split = prefixes.get(i, prefixes.get(str(i))) if isinstance(content, str) else None
        converted.append(
            ChatMessage(
                role=m.get("role", "user"),
                content=content if isinstance(content, str) else (content or ""),
                cache_stable_prefix_chars=int(split) if split else None,
            )
        )
    return LiteLLMProvider._to_litellm_messages(converted)


def _baseline_search_block(session, message: str) -> str:
    """One NAIVE search pass with the raw user message — the typical
    'LLM + web search' wiring people actually use as a baseline. Same engine
    the Sherlock side uses, so the A/B isolates CURATION, not tool access."""
    engine_name = (session.settings or {}).get("search_engine", "duckduckgo")
    if engine_name in (None, "", "off", "none"):
        return ""
    try:
        from sherlock.tools.web_search import create_search

        if session._baseline_engine is None:
            session._baseline_engine = create_search(
                engine_name, api_key=(session.settings or {}).get("search_api_key") or None
            )
        results = session._baseline_engine.search(message[:300], max_results=5) or []
    except Exception:
        return ""
    lines = []
    for r in results[:5]:
        if not isinstance(r, dict) or r.get("error"):
            continue
        snippet = (r.get("content") or r.get("snippet") or "")[:300]
        lines.append(f"- {r.get('title', '')} — {r.get('url', '')}: {snippet}")
    if not lines:
        return ""
    return "Web search results for the user's message:\n" + "\n".join(lines)


def baseline_chat(session, message: str, *, use_search: bool = True) -> dict:
    """The fair single-LLM baseline for A/B mode: the MAIN role's model called
    directly through litellm — full raw history, the user's plain system prompt
    plus today's date, and (by default) ONE naive web-search pass with the raw
    user message. No Sherlock curation/companions/memory. Returns
    ``{"text", "latency_ms", "prompt_tokens", "completion_tokens", "error",
    "searched"}``.
    """
    from datetime import datetime

    t0 = time.time()
    text, pt, ct, err = "", 0, 0, None
    # Today's date is one trivial line any wrapper would add — withholding it
    # would gift Sherlock an unearned win on date questions.
    today = datetime.now().astimezone().strftime("%Y-%m-%d (%A)")
    sys_prompt = f"{session.system_prompt}\n(Today is {today}.)"
    search_block = _baseline_search_block(session, message) if use_search else ""
    user_content = f"{message}\n\n{search_block}" if search_block else message
    messages = (
        [{"role": "system", "content": sys_prompt}]
        + list(session.baseline_history)
        + [{"role": "user", "content": user_content}]
    )
    try:
        model_id, extra = resolve_model_spec(
            session.models.get("main"), getattr(session, "providers", {})
        )
        resp = _call_litellm(model_id, messages, **extra)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        text = ""
    latency_ms = int((time.time() - t0) * 1000)
    if err is None:
        # history keeps the PLAIN user message (search blocks are per-turn aids,
        # not conversation content — mirroring how Sherlock's transcript works)
        session.baseline_history.append({"role": "user", "content": message})
        session.baseline_history.append({"role": "assistant", "content": text})
    session.baseline_tokens["in"] += pt
    session.baseline_tokens["out"] += ct
    return {
        "text": text,
        "latency_ms": latency_ms,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "error": err,
        "searched": bool(search_block),
    }


# Markers that only appear in Sherlock's INTERNAL deep-research prompts
# (round Q&A, planning, review, synthesis) — never in a real user turn.
_INTERNAL_RESEARCH_MARKERS = (
    "Answer these meta-questions",
    "RESEARCH DOCUMENTS:",
    "META-COGNITION QUESTIONS",
    "MULTILINGUAL web-search sweep",
    "reviewing ONE round",
)


def _is_internal_research_prompt(messages: list[dict]) -> bool:
    """True when the last user message is an internal deep-research call
    (round Q&A / synthesis / planning) rather than a real user turn."""
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    content = (last or {}).get("content") or ""
    return any(marker in content for marker in _INTERNAL_RESEARCH_MARKERS)


def make_role_callable(role: str, session, emit):
    """Build a Sherlock chat callable for ``role`` ∈ {main, summary, inference, viz}.

    Only ``main`` streams; every other role (including the v1.12 Stage B1 ``viz``
    role — LLM-4, the inline visualizer) takes the non-streaming branch below.

    Reads the CURRENT model selection from ``session.models`` each call (so a
    mid-session dropdown change takes effect next turn). Emits an ``llm.call``
    event with the exact prompt/response/tokens/latency, and returns a
    ``ChatResponse`` with real token usage so Sherlock's budget telemetry is
    accurate.

    The ``cache_hints`` kwarg makes sherlock's CallableProvider pass its
    prompt-cache hints (it detects the parameter via signature inspection);
    for anthropic-backed roles they become real ``cache_control`` blocks.
    """
    actor = _ROLE_ACTOR.get(role, role)

    def _call(messages: list[dict], cache_hints=None):
        spec = session.models.get(role) or session.models.get("main")
        t0 = time.time()
        text, pt, ct, tt, err = "", 0, 0, 0, None
        cache_read = 0
        model_id = "?"
        try:
            model_id, extra = resolve_model_spec(spec, getattr(session, "providers", {}))
            send_messages = _apply_cache_hints(messages, cache_hints, _spec_provider(spec))
            # Stream ONLY the user-visible main reply — companions (LLM-2/LLM-3)
            # and internal deep-research prompts stay non-streaming (background,
            # not shown live). Each main reply token is pushed to the browser as
            # an `llm.delta` event; reasoning/"thinking" tokens go out as a
            # separate `llm.reasoning_delta`. The full text is still returned to
            # the core (its `f(messages)->str` contract is unchanged).
            reasoning_streamed = []

            def _emit_reasoning(piece: str) -> None:
                reasoning_streamed.append(piece)
                emit(
                    {
                        "type": "llm.reasoning_delta",
                        "actor": actor,
                        "turn": session.turn,
                        "data": {"chunk": piece},
                    }
                )

            if role == "main" and not _is_internal_research_prompt(messages):
                answer_streamed = []

                def _on_delta(piece: str) -> None:
                    answer_streamed.append(piece)
                    emit(
                        {
                            "type": "llm.delta",
                            "actor": actor,
                            "turn": session.turn,
                            "data": {"chunk": piece},
                        }
                    )

                try:
                    resp = _call_litellm_stream(
                        model_id,
                        send_messages,
                        _on_delta,
                        lambda: _stopped(session),
                        on_reasoning=_emit_reasoning,
                        **extra,
                    )
                except Exception:
                    # Provider/route can't stream → fall back (no deltas emitted yet).
                    resp = _call_litellm(model_id, send_messages, **extra)
                # If the stream produced no visible tokens AND no text (an empty
                # or mid-error stream on some route), fall back to non-streaming so
                # the reply is never blank — unless the user explicitly stopped.
                _txt = (
                    (resp.choices[0].message.content or "")
                    if getattr(resp, "choices", None)
                    else ""
                )
                if not answer_streamed and not _txt and not _stopped(session):
                    resp = _call_litellm(model_id, send_messages, **extra)
            else:
                # v1.12 omni fix: LLM-4 must work with image-capable omni models
                # that reject system instructions / demand image modalities.
                if role == "viz":
                    resp = _call_litellm_viz(model_id, send_messages, **extra)
                else:
                    resp = _call_litellm(model_id, send_messages, **extra)
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            # Reasoning models that expose thinking only on the final message (or
            # the non-streaming fallback) — emit it once if nothing streamed live.
            if role == "main" and not reasoning_streamed and resp.choices:
                _final_reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                if _final_reasoning:
                    _emit_reasoning(_final_reasoning)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                tt = getattr(usage, "total_tokens", 0) or (pt + ct)
                cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                if not cache_read:
                    details = (
                        usage.get("prompt_tokens_details")
                        if isinstance(usage, dict)
                        else getattr(usage, "prompt_tokens_details", None)
                    )
                    if isinstance(details, dict):
                        cache_read = int(details.get("cached_tokens") or 0)
                    else:
                        cache_read = int(getattr(details, "cached_tokens", 0) or 0)
        except Exception as exc:  # surface as a wrapper-error (Sherlock skips persisting it)
            err = f"{type(exc).__name__}: {exc}"
            text = f"[wrapper-error: {err}]"
        latency_ms = int((time.time() - t0) * 1000)
        # Force-reasoning: a vanilla model rarely emits the
        # <<sherlock-companions: ...>> control tag on its own, so the inference
        # (LLM-3) and compaction (LLM-2) panels stay empty and the user can't
        # see the system "think". When the toggle is on, append the tag to the
        # MAIN reply so both companions fire every turn. Sherlock strips the tag
        # before the user sees it; we emit the ORIGINAL text for display.
        # Internal deep-research calls (round Q&A / synthesis) also go through
        # this callable — never tag those, or the tag leaks into the persisted
        # final research answer.
        returned_text = text
        if (
            role == "main"
            and not err
            and (session.settings or {}).get("force_companions")
            and "<<sherlock-companions" not in text
            and "<<sherlock-tool" not in text
            and not _is_internal_research_prompt(messages)
        ):
            returned_text = text.rstrip() + "\n<<sherlock-companions: compact, infer>>"
        sys_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        emit(
            {
                "type": "llm.call",
                "actor": actor,
                "turn": session.turn,
                "data": {
                    "role": role,
                    "model": model_id,
                    "system_prompt": sys_prompt,
                    "messages": messages,
                    "response_text": text,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                    "cache_read_tokens": cache_read,
                    "latency_ms": latency_ms,
                    "error": err,
                },
            }
        )
        if err and role == "viz":
            # v1.12 omni fix rev2: a viz error must FAIL FAST — returning the
            # wrapper-error string would go through the static lint as if it
            # were HTML, burning self-review + repair rounds on garbage. The
            # llm.call event above already carries the full error; also print
            # it so the uvicorn log keeps a self-serve diagnosis trail.
            print(f"[viz-error] {model_id}: {err}", flush=True)
            raise RuntimeError(err)
        return ChatResponse(
            text=returned_text,
            model=model_id,
            usage=TokenUsage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                cache_read_tokens=cache_read,
            ),
        )

    return _call
