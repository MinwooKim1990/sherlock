"""Multi-provider web search + page fetch (v0.3.0).

Engines supported (all with the same interface):
- DuckDuckGoSearch (default — no API key needed)
- TavilySearch (key)
- BraveSearch (key)
- ValyuSearch (key)

API keys can be passed directly to the constructor, looked up from an
environment variable, or — via the YAML config — both.

Page fetch is bundled into the same engine instances. Default returns
trafilatura-extracted readable text; pass `raw=True` for the original
HTML.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from sherlock.config import SearchConfig


_DEFAULT_FETCH_TIMEOUT = 15.0
_RAW_HTML_CAP = 200_000
_TEXT_EXTRACT_CAP = 50_000


# v0.8: light, conservative query cleaning so search gets keywords, not
# sentences. The LLM planner does the primary keyword extraction; this is a
# safety-net strip of punctuation + obvious function words. Deliberately
# minimal so it never guts a real CJK keyword.
_EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "with",
    "at",
    "by",
    "from",
    "as",
    "that",
    "this",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "do",
    "does",
    "did",
    "please",
    "tell",
    "me",
    "about",
    "i",
    "my",
    "we",
    "our",
}
# Korean particles (조사) — MULTI-CHAR ONLY, stripped from a trailing position
# when the stem that remains is still ≥2 chars. Single-char particles
# (이/가/은/는/도/...) are deliberately NOT stripped: they corrupt real nouns
# (하와이→하와, 제주도→제주, 고양이→고양) and search engines handle an attached
# single-char particle far better than a truncated stem.
_KO_JOSA = (
    "으로서",
    "으로써",
    "에서는",
    "에게서",
    "이라고",
    "라고는",
    "에서",
    "에게",
    "한테",
    "부터",
    "까지",
    "으로",
    "이라",
    "라는",
    "처럼",
    "보다",
    "마다",
    "조차",
    "마저",
    "밖에",
    "이나",
    "거나",
    "든지",
)
_HANGUL_RE = re.compile(r"[가-힣]")
# Double-quoted phrases ("..." or “...”) survive cleaning verbatim — they are
# deliberate exact-match operators, not noise.
_QUOTED_RE = re.compile(r"\"([^\"]{2,})\"|“([^”]{2,})”")


def _strip_ko_josa(token: str) -> str:
    """Strip one trailing multi-char Korean particle if the token is Hangul and
    the remaining stem stays ≥2 chars. Conservative — never empties a word."""
    if not _HANGUL_RE.search(token):
        return token
    for josa in _KO_JOSA:
        if token.endswith(josa) and len(token) - len(josa) >= 2:
            return token[: -len(josa)]
    return token


def clean_query(query: str, lang: str | None = None) -> str:
    """Normalize a search query into keywords: drop punctuation, English
    stopwords, and trailing Korean particles. Quoted phrases, version numbers
    (3.12), and symbol-bearing terms (C++, C#, node.js) survive intact.
    Best-effort + non-destructive on already-clean keyword strings; never
    returns empty (falls back to the punctuation-stripped original)."""
    if not query:
        return ""
    phrases: list[str] = []

    def _hold(m: re.Match) -> str:
        phrases.append((m.group(1) or m.group(2)).strip())
        return f" __SHPH{len(phrases) - 1}__ "

    held = _QUOTED_RE.sub(_hold, query)
    # Replace punctuation/symbols with spaces; keep word chars (incl. CJK),
    # hyphen, and the token-internal symbols . + # (versions, C++, C#).
    cleaned = re.sub(r"[^\w\s.+#-]", " ", held, flags=re.UNICODE)
    tokens = cleaned.split()
    out: list[str] = []
    for tok in tokens:
        ph = re.fullmatch(r"__SHPH(\d+)__", tok)
        if ph:
            out.append('"' + phrases[int(ph.group(1))] + '"')
            continue
        # Sentence-final dots go; interior dots (3.12, node.js) stay.
        tok = tok.strip(".")
        if not tok or not re.search(r"\w", tok):
            continue
        low = tok.lower()
        # English stopword (latin only — never drop a CJK token this way).
        if low.isascii() and low.isalpha() and low in _EN_STOPWORDS:
            continue
        tok = _strip_ko_josa(tok)
        if tok:
            out.append(tok)
    result = " ".join(out).strip()
    # Never return empty (e.g. an all-stopword query) — fall back to a
    # punctuation-stripped version of the original.
    fallback = " ".join(re.sub(r"[^\w\s-]", " ", query, flags=re.UNICODE).split()).strip()
    return result or fallback or query.strip()


class SearchEngine(ABC):
    """Common interface every search provider implements.

    `search(query, max_results)` returns a list of result dicts with at
    least `title`, `url`, `content` keys.

    `fetch(url, raw=False)` retrieves a page and returns either readable
    text (default, via trafilatura) or raw HTML. Errors return a dict
    with an `error` key — the engine never raises so the agent loop
    stays resilient.
    """

    name: str = "unknown"

    @abstractmethod
    def search(self, query: str, *, max_results: int = 5) -> list[dict]: ...

    def fetch(
        self,
        url: str,
        *,
        raw: bool = False,
        timeout: float = _DEFAULT_FETCH_TIMEOUT,
    ) -> dict:
        return _default_fetch(url, raw=raw, timeout=timeout)


# ---------------------------------------------------------------------------
# Shared fetch helpers
# ---------------------------------------------------------------------------


def _default_fetch(
    url: str,
    *,
    raw: bool = False,
    timeout: float = _DEFAULT_FETCH_TIMEOUT,
    resolver=None,
) -> dict:
    """httpx GET → trafilatura extract (or raw HTML if raw=True).

    v0.5.0: SSRF-guarded — refuses non-http(s) schemes and hosts that
    resolve to private/loopback/link-local/reserved/metadata addresses,
    and re-validates the final URL after redirects. ``resolver`` is passed
    through to the SSRF guard (default: real DNS); inject one for offline
    tests.
    """
    try:
        from sherlock.security.urlguard import is_safe_url

        ok, reason = is_safe_url(url, resolver=resolver)
        if not ok:
            return {"error": f"blocked unsafe url: {reason}", "url": url}
    except Exception:
        if not str(url).lower().startswith(("http://", "https://")):
            return {"error": "blocked non-http url", "url": url}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(
                url, headers={"User-Agent": "Sherlock/0.5 (+https://github.com/MinwooKim1990)"}
            )
            r.raise_for_status()
            try:
                from sherlock.security.urlguard import is_safe_url as _safe2

                ok2, reason2 = _safe2(str(r.url), resolver=resolver)
                if not ok2:
                    return {"error": f"blocked redirect target: {reason2}", "url": str(r.url)}
            except Exception:
                pass
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "url": url}

    if raw:
        return {
            "url": url,
            "status": r.status_code,
            "html": r.text[:_RAW_HTML_CAP],
        }
    return {
        "url": url,
        "status": r.status_code,
        "text": _extract_text(r.text)[:_TEXT_EXTRACT_CAP],
        "image": _extract_og_image(r.text),
        "date": _extract_date(r.text),
    }


def _extract_date(html: str) -> str:
    """Best-effort published/updated date for freshness — an OPAQUE source-reported
    string (never parsed numerically in code). Scans article:published_time /
    og:updated_time / JSON-LD datePublished / <time datetime>. Returns '', never raises."""
    try:
        import re as _re

        for prop in ("article:published_time", "article:modified_time", "og:updated_time"):
            p = _re.escape(prop)
            m = _re.search(
                r'<meta[^>]+(?:property|name)=["\']' + p + r'["\'][^>]+content=["\']([^"\']+)',
                html,
                _re.I,
            ) or _re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']' + p,
                html,
                _re.I,
            )
            if m and m.group(1).strip():
                return m.group(1).strip()[:40]
        m = _re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
        if m:
            return m.group(1).strip()[:40]
        m = _re.search(r"<time[^>]+datetime=[\"']([^\"']+)", html, _re.I)
        if m:
            return m.group(1).strip()[:40]
        return ""
    except Exception:
        return ""


def _extract_og_image(html: str) -> str:
    """Best-effort lead image (og:image / twitter:image) for a report to embed.
    Returns an absolute http(s) URL or '' — never raises, skips relative URLs."""
    try:
        import re as _re

        for prop in ("og:image:secure_url", "og:image", "twitter:image", "og:image:url"):
            p = _re.escape(prop)
            m = _re.search(
                r'<meta[^>]+(?:property|name)=["\']' + p + r'["\'][^>]+content=["\']([^"\']+)',
                html,
                _re.I,
            ) or _re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']' + p,
                html,
                _re.I,
            )
            if m:
                u = (m.group(1) or "").strip()
                if u.startswith("//"):
                    u = "https:" + u
                if u.startswith("http"):
                    return u
        return ""
    except Exception:
        return ""


def _extract_text(html: str) -> str:
    """Try trafilatura first; fall back to BeautifulSoup."""
    try:
        import trafilatura

        out = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        if out:
            return out
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Collapse excess blank lines.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception:
        return html  # last resort


# ---------------------------------------------------------------------------
# DuckDuckGo (default, no key)
# ---------------------------------------------------------------------------


@dataclass
class DuckDuckGoSearch(SearchEngine):
    name: str = "duckduckgo"
    region: str = "wt-wt"
    safesearch: str = "moderate"

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw = list(
                    ddgs.text(
                        query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=max_results,
                    )
                )
        except Exception as exc:
            return [{"error": f"DuckDuckGo failed: {type(exc).__name__}: {exc}", "query": query}]
        out: list[dict] = []
        for r in raw:
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href") or r.get("url", ""),
                    "content": r.get("body") or r.get("snippet", ""),
                    "source": "duckduckgo",
                    "date": r.get("date") or "",
                }
            )
        return out


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


@dataclass
class TavilySearch(SearchEngine):
    api_key: str
    name: str = "tavily"
    _client: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)
        except Exception:
            self._client = None

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        if self._client is None:
            return [{"error": "Tavily client not initialised", "query": query}]
        try:
            res = self._client.search(query=query, max_results=max_results)
        except Exception as exc:
            return [{"error": f"Tavily failed: {type(exc).__name__}: {exc}", "query": query}]
        out: list[dict] = []
        for r in res.get("results", []):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "source": "tavily",
                    "date": r.get("published_date") or "",
                }
            )
        return out

    def fetch(
        self, url: str, *, raw: bool = False, timeout: float = _DEFAULT_FETCH_TIMEOUT
    ) -> dict:
        """Use Tavily's extract endpoint for better extraction. Fall back to default."""
        if self._client is None or raw:
            return _default_fetch(url, raw=raw, timeout=timeout)
        try:
            res = self._client.extract(urls=[url])
            items = res.get("results") if isinstance(res, dict) else None
            if items:
                first = items[0]
                return {
                    "url": url,
                    "status": 200,
                    "text": (first.get("raw_content") or first.get("content") or "")[
                        :_TEXT_EXTRACT_CAP
                    ],
                }
        except Exception:
            pass
        return _default_fetch(url, raw=raw, timeout=timeout)


# ---------------------------------------------------------------------------
# Brave
# ---------------------------------------------------------------------------


@dataclass
class BraveSearch(SearchEngine):
    api_key: str
    name: str = "brave"
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.get(
                    self.endpoint,
                    headers={
                        "X-Subscription-Token": self.api_key,
                        "Accept": "application/json",
                    },
                    params={"q": query, "count": max_results},
                )
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            return [{"error": f"Brave failed: {type(exc).__name__}: {exc}", "query": query}]
        out: list[dict] = []
        for item in (data.get("web", {}).get("results") or [])[:max_results]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("description", ""),
                    "source": "brave",
                    "date": item.get("page_age") or item.get("age") or "",
                }
            )
        return out


# ---------------------------------------------------------------------------
# Valyu
# ---------------------------------------------------------------------------


@dataclass
class ValyuSearch(SearchEngine):
    api_key: str
    name: str = "valyu"
    endpoint: str = "https://api.valyu.network/v1/knowledge"

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(
                    self.endpoint,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "search_type": "web",
                        "max_num_results": max_results,
                    },
                )
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            return [{"error": f"Valyu failed: {type(exc).__name__}: {exc}", "query": query}]
        out: list[dict] = []
        results = data.get("results") or data.get("data") or []
        for item in results[:max_results]:
            out.append(
                {
                    "title": item.get("title", "") or item.get("source", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content")
                    or item.get("snippet")
                    or item.get("description", ""),
                    "source": "valyu",
                    "date": item.get("date") or item.get("published_date") or "",
                }
            )
        return out


# ---------------------------------------------------------------------------
# Stub (no-op, kept for hermetic tests)
# ---------------------------------------------------------------------------


@dataclass
class StubSearch(SearchEngine):
    name: str = "stub"

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        return [
            {
                "title": f"[stub] {query}",
                "url": "https://example.com/",
                "content": "Web search is not configured. Sherlock proceeded without fresh web context.",
                "source": "stub",
            }
        ]

    def fetch(
        self, url: str, *, raw: bool = False, timeout: float = _DEFAULT_FETCH_TIMEOUT
    ) -> dict:
        return {
            "url": url,
            "status": 0,
            "text": "[stub] page fetch disabled.",
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[SearchEngine]] = {
    "duckduckgo": DuckDuckGoSearch,
    "ddg": DuckDuckGoSearch,
    "tavily": TavilySearch,
    "brave": BraveSearch,
    "valyu": ValyuSearch,
    "stub": StubSearch,
}


def _resolve_key(api_key: Optional[str], api_key_env: Optional[str]) -> Optional[str]:
    """Resolve API key from direct value > env var > None."""
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


def create_search(
    engine: str = "duckduckgo",
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    **engine_kwargs,
) -> SearchEngine:
    """One-stop factory.

    Args:
        engine: provider name — "duckduckgo" / "tavily" / "brave" / "valyu" / "stub".
        api_key: direct API key (highest priority).
        api_key_env: env var name to look up the key from (fallback).
        **engine_kwargs: passed through to the engine constructor
            (e.g. `region="us-en"` for DuckDuckGoSearch).

    Returns:
        A `SearchEngine` instance ready to call.

    Raises:
        ValueError: unknown engine name, or required key missing.
    """
    name = (engine or "").strip().lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown search engine '{engine}'. " f"Available: {sorted(set(_REGISTRY.keys()))}"
        )
    cls = _REGISTRY[name]
    if cls is DuckDuckGoSearch or cls is StubSearch:
        return cls(**engine_kwargs)
    key = _resolve_key(api_key, api_key_env)
    if not key:
        raise ValueError(
            f"Search engine '{name}' requires an API key. "
            f"Pass api_key=... directly or set api_key_env to point at an "
            f"environment variable containing the key."
        )
    return cls(api_key=key, **engine_kwargs)


def build_search_engine(cfg: "SearchConfig | None") -> SearchEngine | None:
    """YAML-driven config path. Backward-compat wrapper around create_search."""
    if cfg is None or not cfg.always_on:
        return None
    try:
        return create_search(
            cfg.provider,
            api_key=getattr(cfg, "api_key", None),
            api_key_env=getattr(cfg, "api_key_env", None),
        )
    except Exception:
        return StubSearch()


def build_role_engines(
    cfg: "SearchConfig | None",
) -> tuple[SearchEngine | None, SearchEngine | None]:
    """Build (main_engine, inference_engine) from SearchConfig with per-role overrides.

    Per-role fields (main_provider/main_api_key/main_api_key_env, etc.)
    override the global provider/api_key/api_key_env when set.
    Returns (None, None) if `always_on=False`.
    """
    if cfg is None or not cfg.always_on:
        return None, None

    def _one(role: str) -> SearchEngine:
        prov = getattr(cfg, f"{role}_provider", None) or cfg.provider
        key = getattr(cfg, f"{role}_api_key", None) or getattr(cfg, "api_key", None)
        key_env = getattr(cfg, f"{role}_api_key_env", None) or getattr(cfg, "api_key_env", None)
        try:
            return create_search(prov, api_key=key, api_key_env=key_env)
        except Exception:
            return StubSearch()

    main_engine = _one("main")
    inference_engine = _one("inference")
    return main_engine, inference_engine
