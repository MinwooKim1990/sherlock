"""URL safety / SSRF guard (v0.5.0).

`fetch` is reachable by the main LLM via the `<<sherlock-tool: fetch URL>>`
tag, so a manipulated model could try to reach internal services or cloud
metadata endpoints. `is_safe_url` rejects:
  - non-http(s) schemes
  - hosts that resolve to private / loopback / link-local / reserved IPs
  - cloud metadata IPs (169.254.169.254, fd00:ec2::254)

DNS is resolved so that a public hostname pointing at a private IP
(DNS-rebinding style) is still caught.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Callable, Optional
from urllib.parse import urlparse

_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}

# A resolver maps (host, port) -> list of resolved IP strings. The default
# uses the real DNS; tests (or sandboxed callers) can inject a pure function
# so the SSRF logic can be exercised with NO network access.
Resolver = Callable[[str, int], "list[str]"]


def _default_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port)
    return [info[4][0] for info in infos]


def _ip_is_blocked(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → block
    if str(addr) in _METADATA_IPS:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def is_safe_url(url: str, resolver: Optional[Resolver] = None) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means do NOT fetch.

    ``resolver`` overrides DNS resolution (default: real ``getaddrinfo``).
    Inject one to test the guard offline — e.g. a function that maps a public
    hostname to a private IP exercises the DNS-rebinding defense with no net.
    """
    if not url or not isinstance(url, str):
        return False, "empty url"
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, f"scheme '{parsed.scheme}' not allowed (http/https only)"
    host = parsed.hostname
    if not host:
        return False, "no host"
    # Literal-IP host:
    try:
        ipaddress.ip_address(host)
        if _ip_is_blocked(host):
            return False, f"blocked IP {host}"
        return True, "ok"
    except ValueError:
        pass
    # Hostname → resolve all A/AAAA records and block if ANY is private.
    resolve = resolver or _default_resolver
    try:
        ips = resolve(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except Exception as exc:
        return False, f"dns resolution failed: {exc}"
    for ip in ips:
        if _ip_is_blocked(ip):
            return False, f"host {host} resolves to blocked IP {ip}"
    return True, "ok"
