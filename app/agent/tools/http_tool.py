"""Generic HTTP tool with an SSRF allowlist.

`is_host_allowed` enforces two gates: the host must be on the configured
allowlist AND must not be a loopback/private/link-local address. The private
check wins, so even an allowlisted 'localhost' is refused.
"""
import ipaddress
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

from app.config import settings


def is_host_allowed(url: str, allowed_hosts: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host or host == "localhost":
        return False

    # Block IP literals in private / loopback / link-local / reserved ranges.
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return False
    except ValueError:
        pass  # not an IP literal — it's a hostname

    allowed = [h.lower() for h in allowed_hosts]
    return any(host == h or host.endswith("." + h) for h in allowed)


@tool
def http_request(url: str, method: str = "GET", body: str | None = None) -> str:
    """Call an external HTTP API (GET or POST). Only allowlisted hosts are permitted.

    Returns the status code followed by the (truncated) response body.
    """
    if not is_host_allowed(url, settings.allowed_http_hosts):
        return (
            f"Blocked: '{url}' is not on the allowed host list "
            f"({settings.allowed_http_hosts or 'none configured'})."
        )
    try:
        with httpx.Client(timeout=10.0, follow_redirects=False) as client:
            resp = client.request(method.upper(), url, content=body)
        return f"HTTP {resp.status_code}\n{resp.text[:4000]}"
    except Exception as exc:  # noqa: BLE001
        return f"Request failed: {exc}"
