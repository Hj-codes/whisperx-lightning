from __future__ import annotations

import ipaddress
import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib.parse import urlparse

import httpx


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(url: str) -> str:
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Unsupported URL scheme")
    if not parsed.hostname:
        raise ValueError("URL missing hostname")

    default_port = 443 if parsed.scheme == "https" else 80
    addr_info = socket.getaddrinfo(
        parsed.hostname, parsed.port or default_port, type=socket.SOCK_STREAM
    )
    for family, _, _, _, sockaddr in addr_info:
        ip_str = str(sockaddr[0])
        if family in {socket.AF_INET, socket.AF_INET6} and not _is_public_ip(ip_str):
            raise ValueError("URL resolves to a private or restricted IP")

    return parsed.geturl()


@contextmanager
def download_url_to_tempfile(
    *,
    url: str,
    suffix: str = ".audio",
    timeout_seconds: float = 120.0,
    max_bytes: int = 500 * 1024 * 1024,
) -> Iterator[Path]:
    checked_url = validate_public_url(url)
    total = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = Path(tmp.name)
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client, client.stream(
            "GET", checked_url
        ) as resp:
            resp.raise_for_status()
            validate_public_url(str(resp.url))
            with open(path, "wb") as f:
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Downloaded audio exceeds size limit")
                    f.write(chunk)
        if total <= 0:
            raise ValueError("Downloaded audio is empty")
        yield path
    finally:
        with suppress(Exception):
            path.unlink(missing_ok=True)
