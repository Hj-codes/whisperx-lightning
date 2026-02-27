from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

import httpx


def post_webhook_json(
    *,
    webhook_url: str,
    payload: dict[str, Any],
    timeout_seconds: float = 15.0,
    max_attempts: int = 3,
) -> None:
    secret = str(os.getenv("LIGHTNING_WEBHOOK_SECRET", "") or "").encode("utf-8")
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if secret:
        signature = hmac.new(secret, body, hashlib.sha256).hexdigest()
        headers["X-Lightning-Signature"] = signature

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                resp = client.post(webhook_url, content=body, headers=headers)
                resp.raise_for_status()
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))))
    if last_exc:
        raise last_exc
