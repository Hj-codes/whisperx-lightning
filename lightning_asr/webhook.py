from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlparse
from typing import Any

import httpx

from lightning_asr.logging_utils import get_logger, log_event

logger = get_logger(__name__)


def validate_webhook_url(webhook_url: str, *, timeout_seconds: float = 5.0) -> None:
    parsed = urlparse(str(webhook_url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Webhook URL must be an absolute http(s) URL")

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.head(webhook_url)
    except Exception as exc:
        raise RuntimeError(f"Webhook URL is unreachable: {webhook_url}") from exc

    log_event(
        logger,
        level=20,
        event="webhook_validation",
        message="Webhook endpoint responded to validation probe",
        webhook_url=webhook_url,
        status_code=response.status_code,
    )


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
            log_event(
                logger,
                level=20,
                event="webhook_attempt",
                message="Posting webhook payload",
                webhook_url=webhook_url,
                attempt=attempt,
                max_attempts=max_attempts,
                payload_status=payload.get("status"),
            )
            with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                resp = client.post(webhook_url, content=body, headers=headers)
                resp.raise_for_status()
            log_event(
                logger,
                level=20,
                event="webhook_success",
                message="Webhook delivered successfully",
                webhook_url=webhook_url,
                attempt=attempt,
                status_code=resp.status_code,
                payload_status=payload.get("status"),
            )
            return
        except Exception as exc:
            last_exc = exc
            log_event(
                logger,
                level=40 if attempt >= max_attempts else 30,
                event="webhook_retry" if attempt < max_attempts else "webhook_failure",
                message="Webhook delivery failed",
                webhook_url=webhook_url,
                attempt=attempt,
                max_attempts=max_attempts,
                payload_status=payload.get("status"),
                error=str(exc),
            )
            if attempt >= max_attempts:
                break
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))))
    if last_exc:
        raise last_exc
