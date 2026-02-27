from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class TranscribeRequest(BaseModel):
    audio_url: HttpUrl
    webhook_url: HttpUrl
    model: str = Field(default="large-v3-turbo")
    language: str | None = Field(default=None)
    batch_size: int = Field(default=4, ge=1, le=64)
    chunk_size: int = Field(default=8, ge=1, le=120)
    compute_type: str = Field(default="auto")
    align_model: str | None = Field(default=None)
    return_word_timestamps: bool = Field(default=True)


class AcceptedResponse(BaseModel):
    job_id: str
    status: Literal["accepted"] = "accepted"


class WebhookBase(BaseModel):
    job_id: str
    status: Literal["succeeded", "failed"]
    model: str
    language: str | None


class WebhookSucceeded(WebhookBase):
    status: Literal["succeeded"] = "succeeded"
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]] | None = None
    timings: dict[str, float] | None = None


class WebhookFailed(WebhookBase):
    status: Literal["failed"] = "failed"
    error: str
