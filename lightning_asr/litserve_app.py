from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import suppress
from typing import Any, Literal, cast

import httpx
import litserve as ls

from lightning_asr.job_queue import Job, JobQueue
from lightning_asr.logging_utils import get_logger, log_event
from lightning_asr.schemas import TranscribeRequest
from lightning_asr.url_io import download_url_to_tempfile
from lightning_asr.webhook import post_webhook_json, validate_webhook_url

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "8"))
logger = get_logger(__name__)


def _register_torch_safe_globals(torch_module: Any) -> None:
    # PyTorch 2.6+ safe-loading may reject some OmegaConf classes used by
    # pyannote checkpoints. Allow-list trusted classes before loading models.
    try:
        from omegaconf import DictConfig, ListConfig

        torch_module.serialization.add_safe_globals([ListConfig, DictConfig])
    except Exception:
        return


def _read_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw_value = str(os.environ.get(name, str(default))).strip()
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _read_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw_value = str(os.environ.get(name, str(default))).strip()
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _read_timeout_env(name: str, default: bool | float = False) -> bool | float:
    raw_value = str(os.environ.get(name, str(default))).strip().lower()
    if raw_value in {"false", "off", "none"}:
        return False
    return _read_float_env(name, 0.0, minimum=0.0)


def _read_devices_env(name: str = "LITSERVE_DEVICES") -> int | Literal["auto"]:
    raw_value = str(os.environ.get(name, "auto")).strip().lower()
    if raw_value == "auto":
        return "auto"
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be 'auto' or an integer, got {raw_value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def _normalize_whisperx_device(device: str) -> str:
    normalized = str(device or "").strip().lower()
    if normalized.startswith("cuda"):
        # LitServe passes indexed devices like `cuda:0`, but WhisperX/faster-whisper
        # expects the backend name (`cuda`) and derives the visible GPU itself.
        return "cuda"
    return normalized or "cpu"


class WhisperXLitAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self._is_ready = False
        self._queue_started = False
        log_event(
            logger,
            logging.INFO,
            "setup_started",
            "Initializing Lightning WhisperX service",
            device=device,
        )
        os.environ.setdefault("HF_HOME", "/app/models/huggingface")
        os.environ.setdefault("TORCH_HOME", "/app/models/torch")
        os.environ.setdefault("XDG_CACHE_HOME", "/app/models")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/app/models/huggingface")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/models/huggingface/hub")
        # Keep online access enabled by default so runtime can fetch
        # uncached alignment models (for example Hindi) on demand.
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

        import torch
        import whisperx

        self._torch: Any = torch
        self._whisperx: Any = whisperx
        _register_torch_safe_globals(torch)
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_version = torch.version.cuda if cuda_available else None
        log_event(
            logger,
            logging.INFO,
            "gpu_diagnostics",
            "Resolved GPU runtime information",
            device=device,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_version=cuda_version,
            torch_version=torch.__version__,
        )
        if not str(device).startswith("cuda") and cuda_available:
            logger.warning(
                "LitServe selected device '%s' even though CUDA is available; "
                "check LITSERVE_ACCELERATOR/LITSERVE_DEVICES and deployment GPU assignment.",
                device,
            )
        self._device = device
        self._whisperx_device = _normalize_whisperx_device(device)
        self._queue: JobQueue[TranscribeRequest] = JobQueue(max_queue_size=1000)

        self._model_name = os.environ.get("WHISPERX_MODEL", "large-v3-turbo")
        self._compute_type = self._resolve_compute_type(
            os.environ.get("WHISPERX_COMPUTE_TYPE", "auto")
        )
        self._model: Any = None
        self._align_cache: dict[tuple[str, str | None], tuple[Any, Any]] = {}
        self._asr_options_signature: tuple[tuple[str, Any], ...] = ()
        self._vad_options_signature: tuple[tuple[str, Any], ...] = ()

        self._load_model(self._model_name, self._compute_type)
        self._warmup_model()
        self._queue.start(processor=self._process_job)
        self._queue_started = True
        self._is_ready = True
        log_event(
            logger,
            logging.INFO,
            "setup_completed",
            "Lightning WhisperX service is ready",
            device=device,
            model=self._model_name,
            compute_type=self._compute_type,
        )

    def decode_request(self, request: Any) -> TranscribeRequest:
        if isinstance(request, str):
            return TranscribeRequest.model_validate_json(request)
        if isinstance(request, bytes):
            return TranscribeRequest.model_validate_json(request.decode("utf-8"))
        if isinstance(request, dict):
            return TranscribeRequest.model_validate(request)
        if hasattr(request, "json"):
            parsed = request.json()
            if isinstance(parsed, str):
                return TranscribeRequest.model_validate_json(parsed)
            return TranscribeRequest.model_validate(parsed)
        return TranscribeRequest.model_validate(request)

    def predict(self, request: TranscribeRequest) -> dict[str, Any]:
        job_id = self._queue.submit(request)
        queue_size = None
        with suppress(Exception):
            queue_size = self._queue._q.qsize()
        log_event(
            logger,
            logging.INFO,
            "job_accepted",
            "Accepted Lightning transcription job",
            job_id=job_id,
            language=request.language,
            model=request.model,
            queue_size=queue_size,
        )
        return {"job_id": job_id, "status": "accepted"}

    def encode_response(self, output: dict[str, Any]) -> Any:
        try:
            from fastapi.responses import JSONResponse

            return JSONResponse(content=output, status_code=202)
        except Exception:
            return output

    def _resolve_compute_type(self, requested: str) -> str:
        normalized = (requested or "").strip().lower()
        if not normalized or normalized == "auto":
            return "float16" if str(self._device).startswith("cuda") else "int8"
        return normalized

    def _load_model(
        self,
        model_name: str,
        compute_type: str,
        *,
        asr_options: dict[str, Any] | None = None,
        vad_options: dict[str, Any] | None = None,
    ) -> None:
        started = time.perf_counter()
        resolved_compute_type = self._resolve_compute_type(compute_type)
        resolved_asr_options = asr_options or {}
        resolved_vad_options = vad_options or {}
        self._model_name = model_name
        self._compute_type = resolved_compute_type
        try:
            self._model = self._whisperx.load_model(
                model_name,
                self._whisperx_device,
                compute_type=resolved_compute_type,
                asr_options=resolved_asr_options or None,
                language=None,
                task="transcribe",
                vad_options=resolved_vad_options or None,
            )
            self._asr_options_signature = self._options_signature(resolved_asr_options)
            self._vad_options_signature = self._options_signature(resolved_vad_options)
            log_event(
                logger,
                logging.INFO,
                "model_loaded",
                "Loaded WhisperX model",
                model=model_name,
                device=self._device,
                compute_type=self._compute_type,
                duration_seconds=round(time.perf_counter() - started, 4),
            )
        except ValueError as exc:
            msg = str(exc)
            if "do not support efficient float16 computation" not in msg:
                raise
            fallback_compute_type = "int8_float16" if str(self._device).startswith("cuda") else "int8"
            self._model = self._whisperx.load_model(
                model_name,
                self._whisperx_device,
                compute_type=fallback_compute_type,
                asr_options=resolved_asr_options or None,
                language=None,
                task="transcribe",
                vad_options=resolved_vad_options or None,
            )
            self._compute_type = fallback_compute_type
            self._asr_options_signature = self._options_signature(resolved_asr_options)
            self._vad_options_signature = self._options_signature(resolved_vad_options)
            log_event(
                logger,
                logging.WARNING,
                "model_loaded_with_fallback",
                "Loaded WhisperX model with fallback compute type",
                model=model_name,
                device=self._device,
                requested_compute_type=resolved_compute_type,
                compute_type=self._compute_type,
                duration_seconds=round(time.perf_counter() - started, 4),
            )

    def _warmup_model(self) -> None:
        try:
            import numpy as np

            started = time.perf_counter()
            dummy_audio = np.zeros(16000, dtype="float32")
            self._model.transcribe(audio=dummy_audio, batch_size=1, chunk_size=1, language="en")
            log_event(
                logger,
                logging.INFO,
                "model_warmed",
                "Completed WhisperX model warmup",
                model=self._model_name,
                compute_type=self._compute_type,
                duration_seconds=round(time.perf_counter() - started, 4),
            )
        except Exception:
            logger.exception(
                "WhisperX model warmup failed",
                extra={
                    "event": "model_warmup_failed",
                    "fields": {
                        "model": self._model_name,
                        "compute_type": self._compute_type,
                    },
                },
            )

    def health(self) -> bool:
        return super().health()

    def _options_signature(self, options: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        normalized: list[tuple[str, Any]] = []
        for key, value in options.items():
            if isinstance(value, list):
                normalized.append((key, tuple(value)))
            else:
                normalized.append((key, value))
        return tuple(sorted(normalized))

    def _build_asr_options(self, req: TranscribeRequest) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for key in [
            "beam_size",
            "best_of",
            "patience",
            "length_penalty",
            "temperatures",
            "compression_ratio_threshold",
            "log_prob_threshold",
            "no_speech_threshold",
            "initial_prompt",
        ]:
            value = getattr(req, key)
            if value is not None:
                options[key] = value
        return options

    def _build_vad_options(self, req: TranscribeRequest) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if req.vad_onset is not None:
            options["vad_onset"] = req.vad_onset
        if req.vad_offset is not None:
            options["vad_offset"] = req.vad_offset
        return options

    def _build_transcribe_kwargs(self, req: TranscribeRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": req.batch_size,
            "chunk_size": req.chunk_size if req.chunk_size else CHUNK_SIZE,
            "language": req.language,
        }
        return kwargs

    def _get_align_bundle(self, *, language: str, align_model: str | None) -> tuple[Any, Any]:
        key = (language, align_model)
        if key in self._align_cache:
            return self._align_cache[key]
        bundle = self._whisperx.load_align_model(
            language_code=language, device=self._whisperx_device, model_name=align_model
        )
        self._align_cache[key] = bundle
        return bundle

    def _process_job(self, job: Job[TranscribeRequest]) -> None:
        started = time.perf_counter()
        req = job.payload
        audio = None
        transcript = None
        aligned = None
        download_seconds: float | None = None
        transcription_seconds: float | None = None
        alignment_seconds: float | None = None
        payload_base = {
            "job_id": job.job_id,
            "model": req.model,
            "language": req.language,
        }
        log_event(
            logger,
            logging.INFO,
            "job_started",
            "Processing Lightning transcription job",
            job_id=job.job_id,
            audio_url=str(req.audio_url),
            model=req.model,
            language=req.language,
        )
        try:
            validate_webhook_url(str(req.webhook_url))
            requested_compute_type = self._resolve_compute_type(req.compute_type)
            requested_asr_options = self._build_asr_options(req)
            requested_vad_options = self._build_vad_options(req)
            if (
                req.model != self._model_name
                or requested_compute_type != self._compute_type
                or self._options_signature(requested_asr_options)
                != self._asr_options_signature
                or self._options_signature(requested_vad_options)
                != self._vad_options_signature
            ):
                self._load_model(
                    req.model,
                    requested_compute_type,
                    asr_options=requested_asr_options,
                    vad_options=requested_vad_options,
                )
                self._warmup_model()

            download_started = time.perf_counter()
            with download_url_to_tempfile(url=str(req.audio_url), suffix=".audio") as audio_path:
                audio_bytes = audio_path.stat().st_size
                audio = self._whisperx.load_audio(str(audio_path))
            download_seconds = round(time.perf_counter() - download_started, 4)
            log_event(
                logger,
                logging.INFO,
                "audio_downloaded",
                "Downloaded audio input for transcription",
                job_id=job.job_id,
                audio_bytes=audio_bytes,
                duration_seconds=download_seconds,
            )

            transcription_started = time.perf_counter()
            transcript = self._model.transcribe(audio=audio, **self._build_transcribe_kwargs(req))
            transcription_seconds = round(time.perf_counter() - transcription_started, 4)
            lang = str(transcript.get("language") or req.language or "en")
            log_event(
                logger,
                logging.INFO,
                "transcription_completed",
                "Completed WhisperX transcription",
                job_id=job.job_id,
                detected_language=lang,
                segment_count=len(transcript.get("segments", [])),
                duration_seconds=transcription_seconds,
            )

            alignment_started = time.perf_counter()
            (align_model, align_meta) = self._get_align_bundle(
                language=lang, align_model=req.align_model
            )
            aligned = self._whisperx.align(
                transcript.get("segments", []),
                align_model,
                align_meta,
                audio,
                self._whisperx_device,
                interpolate_method="nearest",
                return_char_alignments=req.return_char_alignments,
            )
            alignment_seconds = round(time.perf_counter() - alignment_started, 4)

            segments = aligned.get("segments", [])
            words = None
            if req.return_word_timestamps:
                flat: list[dict[str, Any]] = []
                for seg in segments:
                    for w in seg.get("words") or []:
                        if isinstance(w, dict):
                            flat.append(w)
                words = flat

            total_seconds = round(time.perf_counter() - started, 4)
            timings = {
                "download_seconds": download_seconds,
                "transcription_seconds": transcription_seconds,
                "alignment_seconds": alignment_seconds,
                "total_seconds": total_seconds,
            }
            log_event(
                logger,
                logging.INFO,
                "alignment_completed",
                "Completed WhisperX alignment",
                job_id=job.job_id,
                language=aligned.get("language", lang),
                segment_count=len(segments),
                word_count=len(words) if words is not None else 0,
                duration_seconds=alignment_seconds,
            )
            post_webhook_json(
                webhook_url=str(req.webhook_url),
                payload={
                    **payload_base,
                    "status": "succeeded",
                    "language": aligned.get("language", lang),
                    "segments": segments,
                    "words": words,
                    "timings": timings,
                },
            )
            log_event(
                logger,
                logging.INFO,
                "job_completed",
                "Lightning transcription job completed",
                job_id=job.job_id,
                total_seconds=total_seconds,
                segment_count=len(segments),
            )
        except Exception as exc:
            safe_error = str(exc)
            if isinstance(exc, httpx.HTTPError):
                safe_error = f"HTTP error: {exc}"
            logger.exception(
                "Lightning transcription job failed",
                extra={
                    "event": "job_failed",
                    "fields": {
                        "job_id": job.job_id,
                        "audio_url": str(req.audio_url),
                        "model": req.model,
                        "language": req.language,
                        "error": safe_error,
                    },
                },
            )
            with suppress(Exception):
                post_webhook_json(
                    webhook_url=str(req.webhook_url),
                    payload={**payload_base, "status": "failed", "error": safe_error},
                )
        finally:
            with suppress(Exception):
                del audio
            with suppress(Exception):
                del transcript
            with suppress(Exception):
                del aligned
            self._cleanup()

    def _cleanup(self) -> None:
        gc.collect()
        if getattr(self._torch, "cuda", None) is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def build_server() -> Any:
    api = WhisperXLitAPI(
        max_batch_size=_read_int_env("LITSERVE_MAX_BATCH_SIZE", 1, minimum=1),
        batch_timeout=_read_float_env("LITSERVE_BATCH_TIMEOUT", 0.0, minimum=0.0),
    )
    accelerator = cast(
        Literal["cpu", "cuda", "mps", "auto"],
        str(os.environ.get("LITSERVE_ACCELERATOR", "cuda")).strip().lower() or "cuda",
    )
    devices = _read_devices_env()
    workers_per_device = _read_int_env("LITSERVE_WORKERS_PER_DEVICE", 1, minimum=1)
    timeout = _read_timeout_env("LITSERVE_TIMEOUT", False)
    logger.info(
        "Configuring LitServe server",
        extra={
            "event": "server_configuration",
            "fields": {
                "accelerator": accelerator,
                "devices": devices,
                "workers_per_device": workers_per_device,
                "max_batch_size": api.max_batch_size,
                "batch_timeout": api.batch_timeout,
                "timeout": timeout,
            },
        },
    )
    return ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        timeout=timeout,
    )


if __name__ == "__main__":
    server = build_server()
    server.run(port=int(os.environ.get("PORT", "8000")))
