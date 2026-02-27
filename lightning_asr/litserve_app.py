from __future__ import annotations

import gc
import os
import time
from contextlib import suppress
from typing import Any

import httpx

from lightning_asr.job_queue import Job, JobQueue
from lightning_asr.schemas import TranscribeRequest
from lightning_asr.url_io import download_url_to_tempfile
from lightning_asr.webhook import post_webhook_json

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "8"))


def _register_torch_safe_globals(torch_module: Any) -> None:
    # PyTorch 2.6+ safe-loading may reject some OmegaConf classes used by
    # pyannote checkpoints. Allow-list trusted classes before loading models.
    try:
        from omegaconf import DictConfig, ListConfig

        torch_module.serialization.add_safe_globals([ListConfig, DictConfig])
    except Exception:
        return


class WhisperXLitAPI:
    def setup(self, device: str) -> None:
        os.environ.setdefault("HF_HOME", "/app/models/huggingface")
        os.environ.setdefault("TORCH_HOME", "/app/models/torch")
        os.environ.setdefault("XDG_CACHE_HOME", "/app/models")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/app/models/huggingface")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/models/huggingface/hub")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        import whisperx

        self._torch: Any = torch
        self._whisperx: Any = whisperx
        _register_torch_safe_globals(torch)
        self._device = device
        self._queue: JobQueue[TranscribeRequest] = JobQueue(max_queue_size=1000)

        self._model_name = "large-v3"
        self._compute_type = "float16"
        self._model: Any = None
        self._align_cache: dict[tuple[str, str | None], tuple[Any, Any]] = {}

        self._load_model(self._model_name, self._compute_type)
        self._queue.start(processor=self._process_job)

    def decode_request(self, request: Any) -> TranscribeRequest:
        if isinstance(request, dict):
            return TranscribeRequest.model_validate(request)
        if hasattr(request, "json"):
            return TranscribeRequest.model_validate(request.json())
        return TranscribeRequest.model_validate(request)

    def predict(self, request: TranscribeRequest) -> dict[str, Any]:
        job_id = self._queue.submit(request)
        return {"job_id": job_id, "status": "accepted"}

    def encode_response(self, output: dict[str, Any]) -> Any:
        try:
            from fastapi.responses import JSONResponse

            return JSONResponse(content=output, status_code=202)
        except Exception:
            return output

    def _load_model(self, model_name: str, compute_type: str) -> None:
        self._model_name = model_name
        self._compute_type = compute_type
        self._model = self._whisperx.load_model(
            model_name,
            self._device,
            compute_type=compute_type,
            language=None,
            task="transcribe",
        )

    def _get_align_bundle(self, *, language: str, align_model: str | None) -> tuple[Any, Any]:
        key = (language, align_model)
        if key in self._align_cache:
            return self._align_cache[key]
        bundle = self._whisperx.load_align_model(
            language_code=language, device=self._device, model_name=align_model
        )
        self._align_cache[key] = bundle
        return bundle

    def _process_job(self, job: Job[TranscribeRequest]) -> None:
        started = time.perf_counter()
        req = job.payload
        payload_base = {
            "job_id": job.job_id,
            "model": req.model,
            "language": req.language,
        }
        try:
            if req.model != self._model_name or req.compute_type != self._compute_type:
                self._load_model(req.model, req.compute_type)

            with download_url_to_tempfile(url=str(req.audio_url), suffix=".audio") as audio_path:
                audio = self._whisperx.load_audio(str(audio_path))
            transcript = self._model.transcribe(
                audio=audio,
                batch_size=req.batch_size,
                chunk_size=req.chunk_size if req.chunk_size else CHUNK_SIZE,
                language=req.language,
            )
            lang = str(transcript.get("language") or req.language or "en")
            (align_model, align_meta) = self._get_align_bundle(
                language=lang, align_model=req.align_model
            )
            aligned = self._whisperx.align(
                transcript.get("segments", []),
                align_model,
                align_meta,
                audio,
                self._device,
                interpolate_method="nearest",
                return_char_alignments=False,
            )

            segments = aligned.get("segments", [])
            words = None
            if req.return_word_timestamps:
                flat: list[dict[str, Any]] = []
                for seg in segments:
                    for w in seg.get("words") or []:
                        if isinstance(w, dict):
                            flat.append(w)
                words = flat

            timings = {"total_seconds": round(time.perf_counter() - started, 4)}
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
        except Exception as exc:
            safe_error = str(exc)
            if isinstance(exc, httpx.HTTPError):
                safe_error = f"HTTP error: {exc}"
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
    import litserve as ls

    api = WhisperXLitAPI()
    return ls.LitServer(api, accelerator="auto")


if __name__ == "__main__":
    server = build_server()
    server.run(port=int(os.environ.get("PORT", "8000")))
