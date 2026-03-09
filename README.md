# whisperx-lightning

WhisperX + LitServe container for async ASR jobs, published via GitHub Container Registry (GHCR) and deployed in Lightning by directly importing GHCR image tags from the Lightning dashboard.

## What is included

- `lightning_asr/` service code
- `lightning_asr/Dockerfile` for CUDA + WhisperX runtime
- `.github/workflows/build-push-ghcr.yml` to publish images to GHCR
- `.env.example` with public-safe placeholders only

## API contract (high level)

The service accepts a transcription request and returns:

```json
{"job_id":"<uuid>","status":"accepted"}
```

The final result is posted to `webhook_url` with either:

- `status: "succeeded"` with `segments`, optional `words`, `timings`
- `status: "failed"` with `error`

## Build image locally

```bash
docker build -t whisperx-lightning:local -f lightning_asr/Dockerfile .
```

The Docker build now copies `requirements.txt` and the model downloader before the rest of `lightning_asr/`, so ordinary service code changes keep the dependency and model layers cached.

## Local GPU smoke test

From the repository root, start the LitServe container with Docker GPU passthrough:

```bash
docker compose -f docker-compose.gpu.yml up --build
```

For a one-off container run:

```bash
docker run --rm --gpus all -p 8000:8000 whisperx-lightning:local
```

Startup logs should show:

- `event="gpu_diagnostics"` with `cuda_available=true`
- `device="cuda"` during setup
- `compute_type="float16"` during model load

## Runtime configuration

The container defaults to GPU-first LitServe settings. Override these env vars if you need a different deployment shape:

| Variable | Default | Description |
| --- | --- | --- |
| `LITSERVE_ACCELERATOR` | `cuda` | LitServe accelerator selection. Use `cuda` for GPU workers and override to `auto` or `cpu` only for debugging. |
| `LITSERVE_DEVICES` | `auto` | Number of devices LitServe should bind to, or `auto` to let LitServe detect the visible GPUs. |
| `LITSERVE_WORKERS_PER_DEVICE` | `1` | Worker processes per device. Keep this low for WhisperX because model memory usage is high. |
| `LITSERVE_MAX_BATCH_SIZE` | `1` | LitServe request batching. The service already batches audio chunks internally, so the external request batch stays at `1`. |
| `LITSERVE_BATCH_TIMEOUT` | `0.0` | LitServe batch wait in seconds before dispatching a request batch. |
| `LITSERVE_TIMEOUT` | `false` | Disables LitServe request timeout so long-running transcriptions are accepted cleanly. |
| `WHISPERX_COMPUTE_TYPE` | `auto` | Defaults to `float16` on CUDA and `int8` on CPU. Set explicitly only when debugging model compatibility. |
| `WHISPERX_MODEL` | `large-v3-turbo` | Startup model used for warmup. Requests can override it, but staying on one model avoids reload churn. |

## Publish flow (GitHub -> GHCR)

1. Push to `main` (or create a version tag like `v1.0.0`).
2. GitHub Actions workflow `Build and Push GHCR` builds and pushes:
   - `ghcr.io/<owner>/whisperx-lightning:latest` (on default branch)
   - `ghcr.io/<owner>/whisperx-lightning:sha-<...>`
   - `ghcr.io/<owner>/whisperx-lightning:vX.Y.Z` (for tags)

## Lightning import from GHCR (no relay script)

Use Lightning dashboard to import your GHCR image directly:

1. Open Lightning dashboard -> create/update your app.
2. Choose container image source and enter:
   - `ghcr.io/<owner>/whisperx-lightning:<tag>`
3. If image/package is private, provide GHCR credentials (PAT with package read access).
4. Set runtime environment variables from `.env.example` as needed.
5. Deploy and verify app health.

## Required permissions and visibility

- GitHub Actions must have `packages: write` permission (already set in workflow).
- For private GHCR package imports in Lightning, use a PAT with at least:
  - `read:packages`
- For public GHCR package imports, authentication may not be required, depending on Lightning settings.

## 18GB image notes

- First build/push/pull can be slow; expect long transfer times.
- Prefer immutable SHA tags for reproducible deploys.
- Keep `requirements.txt` and `lightning_asr/download_models.py` stable for better build cache reuse.
- If cost/latency becomes an issue, follow up with a multi-stage slimming pass.
