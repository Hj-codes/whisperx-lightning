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
- Keep model download layers stable for better build cache reuse.
- If cost/latency becomes an issue, follow up with a multi-stage slimming pass.
