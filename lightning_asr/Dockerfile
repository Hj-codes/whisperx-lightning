FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY lightning_asr /app/lightning_asr

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchaudio==2.8.0+cu128
RUN python3 -m pip install litserve==0.2.17 httpx==0.27.2 pydantic==2.8.2 whisperx==3.7.7 onnxruntime-gpu==1.19.2

ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV XDG_CACHE_HOME=/app/models
# Fix PyTorch 2.6+ weights_only issue during model download
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
# Use float32 during build (no GPU in Docker build), runtime will use GPU with float16
RUN WHISPERX_COMPUTE_TYPE=float32 python3 -m lightning_asr.download_models

ENV PORT=8000
# Ensure CUDA libraries are in the path for GPU detection
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["python3", "-m", "lightning_asr.litserve_app"]
