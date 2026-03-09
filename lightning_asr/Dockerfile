FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="12.8"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --requirement /app/requirements.txt

RUN mkdir -p /app/lightning_asr
COPY lightning_asr/__init__.py /app/lightning_asr/__init__.py
COPY lightning_asr/download_models.py /app/lightning_asr/download_models.py

ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV XDG_CACHE_HOME=/app/models
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# Fix PyTorch 2.6+ weights_only issue during model download
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
# Use float32 during build (no GPU in Docker build), runtime will use GPU with float16
RUN WHISPERX_COMPUTE_TYPE=float32 python3 -m lightning_asr.download_models

COPY lightning_asr /app/lightning_asr

ENV PORT=8000
# Ensure CUDA libraries are in the path for GPU detection
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LITSERVE_ACCELERATOR=cuda
ENV LITSERVE_DEVICES=auto
ENV LITSERVE_WORKERS_PER_DEVICE=1
ENV LITSERVE_MAX_BATCH_SIZE=1
ENV LITSERVE_BATCH_TIMEOUT=0.0
ENV LITSERVE_TIMEOUT=false
CMD ["python3", "-m", "lightning_asr.litserve_app"]
