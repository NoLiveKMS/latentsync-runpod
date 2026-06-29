FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV PIP_NO_CACHE_DIR=1

# System deps (includes build-essential for insightface C++ extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    curl wget ca-certificates \
    build-essential g++ gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip3 install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip3 install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone LatentSync
WORKDIR /opt
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git

WORKDIR /opt/LatentSync

# Install numpy + cython FIRST so insightface can build its C++ extension
RUN pip3 install numpy==1.26.4 cython==3.0.11

# LatentSync deps
RUN pip3 install \
    transformers==4.48.0 \
    diffusers==0.32.1 \
    accelerate==1.2.1 \
    einops==0.8.0 \
    omegaconf==2.3.0 \
    imageio==2.34.1 \
    imageio-ffmpeg==0.5.1 \
    opencv-python-headless==4.10.0.84 \
    scenedetect==0.6.5 \
    ffmpeg-python==0.2.0 \
    python-speech-features==0.6 \
    librosa==0.10.2.post1 \
    pydub==0.25.1 \
    huggingface-hub==0.26.5 \
    mediapipe==0.10.20 \
    av==14.0.1 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    decord==0.6.0 \
    kornia==0.7.3 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.19.2 \
    face-alignment==1.4.1 \
    ninja==1.11.1.1 \
    gradio==5.9.1 \
    DeepCache==0.1.1

# RunPod SDK + boto3 for R2 upload
RUN pip3 install runpod requests boto3

# Download LatentSync 1.6 models at build (~5GB)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ByteDance/LatentSync-1.6', \
    local_dir='/opt/LatentSync/checkpoints', \
    local_dir_use_symlinks=False, \
    max_workers=4)"

# Pre-download StabilityAI VAE checkpoint (~330MB)
RUN python3 -c "\
from diffusers import AutoencoderKL; \
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"


# Pre-warm insightface's buffalo_l model pack (downloads on first use otherwise)
RUN python3 -c "\
import insightface; \
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('insightface buffalo_l cached')" || true

# Pre-warm face-alignment
RUN python3 -c "\
import face_alignment; \
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu'); \
print('face_alignment cached')" || true

COPY handler.py /opt/LatentSync/handler.py

WORKDIR /opt/LatentSync
CMD ["python3", "-u", "handler.py"]
