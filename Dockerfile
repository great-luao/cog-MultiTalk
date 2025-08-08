# RunPod optimized Dockerfile for MultiTalk
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# Set model cache directory
ENV MODEL_CACHE=/workspace/weights
ENV BASE_URL=https://weights.replicate.delivery/default/multitalk/weights/
ENV HF_HOME=/workspace/weights
ENV TORCH_HOME=/workspace/weights
ENV HF_DATASETS_CACHE=/workspace/weights
ENV TRANSFORMERS_CACHE=/workspace/weights
ENV HUGGINGFACE_HUB_CACHE=/workspace/weights

# Install system dependencies and add deadsnakes PPA (no empty continuation lines)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    ninja-build \
    pkg-config && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.12 (separate RUN; no trailing backslash)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pget for fast model downloads
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)" && \
    chmod +x /usr/local/bin/pget

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && \
    pip install flash-attn --no-build-isolation && \
    rm /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["/bin/bash"]
