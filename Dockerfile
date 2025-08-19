# RunPod optimized Dockerfile for MultiTalk
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

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

# Install pget for fast model downloads
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)" && \
    chmod +x /usr/local/bin/pget

# Install tmux
RUN apt-get update && apt-get install -y --no-install-recommends tmux && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.12 (separate RUN; no trailing backslash)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install and configure OpenSSH server (and client) for SSH access
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server openssh-client && \
rm -rf /var/lib/apt/lists/* && \
mkdir -p /var/run/sshd && \
sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# 创建 venv（不带 pip）
RUN python3.12 -m venv --without-pip /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# 在 venv 内安装/升级 pip 工具链
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | /opt/venv/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies (inside venv, include torch)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install Jupyter for optional web access
RUN pip install --no-cache-dir jupyter

# Install flash attn via prebuilt wheel (Py3.12 / Torch 2.7 / CUDA 12.x); fallback to source build if incompatible
RUN wget -O /tmp/flash_attn.whl \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    (pip install --no-cache-dir /tmp/flash_attn.whl && rm -f /tmp/flash_attn.whl) || \
    (echo "Prebuilt flash_attn wheel not found/compatible, falling back to source build..." && \
     rm -f /tmp/flash_attn.whl && \
     export MAX_JOBS=1 && \
     pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir)

# Switch to bash for nvm-related steps
SHELL ["/bin/bash", "-lc"]

# Install Node.js (v22) via nvm and install Claude Code
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    source "$NVM_DIR/nvm.sh" && \
    nvm install 22 && nvm alias default 22 && nvm use default && \
    ln -sf "$NVM_DIR/versions/node/$(nvm version)/bin/node" /usr/local/bin/node && \
    ln -sf "$NVM_DIR/versions/node/$(nvm version)/bin/npm" /usr/local/bin/npm && \
    ln -sf "$NVM_DIR/versions/node/$(nvm version)/bin/npx" /usr/local/bin/npx && \
    npm install -g @anthropic-ai/claude-code && \
    node -v && npm -v

# Configure Warp terminal integration
RUN echo 'printf '"'"'\eP$f{"hook": "SourcedRcFileForWarp", "value": { "shell": "bash"}}\x9c'"'"'' >> ~/.bashrc && \
    echo "Warp terminal integration configured"

# Expose SSH port
EXPOSE 22

# Expose port for potential API usage or HTMLS
EXPOSE 8000 7860 8888

# Set working directory
WORKDIR /workspace

# Default root password (override via env at runtime)
ENV SSH_PASSWORD=runpod

# Use startup script to initialize environment and start sshd
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh
CMD ["/usr/local/bin/startup.sh"]
