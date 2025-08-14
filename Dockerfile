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

# Install Python dependencies (inside venv)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && \
    rm /tmp/requirements.txt

# Install Jupyter for optional web access
RUN pip install --no-cache-dir jupyter

# Install flash attn with right version: 2.8.0 post 2
RUN pip install flash-attn==2.8.0.post2 --no-build-isolation

# Configure Warp terminal integration
RUN echo 'printf '"'"'\eP$f{"hook": "SourcedRcFileForWarp", "value": { "shell": "bash"}}\x9c'"'"'' >> ~/.bashrc && \
    echo "Warp terminal integration configured"

# Expose SSH port
EXPOSE 22

# Configure Git and SSH known_hosts (safe defaults; no private keys baked into image)
RUN git config --global user.name "great-luao" && \
    git config --global user.email "luao@shanghaitech.edu.cn" && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    ssh-keyscan -T 10 github.com >> /root/.ssh/known_hosts

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
