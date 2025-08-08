#!/usr/bin/env bash
set -euo pipefail

# 1) Set root password from env (default provided in Dockerfile)
echo "root:${SSH_PASSWORD:-runpod}" | chpasswd

# 2) Start SSH service in foreground later; ensure host keys exist
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
  ssh-keygen -A
fi

# 3) Configure Git (idempotent)
git config --global user.name "great-luao" || true
git config --global user.email "luao@shanghaitech.edu.cn" || true

# 4) Load SSH key from workspace if present
KEY_PATH="/workspace/.ssh/id_github"
if [ -f "$KEY_PATH" ]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  cp "$KEY_PATH" /root/.ssh/id_github
  chmod 600 /root/.ssh/id_github
  eval "$(ssh-agent -s)"
  ssh-add /root/.ssh/id_github || true
  # Ensure GitHub known_hosts exists
  ssh-keyscan -T 10 github.com >> /root/.ssh/known_hosts 2>/dev/null || true
  echo "Loaded SSH key from $KEY_PATH"
else
  echo "No SSH key found at $KEY_PATH (skipping ssh-add)"
fi

# 5) Optionally start Jupyter Notebook if enabled via env
if [[ "${START_JUPYTER:-false}" == "true" ]]; then
  JUPYTER_TOKEN="${JUPYTER_TOKEN:-runpod}"
  JUPYTER_PORT="${JUPYTER_PORT:-8888}"
  echo "Starting Jupyter Notebook on 0.0.0.0:${JUPYTER_PORT} (token: ${JUPYTER_TOKEN})"
  nohup jupyter notebook --ip=0.0.0.0 --port=${JUPYTER_PORT} \
        --no-browser --NotebookApp.token=${JUPYTER_TOKEN} --NotebookApp.allow_root=True \
        >/var/log/jupyter.log 2>&1 &
fi

# 6) Start sshd in foreground (container stays alive)
exec /usr/sbin/sshd -D


