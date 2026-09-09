#!/usr/bin/env bash
# IBM GPU VSI: NVIDIA driver + app training lab (CUDA uvicorn on port 80).
set -euo pipefail
exec > /var/log/app-gpu-bootstrap.log 2>&1
export DEBIAN_FRONTEND=noninteractive

APP_SLUG="${APP_SLUG:-app7}"
APP_DIR="/opt/${APP_SLUG}-nextaura-us"
PORT="${PORT:-80}"

echo "=== ${APP_SLUG} GPU bootstrap ==="
apt-get update -y
apt-get install -y python3 python3-pip python3-venv gcc curl git

if ! command -v nvidia-smi &>/dev/null; then
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers install -g --no-install-recommends || apt-get install -y nvidia-driver-535-server
fi
nvidia-smi || true

mkdir -p "$APP_DIR"
echo "BOOTSTRAP_OK ${APP_SLUG} — awaiting app deploy via SSH"
