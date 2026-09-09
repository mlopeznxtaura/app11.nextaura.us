#!/usr/bin/env bash
# Run on DigitalOcean droplet after app tarball is in /opt/app2-nextaura-us
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

APP_DIR="${APP_DIR:-/opt/app8-nextaura-us}"
PORT="${PORT:-8088}"

echo "==> OS + Python"
apt-get update -y
apt-get install -y python3 python3-pip python3-venv gcc curl

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected"
  nvidia-smi || true
else
  echo "WARN: nvidia-smi not found — training will use CPU unless you attach a GPU droplet."
fi

cd "$APP_DIR"
mkdir -p checkpoints data

if [[ ! -f .gpu.env ]]; then
  echo "ERROR: missing $APP_DIR/.gpu.env — create it with HF_TOKEN, IBM_CLOUD_API_KEY, COS_BUCKET, TARGET_GB"
  exit 1
fi

chown root:root "$APP_DIR/.gpu.env"
chmod 600 "$APP_DIR/.gpu.env"

echo "==> venv + deps"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

echo "==> smoke import"
set -a
source .gpu.env
set +a
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> systemd unit"
if command -v ufw >/dev/null 2>&1; then
  ufw allow OpenSSH >/dev/null 2>&1 || true
  ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
fi

cat >/etc/systemd/system/app8-nextaura.service <<UNIT
[Unit]
Description=app8.nextaura.us multimodal fine-tune
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.gpu.env
Environment=TRAIN_DEVICE=cuda
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStart=$APP_DIR/.venv/bin/uvicorn server:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable app8-nextaura
systemctl restart app8-nextaura

sleep 2
curl -sf "http://127.0.0.1:$PORT/api/health" | head -c 500 || true
echo
echo "INSTALL_OK — service app8-nextaura on port $PORT"
