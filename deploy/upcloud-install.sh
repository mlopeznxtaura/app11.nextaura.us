#!/usr/bin/env bash
# Install app8.nextaura.us GUI on UpCloud train-1 (CPU, port 80 for firewall).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

APP_DIR="${APP_DIR:-/opt/app8-nextaura-us}"
PORT="${PORT:-80}"

echo "==> OS + Python"
apt-get update -y
apt-get install -y python3 python3-pip python3-venv gcc curl

cd "$APP_DIR"
mkdir -p checkpoints data

if [[ ! -f .gpu.env ]]; then
  echo "ERROR: missing $APP_DIR/.gpu.env"
  exit 1
fi

chown root:root "$APP_DIR/.gpu.env"
chmod 600 "$APP_DIR/.gpu.env"

echo "==> venv + deps (CPU torch)"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt

echo "==> smoke import"
set -a
source .gpu.env
set +a
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> systemd unit"
cat >/etc/systemd/system/app8-nextaura.service <<UNIT
[Unit]
Description=app8.nextaura.us multimodal fine-tune
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.gpu.env
Environment=TRAIN_DEVICE=cpu
ExecStart=$APP_DIR/.venv/bin/uvicorn server:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable app8-nextaura
systemctl restart app8-nextaura

sleep 3
curl -sf "http://127.0.0.1:$PORT/api/health" | head -c 500 || true
echo
echo "INSTALL_OK — service app8-nextaura on port $PORT"
