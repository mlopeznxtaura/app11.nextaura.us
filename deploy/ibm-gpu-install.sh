#!/usr/bin/env bash
# Install app7/app8 training lab on IBM GPU VSI (CUDA).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

APP_SLUG="${APP_SLUG:-app7}"
APP_DIR="${APP_DIR:-/opt/${APP_SLUG}-nextaura-us}"
PORT="${PORT:-80}"
HEADLESS="${HEADLESS:-0}"

echo "==> GPU check"
nvidia-smi || { echo "ERROR: no GPU"; exit 1; }

cd "$APP_DIR"
mkdir -p checkpoints data

if [[ ! -f .gpu.env ]]; then
  echo "ERROR: missing $APP_DIR/.gpu.env"
  exit 1
fi
chmod 600 "$APP_DIR/.gpu.env"

echo "==> venv + CUDA torch"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt

set -a
source .gpu.env
set +a
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

SVC="${APP_SLUG}-nextaura"
DESC="${APP_SLUG} GPU training lab"
EXTRA_ENV=""
if [[ "$HEADLESS" == "1" ]]; then
  DESC="${APP_SLUG} GPU headless training"
  EXTRA_ENV=$'Environment=HEADLESS=1\nEnvironment=AUTO_RESUME=1\n'
fi
cat >/etc/systemd/system/${SVC}.service <<UNIT
[Unit]
Description=${DESC}
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.gpu.env
Environment=TRAIN_DEVICE=cuda
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${EXTRA_ENV}ExecStart=$APP_DIR/.venv/bin/uvicorn server:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "$SVC"
systemctl restart "$SVC"
sleep 3
curl -sf "http://127.0.0.1:$PORT/api/health" | head -c 400 || true
echo
echo "INSTALL_OK ${APP_SLUG} on GPU port $PORT"
