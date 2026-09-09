#!/usr/bin/env bash
# app9 server install — run on app9-train after code tarball is uploaded
set -euo pipefail
cd /opt/app9-nextaura-us
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q fastapi==0.115.6 "uvicorn[standard]==0.32.1" pydantic datasets==3.2.0 ibm-cos-sdk==2.14.3 tiktoken==0.8.0 python-multipart==0.0.20 transformers==4.48.0
# torch already installed system-wide; venv needs access
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 || true
mkdir -p /opt/app9-nextaura-us/checkpoints /opt/app9-nextaura-us/data
ln -sfn /data/checkpoints /opt/app9-nextaura-us/checkpoints
ln -sfn /data/datasets /opt/app9-nextaura-us/data/hf_cache
cp /data/app9.gpu.env /opt/app9-nextaura-us/.gpu.env
cat > /etc/systemd/system/app9-nextaura.service <<'UNIT'
[Unit]
Description=app9 GPU headless training
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/app9-nextaura-us
EnvironmentFile=/opt/app9-nextaura-us/.gpu.env
ExecStart=/opt/app9-nextaura-us/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 80
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now app9-nextaura
sleep 5
curl -s http://localhost/api/health || curl -s http://localhost/api/status | head -c 200
echo APP9_SERVER_OK
