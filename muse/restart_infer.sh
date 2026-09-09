#!/usr/bin/env bash
set -euo pipefail
python3 /tmp/convert_lf.py
sudo cp /tmp/infer_server.lf.py /data/muse/infer_server.py
ps -eo pid=,cmd= | grep muse-venv | grep infer_server | awk '{print $1}' | xargs -r kill
sleep 2
export HF_HOME=/data/datasets
export CUDA_VISIBLE_DEVICES=1
export MUSE_INFER_PORT=8766
nohup /data/muse-venv/bin/python -u /data/muse/infer_server.py >> /tmp/muse_infer.log 2>&1 &
echo "restarted pid=$! gpu=1"
