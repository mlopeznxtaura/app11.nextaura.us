#!/usr/bin/env bash
# Start Muse infer sidecar on leftover VRAM. Does not stop sft_qlora.py.
set -euo pipefail
if [ -f /opt/app9-nextaura-us/.gpu.env ]; then
  set -a
  # shellcheck disable=SC1091
  . /opt/app9-nextaura-us/.gpu.env
  set +a
fi
export HF_HOME="${HF_HOME:-/data/datasets}"
# Prefer the GPU with more free memory so we do not crowd the hot train GPU.
free0=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1)}')
free1=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1)}')
if [ "$free1" -gt "$free0" ]; then
  export CUDA_VISIBLE_DEVICES=1
else
  export CUDA_VISIBLE_DEVICES=0
fi
export MUSE_INFER_PORT="${MUSE_INFER_PORT:-8766}"
pkill -f '/data/muse/infer_server.py' 2>/dev/null || true
sleep 1
nohup /data/muse-venv/bin/python -u /data/muse/infer_server.py >> /tmp/muse_infer.log 2>&1 &
echo "MUSE_INFER_PID=$! gpu=$CUDA_VISIBLE_DEVICES log=/tmp/muse_infer.log"
