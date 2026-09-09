#!/usr/bin/env bash
# Launch Muse-Glimmer coding SFT v2 on app9. GUI playbook: finetune lr 1e-5, 2 epochs, collapse stop.
set -euo pipefail
if [ -f /opt/app9-nextaura-us/.gpu.env ]; then
  set -a
  # shellcheck disable=SC1091
  . /opt/app9-nextaura-us/.gpu.env
  set +a
fi
export MUSE_BASE="${MUSE_BASE:-/data/models/Muse-Glimmer-30B}"
export MUSE_OUT="${MUSE_OUT:-/data/muse/sft-v2}"
export MUSE_TRAIN_LOG="${MUSE_TRAIN_LOG:-/tmp/muse_phase2.log}"
export HF_HOME="${HF_HOME:-/data/datasets}"
mkdir -p "$MUSE_OUT"
cd /data/muse
nohup /data/muse-venv/bin/python -u /data/muse/sft_qlora.py \
  --lr 1e-5 --epochs 2 --steps 800 --rows 20000 --seq 8192 \
  >> "$MUSE_TRAIN_LOG" 2>&1 &
echo "MUSE_SFT_V2_PID=$! log=$MUSE_TRAIN_LOG out=$MUSE_OUT"
