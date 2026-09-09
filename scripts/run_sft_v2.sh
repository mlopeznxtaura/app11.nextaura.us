#!/usr/bin/env bash
# SFT v2: 3898 base · 2 epoch cycles · lr 1e-5 (no GB hell)
set -euo pipefail
API="${API:-http://127.0.0.1:8088}"

curl -sf -X POST "$API/api/import/checkpoint/local" \
  -H "Content-Type: application/json" \
  -d '{"name":"nextaura-50m-step3898.pt"}' >/dev/null

curl -sf -X POST "$API/api/start" -H "Content-Type: application/json" -d '{
  "data_source": "sft-intelligence",
  "target_gb": 0.05,
  "learning_rate": 0.00001,
  "batch_size": 1024,
  "epoch_cycles": true,
  "epochs": 2,
  "warmup_steps": 50,
  "save_every_n_steps": 50,
  "total_train_steps": 500,
  "resume": false
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('sft_v2',d.get('running'),d.get('status'),'step',d.get('train_step'))"
