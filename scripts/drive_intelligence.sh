#!/usr/bin/env bash
# Autonomous floor pipeline: cleanup → benchmarks → 10M smoke pretrain
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/app8-nextaura-us}"
API="${API:-http://127.0.0.1:8088}"
LOG="${LOG:-/tmp/drive_intelligence.log}"
exec > >(tee -a "$LOG") 2>&1

cd "$APP_DIR"
set -a
# shellcheck disable=SC1091
source .gpu.env
set +a
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> $(date -Is) drive_intelligence start"

KEEP="nextaura-50m-step3898.pt"
for f in checkpoints/*.pt; do
  base=$(basename "$f")
  if [[ "$base" != "$KEEP" && "$base" != imported-* ]]; then
    rm -f "$f" && echo "purged $base"
  fi
done

echo "==> floor benchmarks (StorySupra + Supra-50M + our 3898)"
python3 scripts/benchmark_baselines.py --max-new-tokens 100

echo "==> restore clean 50M base"
curl -sf -X POST "$API/api/checkpoints/reset-session" >/dev/null
curl -sf -X POST "$API/api/import/checkpoint/local" \
  -H "Content-Type: application/json" \
  -d '{"name":"nextaura-50m-step3898.pt"}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('loaded',d.get('checkpoint_name'),'loss',d.get('loss'))"

echo "==> 10M StorySupra-shaped smoke pretrain"
curl -sf -X POST "$API/api/checkpoints/reset-session" >/dev/null
sleep 2
curl -sf -X POST "$API/api/start" -H "Content-Type: application/json" -d '{
  "data_source": "mixed",
  "target_gb": 0.08,
  "learning_rate": 0.0003,
  "batch_size": 1024,
  "n_layer": 8,
  "n_head": 8,
  "n_embd": 256,
  "block_size": 256,
  "vocab_size": 8192,
  "ffn_dim": 1024,
  "tokenizer_id": "SupraLabs/StorySupra-10M",
  "mix_local_jsonl": false,
  "mix_local_corpora": false,
  "hf_stream_fable": true,
  "hf_stream_sol": false,
  "hf_stream_kimi": false,
  "hf_stream_nemotron": false,
  "hf_stream_unsolved_math": false,
  "hf_stream_preference": false,
  "hf_stream_fineweb": true,
  "epoch_cycles": false,
  "warmup_steps": 50,
  "save_every_n_steps": 100,
  "total_train_steps": 500,
  "resume": false,
  "apply": {
    "n_layer": true, "n_head": true, "n_embd": true, "block_size": true,
    "vocab_size": true, "ffn_dim": true, "tokenizer_id": true,
    "learning_rate": true, "batch_size": true, "target_gb": true
  }
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('start ok',d.get('ok'),'running',d.get('running'),'status',d.get('status'))"

sleep 8
curl -s "$API/api/status" | python3 -c "import sys,json; s=json.load(sys.stdin); print('train',s.get('running'),'step',s.get('train_step'),'loss',s.get('last_loss'),'params',s.get('model_params'),'err',s.get('error'))"

echo "==> $(date -Is) drive_intelligence done — log $LOG"
