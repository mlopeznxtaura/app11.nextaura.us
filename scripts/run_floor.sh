#!/usr/bin/env bash
# Floor pipeline on app8 GPU: benchmark baselines → 10M smoke pretrain
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/app8-nextaura-us}"
cd "$APP_DIR"
set -a
# shellcheck disable=SC1091
source .gpu.env
set +a
# shellcheck disable=SC1091
source .venv/bin/activate

API="${API:-http://127.0.0.1:8088}"

echo "==> Step 0a: free GPU (reset in-memory session)"
curl -sf -X POST "$API/api/checkpoints/reset-session" >/dev/null

echo "==> Step 0b: benchmark public baselines + our step3898"
python3 scripts/benchmark_baselines.py --max-new-tokens 100

API="${API:-http://127.0.0.1:8088}"

echo "==> Step 1: reset + load clean 50M pretrain for reference infer"
curl -sf -X POST "$API/api/checkpoints/reset-session" >/dev/null
curl -sf -X POST "$API/api/import/checkpoint/local" \
  -H "Content-Type: application/json" \
  -d '{"name":"nextaura-50m-step3898.pt"}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('loaded',d.get('checkpoint_name'),'loss',d.get('loss'))"

echo "==> Step 2: 10M StorySupra-shaped smoke pretrain (fresh weights)"
curl -sf -X POST "$API/api/checkpoints/reset-session" >/dev/null
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
  "resume": false
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('start',d.get('running'),'params',d.get('model_params'),'step',d.get('train_step'))"

echo "==> floor pipeline launched — poll /api/status on app8"
