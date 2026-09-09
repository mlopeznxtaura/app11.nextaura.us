#!/usr/bin/env bash
# Legacy DO droplet deploy — superseded by ../../scripts/deploy-app8-us-train1.ps1 (UpCloud train-1).
set -euo pipefail

DROPLET="${DROPLET:-root@107.170.43.172}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/app2-do-deploy}"
if [[ ! -f "$SSH_KEY" ]]; then
  SSH_KEY="$HOME/.ssh/id_ed25519"
fi
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
APP_DIR="/opt/app8-nextaura-us"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARBALL="/tmp/app8-nextaura-us.tgz"

echo "==> pack (no .venv, no secrets in tarball)"
tar czf "$TARBALL" -C "$ROOT" \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude .gpu.env \
  --exclude '*.pyc' \
  server.py train_engine.py train_gpu.py model.py scalar_features.py convert_model.py agent_contract.py run_history.py reward_eval.py data_jsonl.py multimodal_data.py hf_datasets.py tokenizer_utils.py text_quality.py intelligence_expand.py requirements.txt bootstrap_gpu.sh scripts data/benchmark-prompts.jsonl \
  public deploy worker.js wrangler.toml index.html 2>/dev/null || \
tar czf "$TARBALL" -C "$ROOT" \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude .gpu.env \
  server.py train_engine.py train_gpu.py model.py scalar_features.py convert_model.py agent_contract.py run_history.py reward_eval.py data_jsonl.py multimodal_data.py hf_datasets.py tokenizer_utils.py text_quality.py intelligence_expand.py requirements.txt bootstrap_gpu.sh scripts data/benchmark-prompts.jsonl \
  public deploy worker.js

echo "==> upload to $DROPLET (key: $SSH_KEY)"
ssh "${SSH_OPTS[@]}" "$DROPLET" "mkdir -p $APP_DIR"
scp "${SSH_OPTS[@]}" "$TARBALL" "$DROPLET:/tmp/app2.tgz"
scp "${SSH_OPTS[@]}" "$ROOT/.gpu.env" "$DROPLET:$APP_DIR/.gpu.env"
ssh "${SSH_OPTS[@]}" "$DROPLET" "tar xzf /tmp/app2.tgz -C $APP_DIR && chmod +x $APP_DIR/deploy/do-install.sh && bash $APP_DIR/deploy/do-install.sh"

echo "==> done — test: curl http://107.170.43.172:8080/api/health"
