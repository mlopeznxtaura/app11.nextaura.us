#!/usr/bin/env bash
# Run on Mac after pasting console-bootstrap.sh into the DO droplet console.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SSH_KEY="${SSH_KEY:-$HOME/.ssh/app2-do-deploy}"
bash "$ROOT/scripts/push-to-droplet.sh"

echo "==> verify droplet health"
curl -sf --max-time 15 http://107.170.43.172:8080/api/health | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('hf_token_loaded') is True; print('droplet_health_ok', d.get('device'))"

echo "==> cutover Cloudflare worker"
ORIGIN='http://107.170.43.172:8080'
perl -pi -e "s|^const ORIGIN = .*|const ORIGIN = \"$ORIGIN\";|" "$ROOT/worker.js"
(cd "$ROOT" && npx --yes wrangler deploy)

echo "DONE — https://app2.nextaura.us"
