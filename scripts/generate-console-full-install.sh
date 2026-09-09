#!/usr/bin/env bash
# Generates a one-time paste script for DO console (writes /tmp/do-console-install.sh only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/tmp/do-console-install.sh
DEPLOY_PUB="$(cat "$HOME/.ssh/app2-do-deploy.pub")"
TARBALL_URL='https://s3.us-south.cloud-object-storage.appdomain.cloud/nextaura-fineweb-stage1/deploy/app2-nextaura-us.tgz'

cat >"$OUT" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
APP_DIR=/opt/app2-nextaura-us
mkdir -p ~/.ssh "\$APP_DIR"
chmod 700 ~/.ssh
grep -qF '${DEPLOY_PUB}' ~/.ssh/authorized_keys 2>/dev/null || echo '${DEPLOY_PUB}' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
curl -fsSL '${TARBALL_URL}' -o /tmp/app2.tgz
tar xzf /tmp/app2.tgz -C "\$APP_DIR"
SCRIPT

cat >>"$OUT" <<SCRIPT
cat >"\$APP_DIR/.gpu.env" <<'ENVEOF'
SCRIPT
cat "$ROOT/.gpu.env" >>"$OUT"
cat >>"$OUT" <<'SCRIPT'
ENVEOF
chown root:root "$APP_DIR/.gpu.env"
chmod 600 "$APP_DIR/.gpu.env"
chmod +x "$APP_DIR/deploy/do-install.sh"
bash "$APP_DIR/deploy/do-install.sh"
SCRIPT

chmod 700 "$OUT"
echo "Wrote $OUT ($(wc -c <"$OUT") bytes) — paste into DO console only, do not commit"
