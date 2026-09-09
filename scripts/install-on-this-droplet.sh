#!/usr/bin/env bash
# Run ON the DigitalOcean droplet console (you are already root there).
# Do NOT run on your Mac.
set -euo pipefail

APP_DIR=/opt/app2-nextaura-us
TARBALL_URL='https://s3.us-south.cloud-object-storage.appdomain.cloud/nextaura-fineweb-stage1/deploy/app2-nextaura-us.tgz'

echo "==> GPU check"
nvidia-smi -L || echo "WARN: no GPU yet"

echo "==> download app"
mkdir -p "$APP_DIR"
curl -fsSL "$TARBALL_URL" -o /tmp/app2.tgz
tar xzf /tmp/app2.tgz -C "$APP_DIR"
chmod +x "$APP_DIR/deploy/do-install.sh"

if [[ ! -f "$APP_DIR/.gpu.env" ]]; then
  echo "ERROR: missing $APP_DIR/.gpu.env"
  echo "On your Mac, run:"
  echo "  scp ~/.cursor/skills/ibmcloud-site-blitz/app2-nextaura-us/.gpu.env root@107.170.43.172:$APP_DIR/.gpu.env"
  echo "Or paste the full install script from: bash scripts/generate-console-full-install.sh"
  exit 1
fi

bash "$APP_DIR/deploy/do-install.sh"
echo "==> test"
curl -sf http://127.0.0.1:8080/api/health | head -c 400
echo
echo "INSTALL_OK"
