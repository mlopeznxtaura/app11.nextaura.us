#!/usr/bin/env bash
# Paste this entire script into the DigitalOcean Droplet Console (as root).
set -euo pipefail

DEPLOY_PUB='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDJBVKXZ25KOvNWT2gik4M+zPlwX9N8TaZK/GffrYmrl app2-do-deploy'
USER_PUB='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMnGCS2Cn39cp/Z8H+zoSs/J9w+MmqYMywhc3B49AOg+ mlopeznxtaura@Marcos-MacBook-Air.local'
TARBALL_URL='https://s3.us-south.cloud-object-storage.appdomain.cloud/nextaura-fineweb-stage1/deploy/app2-nextaura-us.tgz'
APP_DIR=/opt/app2-nextaura-us

mkdir -p ~/.ssh "$APP_DIR"
chmod 700 ~/.ssh
for KEY in "$DEPLOY_PUB" "$USER_PUB"; do
  grep -qF "$KEY" ~/.ssh/authorized_keys 2>/dev/null || echo "$KEY" >> ~/.ssh/authorized_keys
done
chmod 600 ~/.ssh/authorized_keys

curl -fsSL "$TARBALL_URL" -o /tmp/app2.tgz
mkdir -p "$APP_DIR"
tar xzf /tmp/app2.tgz -C "$APP_DIR"
chmod +x "$APP_DIR/deploy/do-install.sh"

echo "BOOTSTRAP_OK — deploy key added, app extracted to $APP_DIR"
echo "Next: agent will scp .gpu.env and run do-install.sh"
