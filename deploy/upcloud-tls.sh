#!/usr/bin/env bash
# TLS front-end for app8.nextaura.us on UpCloud (Cloudflare Full SSL).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

APP_PORT="${APP_PORT:-80}"
DOMAIN="${DOMAIN:-app8.nextaura.us}"
CF_TOKEN_FILE="${CF_TOKEN_FILE:-/root/.cloudflare.ini}"

apt-get update -y
apt-get install -y nginx certbot python3-certbot-nginx python3-certbot-dns-cloudflare

mkdir -p /etc/letsencrypt
cat >"$CF_TOKEN_FILE" <<EOF
dns_cloudflare_api_token = ${CF_DNS_TOKEN:?missing CF_DNS_TOKEN}
EOF
chmod 600 "$CF_TOKEN_FILE"

if [[ ! -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]]; then
  certbot certonly --non-interactive --agree-tos --register-unsafely-without-email \
    --dns-cloudflare --dns-cloudflare-credentials "$CF_TOKEN_FILE" \
    -d "$DOMAIN"
fi

cat >/etc/nginx/sites-available/app8-nextaura <<NGINX
server {
    listen 443 ssl;
    server_name ${DOMAIN};

    ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/app8-nextaura /etc/nginx/sites-enabled/app8-nextaura
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx
echo "TLS_OK https://${DOMAIN}"
