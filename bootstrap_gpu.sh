#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip python3-venv gcc linux-headers-generic nvidia-driver-550 nvidia-utils-550
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install --no-cache-dir datasets==3.2.0 ibm-cos-sdk==2.14.3 tiktoken==0.8.0
echo BOOTSTRAP_OK
