#!/usr/bin/env python3
"""Download MCF v2 V-JEPA mix to local NVMe and mirror to IBM COS."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vjepa_local_cache import VJEPA_CACHE_DIR, cache_dir, cache_ready, ensure_vjepa_cache
from world_model_mix import DEFAULT_VJEPA_MIX

VJEPA_MIX_HLS = DEFAULT_VJEPA_MIX


def main() -> None:
    ids = VJEPA_MIX_HLS[:]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — HF download may fail", file=sys.stderr)
    print(f"Cache dir: {cache_dir()} (VJEPA_CACHE_DIR={VJEPA_CACHE_DIR})")
    if cache_ready(ids):
        print("Local cache already complete.")
    result = ensure_vjepa_cache(ids, hf_token=token, cos_sync=True)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("ok"):
        raise SystemExit(1)
    if not cache_ready(ids):
        raise SystemExit("Cache incomplete after ensure_vjepa_cache")


if __name__ == "__main__":
    main()
