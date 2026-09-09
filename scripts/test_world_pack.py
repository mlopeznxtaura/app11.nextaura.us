#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model_mix import VJEPA_MIX_WORLD
from vjepa_local_cache import open_hybrid_vjepa_interleaved
from vjepa_pack import pack_vjepa_rows

token = os.environ.get("HF_TOKEN")
t0 = time.time()
n = 0
for text, meta in pack_vjepa_rows(
    open_hybrid_vjepa_interleaved(VJEPA_MIX_WORLD, token=token),
    pack_rows=128,
    min_chars=4000,
):
    n += 1
    print(f"PACK {n} chars={len(text)} dt={time.time()-t0:.1f}s")
    if n >= 2:
        break
print(f"done packs={n}")
