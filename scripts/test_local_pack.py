#!/usr/bin/env python3
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vjepa_local_cache import open_hybrid_vjepa_interleaved, _row_iter, cache_path, cache_dir
from vjepa_pack import pack_vjepa_rows
from multimodal_data import vjepa_training_text

local = [
    "cbctr/cs2-10k-vjepa2-latents-300",
    "rookierufus/ego10k-vjepa-latents",
    "qinglinhou/sokoban-10k-vjepa2-tokenized",
]
p = cache_path(local[0], cache_dir())
row = next(_row_iter(p))
print("sample text len", len(vjepa_training_text(row, 20)))
t0 = time.time()
n = 0
for text, meta in pack_vjepa_rows(open_hybrid_vjepa_interleaved(local), pack_rows=128, min_chars=4000):
    n += 1
    print(f"local-only PACK {n} chars={len(text)} dt={time.time()-t0:.1f}s")
    break
print("ok")
