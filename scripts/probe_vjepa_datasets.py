#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_stream_loader import probe_hf_train_stream

CANDIDATES = [
    "qinglinhou/sokoban-10k-vjepa2-tokenized",
    "kushagrabaingaha/bmd-vjepa2-features",
    "rookierufus/sokoban-10k-vjepa-latents",
    "phi-9/epic-kitchens-vjepa",
    "rookierufus/epic-kitchens-vjepa",
    "Adjimavo/libero_world_latents_vjepa_m4",
]

token = os.environ.get("HF_TOKEN")
for ds in CANDIDATES:
    try:
        p = probe_hf_train_stream(ds, kind="vjepa", token=token)
        if p.get("ok"):
            prev = str(p.get("text_preview") or "")[:72]
            print(f"OK  {ds}  preview={prev!r}")
        else:
            print(f"BAD {ds}  {p.get('error', p)}")
    except Exception as exc:
        print(f"ERR {ds}  {exc}")
