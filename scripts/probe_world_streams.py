#!/usr/bin/env python3
"""Probe world-model HF streams for MCF v2 mix."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_stream_loader import probe_hf_train_stream
from world_model_mix import VJEPA_MIX_WORLD

token = os.environ.get("HF_TOKEN")
for ds in VJEPA_MIX_WORLD:
    try:
        p = probe_hf_train_stream(ds, kind="vjepa", token=token, max_rows=150)
        if p.get("ok"):
            prev = str(p.get("text_preview") or "")[:100]
            print(f"OK  {ds}  rows={p.get('rows_scanned')}  keys={p.get('keys', [])[:8]}  preview={prev!r}")
        else:
            print(f"BAD {ds}  scanned={p.get('rows_scanned')}  err={p.get('error', p)}")
    except Exception as exc:
        print(f"ERR {ds}  {exc}")
