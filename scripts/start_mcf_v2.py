#!/usr/bin/env python3
"""Start MCF v2 HLS (latent packing) on app8 — uses .venv on GPU host."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

API = os.environ.get("API", "http://127.0.0.1").rstrip("/")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model_mix import DEFAULT_VJEPA_MIX


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{API}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


APPLY_ALL = {k: True for k in (
    "n_layer", "n_head", "n_embd", "block_size", "vocab_size", "ffn_dim", "tokenizer_id",
    "learning_rate", "batch_size", "micro_batch_size", "grad_accum_steps", "target_gb",
    "total_train_steps", "scalar_input_projection", "gradient_checkpointing", "warmup_steps",
    "mixed_precision", "lr_schedule", "save_every_n_steps", "use_flash_attention",
    "hf_streams", "mix_local_jsonl", "mix_local_corpora", "shuffle_hf_datasets",
    "data_source", "epoch_cycles", "hf_dataset_id", "vjepa_dataset_ids", "vjepa_local_cache",
)}


def main() -> None:
    for _ in range(30):
        st = call("GET", "/api/status")
        if not st.get("running"):
            break
        call("POST", "/api/stop")
        time.sleep(2)
    call("POST", "/api/checkpoints/reset-session")

    vjepa_ids = DEFAULT_VJEPA_MIX[:]
    raw = os.environ.get("MCF_VJEPA_MIX", "").strip()
    if raw:
        vjepa_ids = [x.strip() for x in raw.split(",") if x.strip()]

    print(f"[CYAN] V-JEPA world mix ({len(vjepa_ids)} datasets) — hybrid local+HF…", flush=True)
    try:
        from vjepa_local_cache import cached_datasets, sync_cache_from_cos

        sync_cache_from_cos(vjepa_ids)
        ready = cached_datasets(vjepa_ids)
        print(f"[GREEN] cached locally: {len(ready)}/{len(vjepa_ids)} — {ready}", flush=True)
        missing = [d for d in vjepa_ids if d not in ready]
        if missing:
            print(f"[YELLOW] HF hybrid for uncached: {missing}", flush=True)
    except Exception as exc:
        print(f"[YELLOW] cache sync skipped: {exc}", flush=True)

    target_gb = float(os.environ.get("MCF_TARGET_GB", "5.25"))
    pack_rows = int(os.environ.get("MCF_PACK_ROWS", "128"))
    pack_min_chars = int(os.environ.get("MCF_PACK_MIN_CHARS", "4000"))
    body = {
        "data_source": "vjepa",
        "hf_dataset_id": vjepa_ids[0],
        "vjepa_dataset_ids": vjepa_ids,
        "vjepa_pack_rows": pack_rows,
        "vjepa_pack_min_chars": pack_min_chars,
        "vjepa_local_cache": True,
        "target_gb": target_gb,
        "model_params": int(os.environ.get("MCF_MODEL_PARAMS", "361850368")),
        "learning_rate": float(os.environ.get("MCF_LR", "2e-4")),
        "batch_size": int(os.environ.get("MCF_BATCH", "256")),
        "micro_batch_size": int(os.environ.get("MCF_MICRO", "64")),
        "grad_accum_steps": int(os.environ.get("MCF_GRAD_ACCUM", "4")),
        "n_layer": 17,
        "n_head": 15,
        "n_embd": 960,
        "block_size": 256,
        "vocab_size": 50257,
        "ffn_dim": 3840,
        "tokenizer_id": "gpt2",
        "mixed_precision": "bf16",
        "use_flash_attention": True,
        "scalar_input_projection": True,
        "gradient_checkpointing": False,
        "lr_schedule": "cosine",
        "total_train_steps": 0,
        "warmup_steps": 500,
        "save_every_n_steps": 1000,
        "weight_decay": 0.1,
        "dropout": 0.1,
        "english_only": True,
        "min_token_count": 32,
        "epoch_cycles": True,
        "epochs": int(os.environ.get("MCF_EPOCHS", "3")),
        "mix_local_jsonl": False,
        "mix_local_corpora": False,
        "shuffle_hf_datasets": False,
        "run_goal_title": "MCF v2 — Hierarchical Latent Stack (HLS)",
        "run_goal_detail": (
            f"Latent packing 128 frames/block + GCT physics + {len(vjepa_ids)}-way world-model interleave. "
            f"Hybrid local NVMe + IBM COS + HF. Target ~1.5B tokens ({target_gb:g} GB). "
            "resume:false clean start."
        ),
        "run_goal_agent": "mcf-recipe-v2-hls",
        "run_stop_primary": "bytes",
        "resume": False,
        "apply": APPLY_ALL,
    }
    pf = call("POST", "/api/preflight", body)
    if not pf.get("ok"):
        raise SystemExit(json.dumps(pf, indent=2))
    out = call("POST", "/api/start", body)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
