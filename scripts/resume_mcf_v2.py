#!/usr/bin/env python3
"""Resume MCF v2 HLS from last checkpoint (grad checkpointing OFF)."""

from __future__ import annotations

import json
import os
import sys
import urllib.request

API = os.environ.get("API", "http://127.0.0.1").rstrip("/")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model_mix import DEFAULT_VJEPA_MIX

APPLY_ALL = {k: True for k in (
    "n_layer", "n_head", "n_embd", "block_size", "vocab_size", "ffn_dim", "tokenizer_id",
    "learning_rate", "batch_size", "micro_batch_size", "grad_accum_steps", "target_gb",
    "total_train_steps", "scalar_input_projection", "gradient_checkpointing", "warmup_steps",
    "mixed_precision", "lr_schedule", "save_every_n_steps", "use_flash_attention",
    "hf_streams", "mix_local_jsonl", "mix_local_corpora", "shuffle_hf_datasets",
    "data_source", "epoch_cycles", "hf_dataset_id", "vjepa_dataset_ids", "vjepa_local_cache",
)}


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{API}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def main() -> None:
    from vjepa_local_cache import cached_datasets

    st = call("GET", "/api/status")
    if st.get("running"):
        print("[RED] training already running — use scripts/switch_world_mix.py", file=sys.stderr)
        raise SystemExit(1)
    if not st.get("can_resume"):
        print("[RED] session not resumable — can_resume=false", file=sys.stderr)
        raise SystemExit(json.dumps(st, indent=2))

    vjepa_ids = DEFAULT_VJEPA_MIX[:]
    raw = os.environ.get("MCF_VJEPA_MIX", "").strip()
    if raw:
        vjepa_ids = [x.strip() for x in raw.split(",") if x.strip()]
    ready = cached_datasets(vjepa_ids)
    print(f"[CYAN] world mix {len(vjepa_ids)} datasets · local cache {len(ready)}/{len(vjepa_ids)}", flush=True)

    target_gb = float(os.environ.get("MCF_TARGET_GB", str(st.get("target_gb") or 5.25)))
    body = {
        "data_source": "vjepa",
        "hf_dataset_id": vjepa_ids[0],
        "vjepa_dataset_ids": vjepa_ids,
        "vjepa_pack_rows": int(os.environ.get("MCF_PACK_ROWS", "128")),
        "vjepa_pack_min_chars": int(os.environ.get("MCF_PACK_MIN_CHARS", "4000")),
        "vjepa_local_cache": True,
        "target_gb": target_gb,
        "model_params": int(st.get("model_params") or 361850368),
        "learning_rate": float(st.get("learning_rate") or 2e-4),
        "batch_size": int(st.get("batch_size") or 256),
        "micro_batch_size": int(st.get("micro_batch_size") or 64),
        "grad_accum_steps": int(st.get("grad_accum_steps") or 4),
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
        "epoch_cycles": True,
        "epochs": int(os.environ.get("MCF_EPOCHS", "3")),
        "mix_local_jsonl": False,
        "mix_local_corpora": False,
        "shuffle_hf_datasets": False,
        "run_goal_title": st.get("run_goal_title") or "MCF v2 — HLS + world-model mix",
        "run_goal_detail": (
            f"Resume step {st.get('train_step', 0)} · {len(vjepa_ids)}-way world mix · "
            f"grad_ckpt OFF · hybrid cache · target {target_gb:g} GB"
        ),
        "run_goal_agent": "mcf-recipe-v2-hls",
        "run_stop_primary": "bytes",
        "resume": True,
        "apply": APPLY_ALL,
    }
    print(
        f"[CYAN] Resuming from step {st.get('train_step')} · "
        f"{st.get('checkpoint_name')} · gradient_checkpointing=false",
        flush=True,
    )
    pf = call("POST", "/api/preflight", body)
    if not pf.get("ok"):
        raise SystemExit(json.dumps(pf, indent=2))
    out = call("POST", "/api/resume", body)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
