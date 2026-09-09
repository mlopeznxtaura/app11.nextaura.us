#!/usr/bin/env python3
"""Stop MCF v2, deploy world-model mix, resume from checkpoint (no reset-session)."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

API = os.environ.get("API", "http://127.0.0.1").rstrip("/")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model_mix import DEFAULT_VJEPA_MIX, VJEPA_MIX_WORLD

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


def _resolve_mix() -> list[str]:
    raw = os.environ.get("MCF_VJEPA_MIX", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return DEFAULT_VJEPA_MIX[:]


def main() -> None:
    vjepa_ids = _resolve_mix()
    print(f"[CYAN] World-model mix ({len(vjepa_ids)} datasets):", flush=True)
    for ds in vjepa_ids:
        print(f"  · {ds}", flush=True)

    st = call("GET", "/api/status")
    if st.get("running"):
        print(f"[YELLOW] Stopping at step {st.get('train_step')}…", flush=True)
        call("POST", "/api/stop")
        for _ in range(90):
            time.sleep(2)
            st = call("GET", "/api/status")
            if not st.get("running"):
                break
        if st.get("running"):
            raise SystemExit("Training did not stop in time")

    try:
        from vjepa_local_cache import cached_datasets, sync_cache_from_cos

        sync_cache_from_cos(vjepa_ids)
        ready = cached_datasets(vjepa_ids)
        print(f"[GREEN] local cache ready: {len(ready)}/{len(vjepa_ids)} — {ready}", flush=True)
        if ready and len(ready) < len(vjepa_ids):
            missing = [d for d in vjepa_ids if d not in ready]
            print(
                f"[YELLOW] HF streams block interleave — training on cached {len(ready)} now; "
                f"run cache_vjepa_datasets.py for: {missing}",
                flush=True,
            )
            vjepa_ids = ready
    except Exception as exc:
        print(f"[YELLOW] cache sync skipped: {exc}", flush=True)

    st = call("GET", "/api/status")
    target_gb = float(os.environ.get("MCF_TARGET_GB", "5.25"))
    body = {
        "data_source": "vjepa",
        "hf_dataset_id": vjepa_ids[0],
        "vjepa_dataset_ids": vjepa_ids,
        "vjepa_pack_rows": int(os.environ.get("MCF_PACK_ROWS", "128")),
        "vjepa_pack_min_chars": int(os.environ.get("MCF_PACK_MIN_CHARS", "4000")),
        "vjepa_local_cache": True,
        "target_gb": target_gb,
        "model_params": int(st.get("model_params") or 236724480),
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
        "run_goal_title": "MCF v2 — HLS + world-model mix",
        "run_goal_detail": (
            f"Resume step {st.get('train_step', 0)} · {len(vjepa_ids)}-way interleave "
            f"(CS2/ego/sokoban + jepa-wms/LIBERO/behavior/MuJoCo/PhysSim/AgiBot/NMS) · "
            f"hybrid local+HF · target {target_gb:g} GB"
        ),
        "run_goal_agent": "mcf-recipe-v2-hls",
        "run_stop_primary": "bytes",
        "resume": False,
        "stream_skip_rows": 0,
        "apply": APPLY_ALL,
    }
    print(
        f"[CYAN] Resuming from step {st.get('train_step')} · "
        f"{st.get('checkpoint_name')} · {len(vjepa_ids)}-dataset world mix",
        flush=True,
    )
    pf = call("POST", "/api/preflight", body)
    if not pf.get("ok"):
        raise SystemExit(json.dumps(pf, indent=2))
    out = call("POST", "/api/start", body)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
