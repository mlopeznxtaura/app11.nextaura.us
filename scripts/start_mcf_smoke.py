#!/usr/bin/env python3
"""Start MCF (Metamorphic Constraint Field) V-JEPA pretrain on app8."""

from __future__ import annotations

import json
import os
import time
import urllib.request

API = os.environ.get("API", "http://127.0.0.1").rstrip("/")
VARIANT = os.environ.get("MCF_VARIANT", "vjepa").strip().lower()

DEFAULT_VJEPA_MIX = [
    "qinglinhou/sokoban-10k-vjepa2-tokenized",
    "cbctr/cs2-10k-vjepa2-latents-300",
    "rookierufus/ego10k-vjepa-latents",
    "rookierufus/Vjepa_mamba_dataset",
]


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
    "data_source", "epoch_cycles", "hf_dataset_id", "vjepa_dataset_ids",
)}


def _vjepa_mix() -> list[str]:
    raw = os.environ.get("MCF_VJEPA_MIX", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    single = os.environ.get("MCF_VJEPA_DATASET", "").strip()
    if single:
        return [single]
    return DEFAULT_VJEPA_MIX[:]


def main() -> None:
    for _ in range(30):
        st = call("GET", "/api/status")
        if not st.get("running"):
            break
        call("POST", "/api/stop")
        time.sleep(2)
    call("POST", "/api/checkpoints/reset-session")

    target_gb = float(os.environ.get("MCF_TARGET_GB", "24.0"))
    vjepa = VARIANT in ("vjepa", "stream-a", "a")
    vjepa_ids = _vjepa_mix() if vjepa else []
    body = {
        "data_source": "vjepa" if vjepa else "mixed",
        "hf_dataset_id": vjepa_ids[0] if vjepa_ids else None,
        "vjepa_dataset_ids": vjepa_ids if vjepa else [],
        "target_gb": target_gb,
        "model_params": int(os.environ.get("MCF_MODEL_PARAMS", "361850368")),
        "learning_rate": float(os.environ.get("MCF_LR", "2e-4")),
        "batch_size": int(os.environ.get("MCF_BATCH", "768")),
        "micro_batch_size": int(os.environ.get("MCF_MICRO", "128")),
        "grad_accum_steps": int(os.environ.get("MCF_GRAD_ACCUM", "6")),
        "n_layer": int(os.environ.get("MCF_N_LAYER", "17")),
        "n_head": int(os.environ.get("MCF_N_HEAD", "15")),
        "n_embd": int(os.environ.get("MCF_N_EMBD", "960")),
        "block_size": 256,
        "vocab_size": 50257,
        "ffn_dim": int(os.environ.get("MCF_FFN_DIM", "3840")),
        "tokenizer_id": "gpt2",
        "mixed_precision": "bf16",
        "use_flash_attention": True,
        "scalar_input_projection": True,
        "gradient_checkpointing": True,
        "lr_schedule": "cosine",
        "total_train_steps": 0,
        "warmup_steps": 500,
        "save_every_n_steps": 1000,
        "weight_decay": 0.1,
        "dropout": 0.1,
        "english_only": True,
        "min_token_count": 32,
        "epoch_cycles": False,
        "epochs": 1,
        "mix_local_jsonl": False,
        "mix_local_corpora": False,
        "shuffle_hf_datasets": False,
        "hf_stream_fable": False,
        "hf_stream_sol": not vjepa,
        "hf_stream_kimi": False,
        "hf_stream_nemotron": not vjepa,
        "hf_stream_unsolved_math": not vjepa,
        "hf_stream_preference": False,
        "hf_stream_fineweb": False,
        "run_goal_title": (
            "MCF Stream A — V-JEPA multi-dataset physics field"
            if vjepa
            else "MCF — Metamorphic Constraint Field (dual-stream smoke)"
        ),
        "run_goal_detail": (
            f"Interleaved V-JEPA ({len(vjepa_ids)} HF streams): "
            + ", ".join(vjepa_ids[:4])
            + ("…" if len(vjepa_ids) > 4 else "")
            + ". Scalar 12d + grad ckpt. Target "
            + f"{target_gb:g} GB."
            if vjepa
            else "Stream B braid: sol + nemotron + math. Scalar 12d bridge."
        ),
        "run_goal_agent": "mcf-recipe-v1-stream-a" if vjepa else "mcf-recipe-v1",
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
