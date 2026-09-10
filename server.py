"""app11.nextaura.us — open-source CPU-first NextAura training lab."""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

import ibm_boto3
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ibm_botocore.client import Config
from pydantic import BaseModel, Field

from scalar_features import row_scalar_features
from multimodal_data import multimodal_training_text, vjepa_scalar_features, vjepa_training_text
from hf_datasets import (
    DEFAULT_MULTIMODAL_DATASET,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_VJEPA_DATASET,
    MULTIMODAL_DATASETS,
    TOKENIZER_OPTIONS,
    VJEPA_DATASETS,
)
from hf_stream_loader import (
    likely_embedding_only,
    open_hf_train_stream,
    open_hf_train_stream_interleaved,
    probe_hf_train_stream,
)
from data_jsonl import (
    accept_training_text,
    load_jsonl_lines,
    nemotron_math_training_text,
    nemotron_swe_training_text,
    preference_training_text,
    row_training_text,
    unsolved_math_training_text,
)
from train_engine import TrainEngine, should_save_model
from convert_model import import_hf_causal, import_hf_gpt2
from agent_contract import (
    build_agent_contract,
    render_agent_txt,
    render_llms_txt,
    render_robots_txt,
    render_sitemap_xml,
    PUBLIC_URL,
)
from platform_index import build_local_models_payload, build_platform_index, load_cached_platform_index, save_platform_index
from run_history import (
    backfill_disk_checkpoints,
    record_engineering,
    record_reward_eval,
    record_training_finish,
    run_history_bundle,
    has_run_record,
)
from object_storage import (
    download_checkpoint_bytes,
    list_remote_checkpoints,
    storage_status,
    upload_checkpoint_file,
    upcloud_configured,
)
from sft_topics import SFT_TOPICS, generate_topics_sft, topic_catalog, topic_jsonl_filename
from distill_teachers import (
    DISTILL_TOPICS,
    TEACHER_REGISTRY,
    TopicTeacherSession,
    append_live_distill_row,
    distill_catalog,
    distill_jsonl_filename,
    generate_distill_corpus,
    iter_distill_live_prompts,
    make_distill_row,
    merged_distill_filename,
    LIVE_DISTILL_LOG,
)
from model import ModelConfig, estimate_params, flash_attention_available
from tokenizer_utils import tokenizer_guidance_bundle
from text_quality import truncate_chat_answer, wrap_chat_prompt
from reward_eval import load_benchmark_prompts, run_reward_panel
from preflight import run_preflight
from eval_orchestrator import (
    AUTO_EVAL_ON_SAVE,
    COMPARE_EVAL_LIMIT,
    SAVE_EVAL_LIMIT,
    build_panel_result,
    eval_leaderboard,
    pick_best_checkpoint,
)
from timemoe_baseline import (
    TIMEMOE_MODEL_ID,
    benchmark_summary_for_api,
    compare_to_timemoe,
    load_baseline_state,
    run_timemoe_benchmark,
    save_baseline_state,
)

HOST = os.environ.get("PUBLIC_HOST", "app11.nextaura.us")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://app11.nextaura.us")
VOICE_TO_PLAN_URL = "https://app7.nextaura.fit"
DATASET_ID = "HuggingFaceFW/fineweb"
HF_SHUFFLE_BUFFER = 8192
HF_SHUFFLE_MATERIALIZE_MAX = 200_000
FABLE_DATASET_ID = "Glint-Research/Fable-5-traces"
RM_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"
HH_RLHF_DATASET = "Anthropic/hh-rlhf"
SOL_TRACES_DATASET_ID = "greghavens/gpt-5.6-sol-coding-and-debugging-traces"
SOL_TRACES_URL = (
    "https://huggingface.co/datasets/greghavens/gpt-5.6-sol-coding-and-debugging-traces/"
    "resolve/main/traces.jsonl"
)
KIMI_K3_TRACES_DATASET_ID = "greghavens/kimi-k3-coding-and-debugging-traces"
UNSOLVED_MATH_DATASET_ID = "ulamai/UnsolvedMath"
UNSOLVED_MATH_URL = (
    "https://huggingface.co/datasets/ulamai/UnsolvedMath/resolve/main/problems.json"
)
NEMOTRON_MATH_DATASET_ID = "nvidia/Nemotron-SFT-Math-v4"
NEMOTRON_SWE_DATASET_ID = "nvidia/Nemotron-SFT-SWE-v3.5"
DATA_SOURCES = (
    "fineweb",
    "jsonl",
    "local-corpus-mix",
    "sft-intelligence",
    "sft-coding",
    "sft-physics",
    "sft-math",
    "distill-merged",
    "distill-coding",
    "distill-physics",
    "distill-math",
    "distill-live",
    "fable-traces",
    "mixed",
    "openassistant-rm",
    "sol-traces",
    "kimi-k3-traces",
    "unsolved-math",
    "nemotron-math",
    "nemotron-swe",
    "multimodal",
    "vjepa",
)
SFT_TOPIC_SOURCES: dict[str, str] = {
    "sft-coding": "coding",
    "sft-physics": "physics",
    "sft-math": "math",
}
DISTILL_TOPIC_SOURCES: dict[str, str] = {
    "distill-coding": "coding",
    "distill-physics": "physics",
    "distill-math": "math",
    "distill-merged": "merged",
}
MIN_TARGET_GB = float(os.environ.get("MIN_TARGET_GB", "0.001"))  # 1 MB as 0.001 GB


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


DISTILL_LIVE_ENABLED = _env_flag("DISTILL_LIVE_ENABLED", "0")
MASTER_AGENT_ENABLED = _env_flag("MASTER_AGENT_ENABLED", "0")
DEFAULT_N_LAYER = 8
DEFAULT_N_HEAD = 8
DEFAULT_N_EMBD = 256
DEFAULT_BLOCK_SIZE = 256
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_FFN_DIM = 1024
DEFAULT_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "1024"))
DEFAULT_LR = float(os.environ.get("TRAIN_LR", "3e-4"))
FABLE_MERGED_URL = (
    "https://huggingface.co/datasets/Glint-Research/Fable-5-traces/resolve/main/fable5_cot_merged.jsonl"
)
LOCAL_CORPUS_MIX_FILES = (
    "2400-clean.jsonl",
    "4563-curated.jsonl",
    "nextaura-extract-merged.jsonl",
)
LOCAL_CORPUS_MIX_LABEL = "2400+4563-curated+extract"
DEFAULT_CONFIG = "sample-10BT"
CHINCHILLA_TOKENS_PER_PARAM = 20
BYTES_PER_TRAINING_TOKEN = 3.5  # UTF-8 bytes ≈ model tokens for mixed corpora
FOOTPRINT_10M_CFG = ModelConfig.footprint_10m()
FOOTPRINT_25M_CFG = ModelConfig.footprint_25m()
FOOTPRINT_50M_CFG = ModelConfig.footprint_50m()
FOOTPRINT_10M_PARAMS = estimate_params(FOOTPRINT_10M_CFG)
FOOTPRINT_25M_PARAMS = estimate_params(FOOTPRINT_25M_CFG)
FOOTPRINT_50M_PARAMS = estimate_params(FOOTPRINT_50M_CFG)
DEFAULT_MODEL_PARAMS = FOOTPRINT_10M_PARAMS
DEFAULT_SMOKE_TARGET_GB = 0.08  # ~0.1× of 10M 0.25× Chinchilla row — ladder smokes


def chinchilla_target_tokens(model_params: int | float = DEFAULT_MODEL_PARAMS) -> int:
    """Chinchilla-optimal training tokens ≈ 20 × params."""
    return int(model_params) * CHINCHILLA_TOKENS_PER_PARAM


def chinchilla_train_steps(
    model_params: int | float = DEFAULT_MODEL_PARAMS,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    tokens_per_step = max(1, block_size * batch_size)
    return max(1, math.ceil(chinchilla_target_tokens(model_params) / tokens_per_step))


def chinchilla_target_gb(model_params: int | float = DEFAULT_MODEL_PARAMS) -> float:
    """Byte budget proxy aligned with Chinchilla token target."""
    tokens = chinchilla_target_tokens(model_params)
    return round(tokens * BYTES_PER_TRAINING_TOKEN / (1024**3), 2)


_CHINCHILLA_MULT_LABELS: dict[float, str] = {
    0.25: "0.25× smoke",
    0.5: "0.5× light",
    1.0: "1× Chinchilla optimal",
    2.0: "2× extended",
    3.0: "3× heavy",
    10.0: "10× legacy",
}

_CHINCHILLA_FINETUNE_MULT_LABELS: dict[float, str] = {
    0.01: "0.01× smoke",
    0.05: "0.05× light",
    0.1: "0.1× standard",
    0.25: "0.25× extended",
    0.5: "0.5× heavy",
    1.0: "1× full replay",
}


def _fmt_gb_short(gb: float) -> str:
    if gb < 0.01:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.2f} GB"


def _pretrain_mult_recommendation(mult: float, model_params: int) -> str:
    gb = chinchilla_target_gb(model_params) * mult
    m = model_params / 1e6
    if mult <= 0.5:
        lr = "lr 2e-4–3e-4"
    elif mult <= 2.0:
        lr = "lr 1e-4–2e-4"
    elif mult <= 3.0:
        lr = "lr 5e-5–1e-4"
    else:
        lr = "lr 3e-5–1e-4"
    epoch = "epochs off (1–3 only if JSONL <50 MB)" if mult == 0.25 else "epochs off"
    if mult == 0.25 and model_params <= FOOTPRINT_50M_PARAMS:
        cap = f"smoke cap 0.05–{_fmt_gb_short(gb)}"
        if model_params <= FOOTPRINT_10M_PARAMS * 1.05:
            cap += " · ladder default 0.08 GB"
        return f"{lr} · {cap} ({m:.1f}M) · {epoch}"
    return f"{lr} · ~{_fmt_gb_short(gb)} ({m:.1f}M) · {epoch}"


def _finetune_mult_recommendation(mult: float, model_params: int) -> str:
    gb = chinchilla_target_gb(model_params) * mult
    m = model_params / 1e6
    if mult <= 0.01:
        lr = "lr 1e-5–3e-5"
        epochs = "epochs 1–5"
    elif mult <= 0.1:
        lr = "lr 1e-5–5e-5"
        epochs = "epochs off"
    elif mult <= 0.25:
        lr = "lr 5e-6–5e-5"
        epochs = "epochs off"
    elif mult <= 0.5:
        lr = "lr 3e-6–3e-5"
        epochs = "epochs off · memorization risk"
    else:
        lr = "lr 3e-6–1e-5"
        epochs = "epochs off · avoid unless ablation"
    base = "pretrained base required" if model_params <= FOOTPRINT_10M_PARAMS * 1.05 else "load pretrain ckpt first"
    return f"{lr} · ~{_fmt_gb_short(gb)} ({m:.1f}M) · {epochs} · {base}"


def _footprint_pretrain_recommendation(params: int, label: str) -> str:
    gb = chinchilla_target_gb(params)
    smoke = chinchilla_target_gb(params) * 0.25
    if params <= FOOTPRINT_10M_PARAMS * 1.05:
        return (
            f"tok StorySupra 8K · 1× ≈{_fmt_gb_short(gb)} · smoke 0.05–{_fmt_gb_short(smoke)} "
            f"(default 0.08 GB) · lr 2e-4–3e-4 · epochs off"
        )
    if params <= FOOTPRINT_25M_PARAMS * 1.05:
        return (
            f"tok StorySupra 8K · 1× ≈{_fmt_gb_short(gb)} · distill student "
            f"· smoke ~{_fmt_gb_short(smoke)} (0.2 GB) · lr 1e-5–2e-5 · POST /api/distill/generate first"
        )
    if params <= FOOTPRINT_50M_PARAMS * 1.05:
        return (
            f"tok gpt2 · 1× ≈{_fmt_gb_short(gb)} · smoke ~{_fmt_gb_short(smoke)} "
            f"· scale only after 10M smokes pass · lr 2e-4–3e-4"
        )
    if params <= 60_000_000:
        return f"tok gpt2 · 1× ≈{_fmt_gb_short(gb)} · lr 2e-4–3e-4 · epochs off"
    if params <= 140_000_000:
        return f"tok gpt2 · 1× ≈{_fmt_gb_short(gb)} · lr 1e-4–2e-4 · ~8 GB at 1×"
    return f"tok gpt2 · 1× ≈{_fmt_gb_short(gb)} · lr 5e-5–1e-4"


def _footprint_finetune_recommendation(params: int) -> str:
    gb = chinchilla_target_gb(params) * 0.1
    if params <= FOOTPRINT_10M_PARAMS * 1.05:
        return f"tok match ckpt · 0.1× ≈{_fmt_gb_short(gb)} · lr 1e-5–3e-5 · epochs 1–5"
    if params <= FOOTPRINT_25M_PARAMS * 1.05:
        return (
            f"tok StorySupra 8K · 0.1× ≈{_fmt_gb_short(gb)} · lr 1e-5–2e-5 "
            f"· reset session + distill-merged · teachers via /api/distill/generate"
        )
    if params <= FOOTPRINT_50M_PARAMS * 1.05:
        return f"tok match ckpt · 0.1× ≈{_fmt_gb_short(gb)} · lr 1e-5–1e-4 · load 10M/68M pretrain first"
    if params <= 140_000_000:
        return f"tok match ckpt · 0.1× ≈{_fmt_gb_short(gb)} · lr 5e-6–5e-5"
    return f"tok match ckpt · 0.1× ≈{_fmt_gb_short(gb)} · lr 3e-6–3e-5"


_CHINCHILLA_FOOTPRINTS: tuple[tuple[int, str], ...] = (
    (FOOTPRINT_10M_PARAMS, "~12.6M (8L×256d · StorySupra 8K)"),
    (FOOTPRINT_25M_PARAMS, "~25.4M (10L×336d · StorySupra 8K · distill student)"),
    (FOOTPRINT_50M_PARAMS, "~68M (8L×512d · gpt2)"),
    (124_000_000, "GPT-2 124M"),
    (350_000_000, "GPT-2 medium"),
)


def _chinchilla_scale_rows(
    optimal_tokens: int,
    tokens_per_step: int,
    mults: tuple[float, ...],
    labels: dict[float, str],
    model_params: int,
    *,
    mode: str = "pretrain",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rec_fn = _pretrain_mult_recommendation if mode == "pretrain" else _finetune_mult_recommendation
    for mult in mults:
        tokens = max(1, int(optimal_tokens * mult))
        bytes_total = tokens * BYTES_PER_TRAINING_TOKEN
        rows.append(
            {
                "mult": mult,
                "label": labels[mult],
                "tokens": tokens,
                "gb": round(bytes_total / (1024**3), 4),
                "mb": round(bytes_total / (1024**2), 2),
                "steps": max(1, math.ceil(tokens / tokens_per_step)),
                "recommendation": rec_fn(mult, model_params),
            }
        )
    return rows


def _chinchilla_footprint_rows(
    *,
    block_size: int,
    batch_size: int,
    mode: str = "pretrain",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rec_fn = _footprint_pretrain_recommendation if mode == "pretrain" else _footprint_finetune_recommendation
    for footprint_params, footprint_label in _CHINCHILLA_FOOTPRINTS:
        tokens = chinchilla_target_tokens(footprint_params)
        rows.append(
            {
                "params": footprint_params,
                "label": footprint_label,
                "tokens": tokens,
                "gb": chinchilla_target_gb(footprint_params),
                "steps": chinchilla_train_steps(
                    footprint_params,
                    block_size=block_size,
                    batch_size=batch_size,
                ),
                "recommendation": rec_fn(footprint_params, footprint_label)
                if mode == "pretrain"
                else rec_fn(footprint_params),
            }
        )
    return rows


def chinchilla_scaling_bundle(
    model_params: int | float,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_embd: int = DEFAULT_N_EMBD,
    vocab_size: int = 50257,
    mode: str = "pretrain",
    checkpoint_vocab: int | None = None,
    checkpoint_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Live Chinchilla table: tokens ≈ 20×params, byte budget, pretrain + finetune modes."""
    params = max(1, int(model_params))
    tokens_per_step = max(1, int(block_size) * int(batch_size))
    optimal_tokens = chinchilla_target_tokens(params)
    pretrain_multipliers = _chinchilla_scale_rows(
        optimal_tokens,
        tokens_per_step,
        (0.25, 0.5, 1.0, 2.0, 3.0, 10.0),
        _CHINCHILLA_MULT_LABELS,
        params,
        mode="pretrain",
    )
    finetune_multipliers = _chinchilla_scale_rows(
        optimal_tokens,
        tokens_per_step,
        (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
        _CHINCHILLA_FINETUNE_MULT_LABELS,
        params,
        mode="finetune",
    )
    pretrain_footprints = _chinchilla_footprint_rows(
        block_size=block_size,
        batch_size=batch_size,
        mode="pretrain",
    )
    finetune_footprints = _chinchilla_footprint_rows(
        block_size=block_size,
        batch_size=batch_size,
        mode="finetune",
    )
    optimal_gb = chinchilla_target_gb(params)
    return {
        "law": "Hoffmann/Chinchilla: train on ~20 tokens per parameter for compute-optimal pretrain",
        "tokens_per_param": CHINCHILLA_TOKENS_PER_PARAM,
        "bytes_per_token": BYTES_PER_TRAINING_TOKEN,
        "tokens_per_step": tokens_per_step,
        "model_params": params,
        "optimal_tokens": optimal_tokens,
        "optimal_gb": optimal_gb,
        "optimal_steps": chinchilla_train_steps(
            params,
            block_size=block_size,
            batch_size=batch_size,
        ),
        "multipliers": pretrain_multipliers,
        "footprints": pretrain_footprints,
        "modes": {
            "pretrain": {
                "title": "Pretrain scaling",
                "law": "Hoffmann/Chinchilla: ~20 tokens per parameter · mixed / HF / FineWeb streams",
                "reference_mult": 1.0,
                "reference_label": "1× Chinchilla optimal",
                "multipliers": pretrain_multipliers,
                "footprints": pretrain_footprints,
            },
            "finetune": {
                "title": "Fine-tuning budget",
                "law": "SFT / finetune: 1–10% of pretrain byte budget typical · lower lr · pretrained base required",
                "reference_mult": 0.1,
                "reference_label": "0.1× standard SFT",
                "multipliers": finetune_multipliers,
                "footprints": finetune_footprints,
            },
        },
        "tokenizer": tokenizer_guidance_bundle(
            params,
            n_embd=n_embd,
            vocab_size=vocab_size,
            mode=mode,
            checkpoint_vocab=checkpoint_vocab,
            checkpoint_tokenizer=checkpoint_tokenizer,
        ),
    }


_env_target_gb = os.environ.get("TARGET_GB", "").strip()
DEFAULT_TARGET_GB = float(_env_target_gb) if _env_target_gb else DEFAULT_SMOKE_TARGET_GB
DEFAULT_TOTAL_TRAIN_STEPS = chinchilla_train_steps()
DEFAULT_WARMUP_STEPS = 500
DEFAULT_SAVE_EVERY_N_STEPS = 2000
DEFAULT_MIXED_PRECISION = os.environ.get("TRAIN_MIXED_PRECISION", "bf16")
DEFAULT_MICRO_BATCH = int(os.environ.get("TRAIN_MICRO_BATCH", "256"))
MAX_CUDA_MICRO_BATCH = int(os.environ.get("TRAIN_MICRO_BATCH_MAX", "512"))
CHECKPOINT_KEY = os.environ.get("COS_CHECKPOINT_KEY", "app7/stage1/checkpoint.json")
MODEL_KEY = os.environ.get("COS_MODEL_KEY", "app7/stage1/model/latest.pt")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
DATA_DIR = os.environ.get("DATA_DIR", "data")
TRAINING_HEARTBEAT = os.environ.get("TRAINING_HEARTBEAT", "/tmp/app7-nextaura-training.heartbeat")
MAX_CHECKPOINT_MB = int(os.environ.get("MAX_CHECKPOINT_MB", "1024"))
MAX_JSONL_MB = int(os.environ.get("MAX_JSONL_MB", "256"))

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
GPU_PEER_URLS = [
    u.strip().rstrip("/")
    for u in os.environ.get("GPU_PEER_URLS", "").split(",")
    if u.strip()
]


def _nvidia_smi_vram_mb() -> tuple[float, float] | None:
    import subprocess

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        used_s, total_s = proc.stdout.strip().split(",", 1)
        return float(used_s.strip()), float(total_s.strip())
    except Exception:
        return None


def _local_gpu_vram_stats() -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "host": HOST,
        "device": DEVICE,
        "name": "",
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "total_mb": 0.0,
        "utilization_pct": 0.0,
    }
    smi = _nvidia_smi_vram_mb()
    if smi:
        used_mb, total_mb = smi
        out.update(
            {
                "available": True,
                "allocated_mb": round(used_mb, 1),
                "total_mb": round(total_mb, 1),
                "utilization_pct": round(100.0 * used_mb / total_mb, 1) if total_mb else 0.0,
            }
        )
    if DEVICE != "cuda":
        return out
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        torch_alloc = torch.cuda.memory_allocated(idx) / (1024**2)
        torch_reserved = torch.cuda.memory_reserved(idx) / (1024**2)
        total_mb = props.total_memory / (1024**2)
        allocated_mb = max(out.get("allocated_mb") or 0.0, torch_alloc)
        out.update(
            {
                "available": True,
                "name": props.name,
                "allocated_mb": round(allocated_mb, 1),
                "reserved_mb": round(max(out.get("reserved_mb") or 0.0, torch_reserved), 1),
                "total_mb": round(out.get("total_mb") or total_mb, 1),
            }
        )
        total = float(out["total_mb"] or 0.0)
        if total > 0:
            out["utilization_pct"] = round(100.0 * float(out["allocated_mb"]) / total, 1)
    except Exception:
        pass
    return out


def _gpu_vram_snapshot(*, include_peers: bool = True) -> dict[str, Any]:
    local = _local_gpu_vram_stats()
    peers: list[dict[str, Any]] = []
    if include_peers and GPU_PEER_URLS:
        import urllib.error
        import urllib.request

        for peer_url in GPU_PEER_URLS:
            peer: dict[str, Any] = {"host": peer_url, "available": False}
            try:
                req = urllib.request.Request(
                    f"{peer_url}/api/gpu/vram?peers=0",
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    peer = json.loads(resp.read().decode())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                peer["error"] = str(exc)[:120]
            peers.append(peer)
    combined_allocated = float(local.get("allocated_mb") or 0.0) + sum(
        float(p.get("allocated_mb") or 0.0) for p in peers if p.get("available")
    )
    combined_total = float(local.get("total_mb") or 0.0) + sum(
        float(p.get("total_mb") or 0.0) for p in peers if p.get("available")
    )
    return {
        "local": local,
        "peers": peers,
        "combined_allocated_mb": round(combined_allocated, 1),
        "combined_total_mb": round(combined_total, 1),
        "combined_utilization_pct": round(100.0 * combined_allocated / combined_total, 1)
        if combined_total
        else 0.0,
    }


def _touch_training_heartbeat() -> None:
    try:
        with open(TRAINING_HEARTBEAT, "a", encoding="utf-8"):
            os.utime(TRAINING_HEARTBEAT, None)
    except OSError:
        try:
            with open(TRAINING_HEARTBEAT, "w", encoding="utf-8"):
                pass
        except OSError:
            pass


def _clear_training_heartbeat() -> None:
    try:
        os.remove(TRAINING_HEARTBEAT)
    except OSError:
        pass


IBM_API_KEY = os.environ.get("IBM_CLOUD_API_KEY", "").strip()
COS_CRN = os.environ.get("COS_CRN", "").strip()
COS_BUCKET = os.environ.get("COS_BUCKET", "").strip()
COS_ENDPOINT = os.environ.get(
    "COS_ENDPOINT",
    "https://s3.us-south.cloud-object-storage.appdomain.cloud",
)

FINEWEB_CONFIGS = [
    "sample-10BT",
    "sample-100BT",
    "sample-350BT",
    "CC-MAIN-2024-10",
    "CC-MAIN-2023-50",
    "default",
]

app = FastAPI(title="NextAura Multimodal Fine-tune", version="1.0")

_hf_probe_cache: dict[str, dict[str, Any]] = {}
_hf_probe_lock = threading.Lock()
_hf_probe_started = False


def _probe_hf_dataset(ds_id: str, kind: str) -> dict[str, Any]:
    try:
        return probe_hf_train_stream(ds_id, kind=kind, token=HF_TOKEN or None, max_rows=80)
    except Exception as exc:
        return {"ok": False, "dataset_id": ds_id, "kind": kind, "error": str(exc)}


def _hf_probe_background() -> None:
    global _hf_probe_started
    priority_mm = [
        DEFAULT_MULTIMODAL_DATASET,
        "multimodal-reasoning-lab/Zebra-CoT",
        "DAMO-NLP-SG/multimodal_textbook",
        "SamBP069/olives-multimodal-dataset",
    ]
    priority_vj = [DEFAULT_VJEPA_DATASET, "rookierufus/epic-kitchens-vjepa", "phi-9/epic-kitchens-vjepa"]
    ordered_mm = priority_mm + [d for d in MULTIMODAL_DATASETS if d not in priority_mm]
    ordered_vj = priority_vj + [d for d in VJEPA_DATASETS if d not in priority_vj]
    for ds_id in ordered_mm:
        result = _probe_hf_dataset(ds_id, "multimodal")
        with _hf_probe_lock:
            _hf_probe_cache[ds_id] = result
    for ds_id in ordered_vj:
        result = _probe_hf_dataset(ds_id, "vjepa")
        with _hf_probe_lock:
            _hf_probe_cache[ds_id] = result
    ok_mm = sum(1 for d in ordered_mm if _hf_probe_cache.get(d, {}).get("ok"))
    ok_vj = sum(1 for d in ordered_vj if _hf_probe_cache.get(d, {}).get("ok"))
    S.add_log(f"[CYAN] HF probe cache: {ok_mm}/{len(ordered_mm)} multimodal · {ok_vj}/{len(ordered_vj)} V-JEPA streams OK", "cyan")


def _hf_probe_snapshot() -> dict[str, dict[str, Any]]:
    with _hf_probe_lock:
        return dict(_hf_probe_cache)


def _verified_hf_ids(kind: str) -> list[str]:
    snap = _hf_probe_snapshot()
    registry = MULTIMODAL_DATASETS if kind == "multimodal" else VJEPA_DATASETS
    verified = [d for d in registry if snap.get(d, {}).get("ok")]
    if verified:
        return verified
    default = DEFAULT_MULTIMODAL_DATASET if kind == "multimodal" else DEFAULT_VJEPA_DATASET
    return [default]
app.mount("/static", StaticFiles(directory="public"), name="static")

_engine: TrainEngine | None = None
_engine_lock = threading.Lock()
_reward_model = None
_reward_tokenizer = None
_reward_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _pick_device() -> str:
    import torch

    forced = os.environ.get("TRAIN_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


os.environ.setdefault("TRAIN_DEVICE", "cpu")
DEVICE = _pick_device()
if DEVICE == "cuda":
    os.environ.setdefault("TRAIN_BATCH_SIZE", os.environ.get("TRAIN_BATCH_SIZE", "1024"))
elif DEVICE == "mps":
    os.environ.setdefault("TRAIN_BATCH_SIZE", os.environ.get("TRAIN_BATCH_SIZE", "32"))
import train_engine as _te

_te.BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))


def _get_engine() -> TrainEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = TrainEngine(device=DEVICE)
        return _engine


def _cos_client():
    if not IBM_API_KEY:
        return None
    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=IBM_API_KEY,
        ibm_service_instance_id=COS_CRN,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT,
    )


class StreamState:
    def __init__(self) -> None:
        self.running = False
        self.paused = False
        self.status = "idle"
        self.config = DEFAULT_CONFIG
        self.target_gb = DEFAULT_TARGET_GB
        self.model_params = DEFAULT_MODEL_PARAMS
        self.learning_rate = DEFAULT_LR
        self.batch_size = DEFAULT_BATCH_SIZE
        self.rows = 0
        self.skipped = 0
        self.bytes_written = 0
        self.stream_index = 0
        self.train_step = 0
        self.last_loss = 0.0
        self.last_tok_s = 0.0
        self.last_id = ""
        self.last_url = ""
        self.error = ""
        self.model_ready = False
        self.checkpoint_name = ""
        self.data_source = "fineweb"
        self.jsonl_file = "4563-curated.jsonl"
        self.epochs = 1
        self.epoch_cycles = False
        self.jsonl_mix_ratio = 0.25
        self.mix_local_jsonl = True
        self.jsonl_total_rows = 0
        self.jsonl_pool_rows = 0
        self.jsonl_epoch = 0
        self.jsonl_hits = 0
        self.fable_hits = 0
        self.preference_hits = 0
        self.sol_hits = 0
        self.kimi_hits = 0
        self.math_hits = 0
        self.nemotron_hits = 0
        self.swe_hits = 0
        self.mix_local_corpora = False
        self.shuffle_hf_datasets = False
        self.hf_stream_fable = True
        self.hf_stream_sol = True
        self.hf_stream_kimi = True
        self.hf_stream_nemotron = True
        self.hf_stream_unsolved_math = True
        self.hf_stream_preference = True
        self.hf_stream_fineweb = False
        self.hf_dataset_id = DEFAULT_MULTIMODAL_DATASET
        self.tokenizer_id = DEFAULT_TOKENIZER_ID
        self.multimodal_hits = 0
        self.vjepa_hits = 0
        self.run_goal_title = ""
        self.run_goal_detail = ""
        self.run_stop_primary = "bytes"
        self.run_goal_agent = ""
        self.run_goal_started_at = ""
        self.train_wall_started_at = 0.0
        self.last_step_monotonic = 0.0
        self.last_step_interval_s = 0.0
        self.last_epoch_duration_s = 0.0
        self.stream_epoch = 0
        self.log_lines: list[dict[str, Any]] = []
        self.skip_cos_hydrate = False
        self.pending_save_eval: tuple[str, int] | None = None
        self.last_eval_mean: float | None = None
        self.last_eval_checkpoint: str = ""
        self.last_eval_at: str = ""
        self.best_eval_mean: float | None = None
        self.best_eval_checkpoint: str = ""
        self.stop_reason: str = ""
        self.stop_reason_label: str = ""
        self.loaded_local_checkpoint: str = ""

    def _hf_stream_count(self) -> int:
        flags = (
            self.hf_stream_fable,
            self.hf_stream_sol,
            self.hf_stream_kimi,
            self.hf_stream_nemotron,
            self.hf_stream_unsolved_math,
            self.hf_stream_preference,
            self.hf_stream_fineweb,
        )
        return sum(1 for f in flags if f)

    def add_log(self, msg: str, color: str = "white") -> None:
        self.log_lines.append({"msg": msg, "color": color, "t": _now()})
        if len(self.log_lines) > 2000:
            del self.log_lines[:1000]

    def apply_checkpoint(self, ckpt: dict[str, Any]) -> None:
        self.status = str(ckpt.get("status") or "idle")
        if self.status == "streaming":
            self.status = "interrupted"
        self.config = str(ckpt.get("config") or DEFAULT_CONFIG)
        self.target_gb = float(ckpt.get("target_gb") or DEFAULT_TARGET_GB)
        self.model_params = int(ckpt.get("model_params") or DEFAULT_MODEL_PARAMS)
        self.learning_rate = float(ckpt.get("learning_rate") or DEFAULT_LR)
        self.batch_size = int(ckpt.get("batch_size") or DEFAULT_BATCH_SIZE)
        self.rows = int(ckpt.get("rows") or 0)
        self.skipped = int(ckpt.get("skipped") or 0)
        self.bytes_written = int(ckpt.get("bytes_written") or 0)
        self.stream_index = int(ckpt.get("stream_index") or 0)
        self.train_step = int(ckpt.get("train_step") or 0)
        self.last_loss = float(ckpt.get("last_loss") or 0.0)
        self.last_tok_s = float(ckpt.get("last_tok_s") or 0.0)
        self.model_ready = bool(ckpt.get("model_ready"))
        self.data_source = str(ckpt.get("data_source") or self.data_source)
        self.jsonl_file = str(ckpt.get("jsonl_file") or self.jsonl_file)
        self.epochs = int(ckpt.get("epochs") or self.epochs)
        self.epoch_cycles = bool(ckpt.get("epoch_cycles"))
        self.jsonl_total_rows = int(ckpt.get("jsonl_total_rows") or self.jsonl_total_rows)
        self.jsonl_pool_rows = int(ckpt.get("jsonl_pool_rows") or self.jsonl_pool_rows)
        self.jsonl_epoch = int(ckpt.get("jsonl_epoch") or self.jsonl_epoch)
        self.jsonl_mix_ratio = float(ckpt.get("jsonl_mix_ratio") or self.jsonl_mix_ratio)
        self.mix_local_jsonl = bool(ckpt.get("mix_local_jsonl", True))
        self.jsonl_hits = int(ckpt.get("jsonl_hits") or 0)
        self.fable_hits = int(ckpt.get("fable_hits") or 0)
        self.preference_hits = int(ckpt.get("preference_hits") or 0)
        self.sol_hits = int(ckpt.get("sol_hits") or 0)
        self.kimi_hits = int(ckpt.get("kimi_hits") or 0)
        self.math_hits = int(ckpt.get("math_hits") or 0)
        self.nemotron_hits = int(ckpt.get("nemotron_hits") or 0)
        self.swe_hits = int(ckpt.get("swe_hits") or 0)
        self.mix_local_corpora = bool(ckpt.get("mix_local_corpora"))
        self.shuffle_hf_datasets = bool(ckpt.get("shuffle_hf_datasets"))
        self.hf_stream_fable = bool(ckpt.get("hf_stream_fable", True))
        self.hf_stream_sol = bool(ckpt.get("hf_stream_sol", True))
        self.hf_stream_kimi = bool(ckpt.get("hf_stream_kimi", True))
        self.hf_stream_nemotron = bool(ckpt.get("hf_stream_nemotron", True))
        self.hf_stream_unsolved_math = bool(ckpt.get("hf_stream_unsolved_math", True))
        self.hf_stream_preference = bool(ckpt.get("hf_stream_preference", True))
        self.hf_stream_fineweb = bool(ckpt.get("hf_stream_fineweb", False))
        self.hf_dataset_id = str(ckpt.get("hf_dataset_id") or self.hf_dataset_id)
        self.tokenizer_id = str(ckpt.get("tokenizer_id") or self.tokenizer_id)
        self.multimodal_hits = int(ckpt.get("multimodal_hits") or 0)
        self.vjepa_hits = int(ckpt.get("vjepa_hits") or 0)
        self.run_goal_title = str(ckpt.get("run_goal_title") or self.run_goal_title)
        self.run_goal_detail = str(ckpt.get("run_goal_detail") or self.run_goal_detail)
        self.run_stop_primary = str(ckpt.get("run_stop_primary") or self.run_stop_primary or "bytes")
        self.run_goal_agent = str(ckpt.get("run_goal_agent") or self.run_goal_agent)
        self.run_goal_started_at = str(ckpt.get("run_goal_started_at") or self.run_goal_started_at)

    def run_goal_snapshot(self, eng: TrainEngine, target_bytes: int) -> dict[str, Any]:
        byte_pct = 0.0
        if target_bytes > 0:
            byte_pct = min(100.0, (self.bytes_written / target_bytes) * 100)
        step_cap = int(eng.total_train_steps or 0)
        steps_done = max(0, eng.step - eng.run_start_step) if eng.run_start_step else eng.step
        step_pct = min(100.0, (steps_done / step_cap) * 100) if step_cap > 0 else 0.0
        byte_met = target_bytes > 0 and self.bytes_written >= int(target_bytes * 0.999)
        step_met = step_cap > 0 and steps_done >= step_cap
        primary = self.run_stop_primary or "bytes"
        if self.status == "complete":
            if self.stop_reason_label:
                outcome = self.stop_reason or "complete"
                outcome_label = self.stop_reason_label
            elif primary == "bytes" and not byte_met and step_met:
                outcome = "partial_step_cap"
                outcome_label = "Step cap hit before byte target — set total_train_steps=0 to run to GB"
            elif byte_met:
                outcome = "byte_target_met"
                outcome_label = "Byte target reached — goal met"
            elif step_met and primary in ("steps", "chinchilla"):
                outcome = "step_cap_met"
                outcome_label = "Step / Chinchilla cap reached — goal met"
            elif step_met:
                outcome = "step_cap_only"
                outcome_label = "Stopped at step cap (byte target not reached)"
            else:
                outcome = "complete"
                outcome_label = "Run complete"
        elif self.running:
            outcome = "running"
            outcome_label = "Training in progress"
        else:
            outcome = self.status or "idle"
            outcome_label = str(self.status or "idle")
        goal_met = (
            (primary == "bytes" and byte_met)
            or (primary in ("steps", "chinchilla") and step_met and byte_met)
            or (primary == "steps" and step_met and not target_bytes)
        )
        return {
            "title": self.run_goal_title,
            "detail": self.run_goal_detail,
            "stop_primary": primary,
            "agent": self.run_goal_agent,
            "started_at": self.run_goal_started_at,
            "byte_target_gb": self.target_gb,
            "byte_progress_pct": round(byte_pct, 2),
            "byte_met": byte_met,
            "step_cap": step_cap,
            "step_progress_pct": round(step_pct, 2),
            "step_met": step_met,
            "outcome": outcome,
            "outcome_label": outcome_label,
            "goal_met": goal_met,
        }

    def progress_pct(self) -> float:
        eng = _get_engine()
        byte_pct = 0.0
        if self.target_gb:
            byte_pct = (self.bytes_written / (1024**3) / self.target_gb) * 100
        step_pct = 0.0
        if eng.total_train_steps > 0:
            step_pct = (eng.step / eng.total_train_steps) * 100
        token_pct = 0.0
        chinchilla_tokens = chinchilla_target_tokens(eng.param_count)
        if chinchilla_tokens > 0:
            token_pct = (eng.total_tokens / chinchilla_tokens) * 100
        pct = max(byte_pct, step_pct, token_pct)
        if self.data_source == "jsonl" and self.epoch_cycles and self.jsonl_total_rows > 0:
            row_pct = (self.rows / self.jsonl_total_rows) * 100
            return min(100.0, max(pct, row_pct))
        return min(100.0, pct)

    def snapshot(self) -> dict[str, Any]:
        eng = _get_engine()
        gb = self.bytes_written / (1024**3)
        pct = self.progress_pct()
        dataset_label = {
            "jsonl": f"jsonl:{self.jsonl_file}",
            "local-corpus-mix": f"jsonl:{LOCAL_CORPUS_MIX_LABEL}",
            "fable-traces": FABLE_DATASET_ID,
            "mixed": f"mixed:local+hf({self._hf_stream_count()})",
            "openassistant-rm": f"{HH_RLHF_DATASET} (RM: {RM_MODEL_ID})",
            "sol-traces": SOL_TRACES_DATASET_ID,
            "kimi-k3-traces": KIMI_K3_TRACES_DATASET_ID,
            "unsolved-math": UNSOLVED_MATH_DATASET_ID,
            "nemotron-math": NEMOTRON_MATH_DATASET_ID,
            "nemotron-swe": NEMOTRON_SWE_DATASET_ID,
            "multimodal": self.hf_dataset_id or DEFAULT_MULTIMODAL_DATASET,
            "vjepa": self.hf_dataset_id or DEFAULT_VJEPA_DATASET,
        }.get(self.data_source, DATASET_ID)
        cfg = eng.model_config
        tokens_per_step = cfg.block_size * eng.batch_size
        forward_passes = eng.grad_accum_steps if eng.micro_batch_size < eng.batch_size else 1
        loaded_ckpt = self.loaded_local_checkpoint
        if not loaded_ckpt and self.checkpoint_name.endswith(".pt"):
            loaded_ckpt = self.checkpoint_name
        if not loaded_ckpt and eng.step > 0:
            loaded_ckpt = _find_local_checkpoint_for_step(eng.step)
        return {
            "running": self.running,
            "paused": self.paused,
            "status": self.status,
            "host": HOST,
            "public_url": PUBLIC_URL,
            "dataset": dataset_label,
            "data_source": self.data_source,
            "jsonl_file": self.jsonl_file,
            "epochs": self.epochs,
            "epoch_cycles": self.epoch_cycles,
            "jsonl_mix_ratio": self.jsonl_mix_ratio,
            "mix_local_jsonl": self.mix_local_jsonl,
            "mix_local_corpora": self.mix_local_corpora,
            "jsonl_total_rows": self.jsonl_total_rows,
            "jsonl_pool_rows": self.jsonl_pool_rows,
            "jsonl_epoch": self.jsonl_epoch,
            "jsonl_hits": self.jsonl_hits,
            "fable_hits": self.fable_hits,
            "preference_hits": self.preference_hits,
            "sol_hits": self.sol_hits,
            "kimi_hits": self.kimi_hits,
            "math_hits": self.math_hits,
            "nemotron_hits": self.nemotron_hits,
            "swe_hits": self.swe_hits,
            "multimodal_hits": self.multimodal_hits,
            "vjepa_hits": self.vjepa_hits,
            "hf_dataset_id": self.hf_dataset_id,
            "tokenizer_id": self.tokenizer_id,
            "shuffle_hf_datasets": self.shuffle_hf_datasets,
            "hf_stream_fable": self.hf_stream_fable,
            "hf_stream_sol": self.hf_stream_sol,
            "hf_stream_kimi": self.hf_stream_kimi,
            "hf_stream_nemotron": self.hf_stream_nemotron,
            "hf_stream_unsolved_math": self.hf_stream_unsolved_math,
            "hf_stream_preference": self.hf_stream_preference,
            "hf_stream_fineweb": self.hf_stream_fineweb,
            "config": self.config,
            "target_gb": self.target_gb,
            "model_params": eng.param_count,
            "model_backend": getattr(eng, "model_backend", "gpt"),
            "hf_repo_id": getattr(eng, "hf_repo_id", "") or None,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "dropout": cfg.dropout,
            "weight_decay": eng.weight_decay,
            "vocab_size": cfg.vocab_size,
            "ffn_dim": cfg.ffn_dim,
            "use_flash_attention": cfg.use_flash_attention,
            "scalar_input_projection": cfg.scalar_input_projection,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "lr_schedule": eng.lr_schedule,
            "total_train_steps": eng.total_train_steps,
            "flash_attention_available": flash_attention_available(),
            "effective_batch": eng.batch_size,
            "tokens_per_step": tokens_per_step,
            "forward_passes_per_step": forward_passes,
            "learning_rate": eng.learning_rate,
            "batch_size": eng.batch_size,
            "micro_batch_size": eng.micro_batch_size,
            "grad_accum_steps": eng.grad_accum_steps,
            "warmup_steps": eng.warmup_steps,
            "mixed_precision": eng.mixed_precision,
            "save_every_n_steps": eng.save_every_n_steps,
            "rows": self.rows,
            "skipped": self.skipped,
            "bytes_written": self.bytes_written,
            "size_gb": round(gb, 4),
            "size_mb": round(self.bytes_written / (1024**2), 2),
            "progress_pct": round(pct, 2),
            "train_step": eng.step,
            "total_tokens": eng.total_tokens,
            "chinchilla_target_tokens": chinchilla_target_tokens(eng.param_count),
            "local_corpus_mix_label": LOCAL_CORPUS_MIX_LABEL,
            "last_loss": eng.last_loss,
            "last_tok_s": round(self.last_tok_s, 1),
            "step_interval_s": round(self.last_step_interval_s, 1) if self.last_step_interval_s > 0 else None,
            "mb_per_hour": round(_mb_per_hour(self.bytes_written, self.train_wall_started_at), 1)
            if self.train_wall_started_at > 0
            else None,
            "last_epoch_duration_s": round(self.last_epoch_duration_s, 1) if self.last_epoch_duration_s > 0 else None,
            "stream_epoch": self.stream_epoch or None,
            "stream_index": self.stream_index,
            "model_ready": eng.step > 0,
            "export_ready": eng.step > 0,
            "can_resume": (
                not self.running
                and self.status != "complete"
                and self.bytes_written > 0
                and self.bytes_written < int(self.target_gb * (1024**3))
                and (
                    self.data_source != "jsonl"
                    or not self.epoch_cycles
                    or self.jsonl_total_rows <= 0
                    or self.rows < self.jsonl_total_rows
                )
            ),
            "complete": self.status == "complete",
            "checkpoint_name": self.checkpoint_name,
            "loaded_checkpoint": loaded_ckpt or "",
            "checkpoint_storage": self.checkpoint_name
            if self.checkpoint_name and not self.checkpoint_name.endswith(".pt")
            else "",
            "stop_reason": self.stop_reason or None,
            "stop_reason_label": self.stop_reason_label or None,
            "run_steps_done": max(0, eng.step - eng.run_start_step) if eng.run_start_step else eng.step,
            "run_step_cap": eng.total_train_steps if eng.total_train_steps > 0 else None,
            "device": DEVICE,
            "storage": "ibm-cos",
            "voice_to_plan_url": VOICE_TO_PLAN_URL,
            "run_goal": self.run_goal_snapshot(eng, int(self.target_gb * (1024**3))),
            "eval": {
                "last_mean": self.last_eval_mean,
                "last_checkpoint": self.last_eval_checkpoint or None,
                "last_at": self.last_eval_at or None,
                "best_mean": self.best_eval_mean,
                "best_checkpoint": self.best_eval_checkpoint or None,
                "auto_on_save": AUTO_EVAL_ON_SAVE,
            },
            "gpu_vram": _gpu_vram_snapshot(),
        }

    def checkpoint_dict(self) -> dict[str, Any]:
        eng = _get_engine()
        return {
            "version": 2,
            "status": self.status,
            "config": self.config,
            "target_gb": self.target_gb,
            "model_params": self.model_params,
            "learning_rate": eng.learning_rate,
            "batch_size": eng.batch_size,
            "micro_batch_size": eng.micro_batch_size,
            "grad_accum_steps": eng.grad_accum_steps,
            "rows": self.rows,
            "skipped": self.skipped,
            "bytes_written": self.bytes_written,
            "stream_index": self.stream_index,
            "train_step": eng.step,
            "last_loss": eng.last_loss,
            "last_tok_s": self.last_tok_s,
            "model_ready": eng.step > 0,
            "model_key": MODEL_KEY,
            "dataset": DATASET_ID if self.data_source == "fineweb" else self.data_source,
            "data_source": self.data_source,
            "jsonl_file": self.jsonl_file,
            "epochs": self.epochs,
            "epoch_cycles": self.epoch_cycles,
            "jsonl_mix_ratio": self.jsonl_mix_ratio,
            "mix_local_jsonl": self.mix_local_jsonl,
            "mix_local_corpora": self.mix_local_corpora,
            "jsonl_total_rows": self.jsonl_total_rows,
            "jsonl_pool_rows": self.jsonl_pool_rows,
            "jsonl_epoch": self.jsonl_epoch,
            "jsonl_hits": self.jsonl_hits,
            "fable_hits": self.fable_hits,
            "preference_hits": self.preference_hits,
            "sol_hits": self.sol_hits,
            "kimi_hits": self.kimi_hits,
            "math_hits": self.math_hits,
            "nemotron_hits": self.nemotron_hits,
            "swe_hits": self.swe_hits,
            "multimodal_hits": self.multimodal_hits,
            "vjepa_hits": self.vjepa_hits,
            "hf_dataset_id": self.hf_dataset_id,
            "tokenizer_id": self.tokenizer_id,
            "shuffle_hf_datasets": self.shuffle_hf_datasets,
            "hf_stream_fable": self.hf_stream_fable,
            "hf_stream_sol": self.hf_stream_sol,
            "hf_stream_kimi": self.hf_stream_kimi,
            "hf_stream_nemotron": self.hf_stream_nemotron,
            "hf_stream_unsolved_math": self.hf_stream_unsolved_math,
            "hf_stream_preference": self.hf_stream_preference,
            "hf_stream_fineweb": self.hf_stream_fineweb,
            "n_layer": eng.model_config.n_layer,
            "n_head": eng.model_config.n_head,
            "n_embd": eng.model_config.n_embd,
            "block_size": eng.model_config.block_size,
            "dropout": eng.model_config.dropout,
            "weight_decay": eng.weight_decay,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_goal_title": self.run_goal_title,
            "run_goal_detail": self.run_goal_detail,
            "run_stop_primary": self.run_stop_primary,
            "run_goal_agent": self.run_goal_agent,
            "run_goal_started_at": self.run_goal_started_at,
        }


S = StreamState()
_run_lock = threading.Lock()


class ConfigApply(BaseModel):
    n_layer: bool = True
    n_head: bool = True
    n_embd: bool = True
    block_size: bool = True
    dropout: bool = False
    learning_rate: bool = True
    batch_size: bool = True
    micro_batch_size: bool = True
    grad_accum_steps: bool = True
    weight_decay: bool = False
    warmup_steps: bool = True
    mixed_precision: bool = True
    save_every_n_steps: bool = True
    ffn_dim: bool = False
    vocab_size: bool = False
    use_flash_attention: bool = True
    scalar_input_projection: bool = False
    gradient_checkpointing: bool = False
    lr_schedule: bool = True
    total_train_steps: bool = True
    epoch_cycles: bool = True
    mix_local_jsonl: bool = True
    jsonl_mix_ratio: bool = True
    shuffle_hf_datasets: bool = True
    mix_local_corpora: bool = True
    hf_streams: bool = True
    jsonl_file: bool = True
    target_gb: bool = True
    tokenizer_id: bool = True
    hf_dataset_id: bool = True
    vjepa_dataset_ids: bool = True
    vjepa_local_cache: bool = True


class StartIn(BaseModel):
    config: str = Field(default=DEFAULT_CONFIG, max_length=80)
    target_gb: float = Field(default=DEFAULT_TARGET_GB, ge=MIN_TARGET_GB, le=25.0)
    model_params: int = Field(default=DEFAULT_MODEL_PARAMS, ge=1_000_000, le=80_000_000_000)
    learning_rate: float = Field(default=DEFAULT_LR, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=8192)
    data_source: str = Field(default="mixed", max_length=40)
    hf_dataset_id: str = Field(default=DEFAULT_MULTIMODAL_DATASET, max_length=200)
    vjepa_dataset_ids: list[str] = Field(default_factory=list, max_length=24)
    vjepa_pack_rows: int = Field(default=0, ge=0, le=512)
    vjepa_pack_min_chars: int = Field(default=4000, ge=500, le=200_000)
    vjepa_local_cache: bool = False
    tokenizer_id: str = Field(default=DEFAULT_TOKENIZER_ID, max_length=120)
    jsonl_file: str = Field(default="4563-curated.jsonl", max_length=200)
    epochs: int = Field(default=1, ge=1, le=500)
    epoch_cycles: bool = False
    jsonl_mix_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    mix_local_jsonl: bool = True
    mix_local_corpora: bool = True
    shuffle_hf_datasets: bool = False
    hf_stream_fable: bool = True
    hf_stream_sol: bool = True
    hf_stream_kimi: bool = True
    hf_stream_nemotron: bool = True
    hf_stream_unsolved_math: bool = True
    hf_stream_preference: bool = True
    hf_stream_fineweb: bool = False
    english_only: bool = True
    min_token_count: int = Field(default=32, ge=0, le=10_000)
    n_layer: int = Field(default=DEFAULT_N_LAYER, ge=2, le=128)
    n_head: int = Field(default=DEFAULT_N_HEAD, ge=2, le=128)
    n_embd: int = Field(default=DEFAULT_N_EMBD, ge=128, le=8192)
    block_size: int = Field(default=DEFAULT_BLOCK_SIZE, ge=128, le=8192)
    vocab_size: int = Field(default=50257, ge=1000, le=256000)
    ffn_dim: int | None = Field(default=None, ge=128, le=16384)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    use_flash_attention: bool = True
    scalar_input_projection: bool = False
    gradient_checkpointing: bool = False
    lr_schedule: str = Field(default="cosine", pattern="^(constant|warmup|cosine)$")
    total_train_steps: int = Field(default=DEFAULT_TOTAL_TRAIN_STEPS, ge=0, le=10_000_000)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    micro_batch_size: int | None = Field(default=DEFAULT_MICRO_BATCH, ge=1, le=8192)
    grad_accum_steps: int | None = Field(default=None, ge=1, le=2048)
    warmup_steps: int = Field(default=DEFAULT_WARMUP_STEPS, ge=0, le=100_000)
    mixed_precision: str = Field(default=DEFAULT_MIXED_PRECISION, pattern="^(fp32|fp16|bf16)$")
    save_every_n_steps: int = Field(default=DEFAULT_SAVE_EVERY_N_STEPS, ge=0, le=100_000)
    apply: ConfigApply = Field(default_factory=ConfigApply)
    resume: bool = True
    run_goal_title: str = Field(default="", max_length=200)
    run_goal_detail: str = Field(default="", max_length=2000)
    run_stop_primary: str = Field(default="bytes", pattern="^(bytes|steps|chinchilla)$")
    run_goal_agent: str = Field(default="", max_length=80)
    stream_skip_rows: int = Field(default=0, ge=0, le=100_000_000)


def _load_checkpoint() -> dict[str, Any] | None:
    cos = _cos_client()
    if not cos:
        return None
    try:
        obj = cos.get_object(Bucket=COS_BUCKET, Key=CHECKPOINT_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def _params_m_slug(param_count: int) -> str:
    return f"{round(param_count / 1_000_000)}m"


def _model_family_slug(eng: TrainEngine) -> str:
    loaded = (S.checkpoint_name or "").removesuffix(".pt")
    meta_tag = str((eng.session_meta or {}).get("model_tag") or "").strip()
    if meta_tag:
        return re.sub(r"[^a-z0-9]+", "-", meta_tag.lower()).strip("-")[:32]
    if loaded:
        low = loaded.lower()
        if low.startswith("distilgpt2"):
            return "distilgpt2"
        if low.startswith("gpt2"):
            return "gpt2"
        if low.startswith("imported-"):
            return re.sub(r"[^a-z0-9]+", "-", loaded[9:].lower()).strip("-")[:32]
        if low.startswith("nextaura-"):
            return "nextaura"
        return re.sub(r"[^a-z0-9]+", "-", loaded.split("-")[0].lower()).strip("-")[:32] or "nextaura"
    cfg = str(S.config or "").strip().lower()
    if "distilgpt2" in cfg:
        return "distilgpt2"
    if cfg and cfg not in ("sample-10bt", "mixed"):
        return re.sub(r"[^a-z0-9]+", "-", cfg).strip("-")[:32]
    return "nextaura"


def _checkpoint_basename(eng: TrainEngine, step: int | None = None) -> str:
    step_n = int(step if step is not None else eng.step)
    family = _model_family_slug(eng)
    return f"{family}-{_params_m_slug(eng.param_count)}-step{step_n}.pt"


def _hf_streams_label() -> str:
    """Human label for active HF streams (mixed mode) or trace-only sources."""
    ds = (S.data_source or "").strip()
    if ds == "sol-traces":
        return f"HF coding · {SOL_TRACES_DATASET_ID}"
    if ds == "kimi-k3-traces":
        return f"HF coding · {KIMI_K3_TRACES_DATASET_ID}"
    if ds != "mixed":
        return ds
    names: list[str] = []
    for enabled, label in (
        (S.hf_stream_fable, "fable"),
        (S.hf_stream_sol, "sol"),
        (S.hf_stream_kimi, "kimi"),
        (S.hf_stream_nemotron, "nemotron"),
        (S.hf_stream_unsolved_math, "math"),
        (S.hf_stream_preference, "pref"),
        (S.hf_stream_fineweb, "fineweb"),
    ):
        if enabled:
            names.append(label)
    if not names:
        return "mixed (no HF streams enabled)"
    return "HF [" + "+".join(names) + "]"


def _training_mode_for_source(data_source: str) -> str:
    ds = (data_source or "").strip()
    goal = f"{S.run_goal_title or ''} {S.run_goal_detail or ''}".lower()
    if any(k in goal for k in ("sft", "finetune", "fine-tune", "finetune")):
        return "finetune"
    if S.loaded_local_checkpoint and ds in (
        "mixed",
        "sol-traces",
        "kimi-k3-traces",
        "fable-traces",
        "fineweb",
    ):
        return "finetune"
    if ds in ("sft-intelligence", "jsonl", "local-corpus-mix", "nemotron-swe"):
        return "finetune"
    if ds in SFT_TOPIC_SOURCES or ds in DISTILL_TOPIC_SOURCES or ds == "distill-live":
        return "finetune"
    return "pretrain"


def _suggested_infer_prompts(data_source: str, hf_dataset_id: str = "") -> list[str]:
    ds = data_source or ""
    if ds == "multimodal":
        return [
            "Describe the scene and list the key objects.",
            "What is the geometric relationship in this problem?",
            "Step-by-step: how would you solve this visual reasoning task?",
        ]
    if ds == "vjepa":
        return [
            "Summarize the motion or action in this clip.",
            "What happens next in this video latent sequence?",
        ]
    if ds in DISTILL_TOPIC_SOURCES or ds == "distill-merged" or ds == "distill-live":
        return [
            "Write a short Python function that reverses a linked list.",
            "Explain buoyancy for a block submerged in water.",
            "Solve: if x² + 5x + 6 = 0, what are the roots?",
        ]
    if ds in SFT_TOPIC_SOURCES or ds == "sft-intelligence":
        return [
            "What should the agent do next to complete this task?",
            "Critique this plan and suggest a safer alternative.",
        ]
    if ds == "nemotron-swe":
        return [
            "Given this repo issue, list the files you would inspect first and why.",
            "Write a minimal patch plan for a failing CI test without changing unrelated code.",
        ]
    if ds in ("sol-traces", "kimi-k3-traces") or (ds == "mixed" and not hf_dataset_id):
        return [
            "Write a Python function that reverses a linked list in O(n) time.",
            "Debug this code: def add(a,b): return a - b  # should add",
            "Explain your reasoning step-by-step for a binary search implementation.",
        ]
    if ds == "mixed" or ds.endswith("-traces"):
        return [
            "Continue this reasoning trace in one paragraph.",
            "What is the main claim and supporting evidence?",
        ]
    return [
        "Complete this sentence in one coherent paragraph.",
        "Explain the topic as if teaching a beginner.",
    ]


def _infer_context(eng: TrainEngine) -> dict[str, Any]:
    """Rich context for Stage 2 — what we trained, why we stopped, what to test."""
    meta = _infer_meta(eng)
    cfg = eng.model_config
    params = int(eng.param_count or 0)
    chinchilla_gb = chinchilla_target_gb(params) if params else 0.0
    target_gb = float(S.target_gb or 0)
    mode = _training_mode_for_source(S.data_source)
    bytes_mb = round(S.bytes_written / (1024**2), 2) if S.bytes_written else 0.0
    ratio = (target_gb / chinchilla_gb) if chinchilla_gb > 0 else None
    if ratio is not None:
        if ratio <= 0.12:
            scale_note = f"smoke run ({ratio:.2f}× Chinchilla optimal {chinchilla_gb:.2f} GB) — expect weak inference"
        elif ratio < 0.4:
            scale_note = f"short run ({ratio:.2f}× Chinchilla) — sanity check only"
        elif ratio < 0.9:
            scale_note = f"partial pretrain ({ratio:.2f}× Chinchilla)"
        else:
            scale_note = f"near Chinchilla optimal ({ratio:.2f}×)"
    else:
        scale_note = "byte target not set"
    if S.running:
        weights = "live GPU weights (training in progress)"
    elif eng.step > 0 and not S.checkpoint_name:
        weights = "in-memory weights from current/last run (not a saved .pt)"
    elif S.checkpoint_name:
        weights = f"loaded checkpoint {S.checkpoint_name}"
    else:
        weights = "none — train or load a .pt first"
    loaded_ckpt = S.loaded_local_checkpoint
    if not loaded_ckpt and S.checkpoint_name.endswith(".pt"):
        loaded_ckpt = S.checkpoint_name
    train_ckpt = S.checkpoint_name or (_checkpoint_basename(eng) if eng.step > 0 else "")
    ckpt_mismatch = bool(
        loaded_ckpt
        and train_ckpt
        and loaded_ckpt.endswith(".pt")
        and train_ckpt.endswith(".pt")
        and loaded_ckpt != train_ckpt
    )
    hits: list[str] = []
    if S.multimodal_hits:
        hits.append(f"mm {S.multimodal_hits}")
    if S.vjepa_hits:
        hits.append(f"vjepa {S.vjepa_hits}")
    if S.fable_hits:
        hits.append(f"fable {S.fable_hits}")
    if S.sol_hits:
        hits.append(f"sol {S.sol_hits}")
    if S.kimi_hits:
        hits.append(f"kimi {S.kimi_hits}")
    if S.nemotron_hits:
        hits.append(f"nemotron {S.nemotron_hits}")
    if S.swe_hits:
        hits.append(f"swe {S.swe_hits}")
    if S.math_hits:
        hits.append(f"math {S.math_hits}")
    if S.preference_hits:
        hits.append(f"pref {S.preference_hits}")
    if S.jsonl_hits:
        hits.append(f"local {S.jsonl_hits}")
    return {
        **meta,
        "hf_stream_label": _hf_streams_label(),
        "training_mode": mode,
        "training_mode_label": "Pretraining" if mode == "pretrain" else "Fine-tuning",
        "is_smoke": target_gb > 0 and target_gb <= 0.1,
        "target_gb": target_gb,
        "bytes_consumed_mb": bytes_mb,
        "bytes_target_mb": round(target_gb * 1024, 1),
        "chinchilla_optimal_gb": chinchilla_gb,
        "chinchilla_ratio": round(ratio, 3) if ratio is not None else None,
        "byte_scale_note": scale_note,
        "stop_reason": S.stop_reason or None,
        "stop_reason_label": S.stop_reason_label or None,
        "session_status": S.status,
        "tokenizer_id": eng.tokenizer_id,
        "vocab_size": cfg.vocab_size if cfg else None,
        "block_size": cfg.block_size if cfg else None,
        "scalar_input_projection": cfg.scalar_input_projection if cfg else False,
        "weights_source": weights,
        "loaded_checkpoint": loaded_ckpt or None,
        "training_checkpoint": train_ckpt or None,
        "checkpoint_mismatch": ckpt_mismatch,
        "stream_hits": " · ".join(hits) if hits else None,
        "eval_last_mean": S.last_eval_mean,
        "eval_best_mean": S.best_eval_mean,
        "eval_best_checkpoint": S.best_eval_checkpoint or None,
        "train_steps": eng.step,
        "chinchilla_train_steps": chinchilla_train_steps(
            params,
            block_size=cfg.block_size if cfg else DEFAULT_BLOCK_SIZE,
            batch_size=eng.batch_size,
        )
        if params
        else None,
        "total_train_steps_cap": eng.total_train_steps or 0,
        "jsonl_file": S.jsonl_file or None,
        "run_goal_title": S.run_goal_title or None,
        "suggested_prompts": _suggested_infer_prompts(S.data_source, S.hf_dataset_id or ""),
    }


def _infer_meta(eng: TrainEngine) -> dict[str, Any]:
    """Live inference labels from the loaded / in-memory training checkpoint only."""
    cfg = eng.model_config
    params = int(eng.param_count or 0)
    params_m = round(params / 1_000_000, 1)
    family = _model_family_slug(eng)
    slug = _params_m_slug(params) if params > 0 else "0m"
    ckpt = S.checkpoint_name or (_checkpoint_basename(eng) if eng.step > 0 else "")
    if ckpt.startswith("cos:") and eng.step > 0:
        ckpt = _checkpoint_basename(eng)
    arch = f"{cfg.n_layer}L×{cfg.n_embd}d" if cfg else ""
    if getattr(eng, "model_backend", "gpt") == "hf":
        repo = (getattr(eng, "hf_repo_id", "") or "").strip()
        short = repo.split("/")[-1] if repo else "causal LM"
        arch = f"HF:{short}"
    data_src = S.data_source or ""
    hf_ds = S.hf_dataset_id or ""
    if data_src in ("mixed", "sol-traces", "kimi-k3-traces"):
        stream = _hf_streams_label()
    elif data_src == "multimodal" and hf_ds:
        stream = f"multimodal:{hf_ds.split('/')[-1]}"
    elif data_src == "vjepa" and hf_ds:
        stream = f"vjepa:{hf_ds.split('/')[-1]}"
    else:
        stream = data_src
    model_id = f"{family}-{slug}"
    model_label = " · ".join(
        x
        for x in [
            f"{params_m:g}M",
            arch,
            stream,
            ckpt,
        ]
        if x
    )
    return {
        "model": model_id,
        "model_label": model_label,
        "model_params": params,
        "model_params_m": params_m,
        "n_layer": cfg.n_layer if cfg else None,
        "n_embd": cfg.n_embd if cfg else None,
        "block_size": cfg.block_size if cfg else None,
        "data_source": data_src or None,
        "hf_dataset_id": hf_ds or None,
        "checkpoint_name": ckpt or None,
        "training": S.running,
    }


def _session_meta_snapshot(eng: TrainEngine) -> dict[str, Any]:
    return {
        "data_source": S.data_source,
        "jsonl_file": S.jsonl_file or None,
        "target_gb": S.target_gb,
        "jsonl_epochs": S.jsonl_epoch or None,
        "jsonl_total_rows": S.jsonl_total_rows or None,
        "bytes_written_mb": round(S.bytes_written / (1024**2), 2) if S.bytes_written else None,
        "model_tag": _model_family_slug(eng),
        "config": S.config or None,
    }


def _attach_session_meta(eng: TrainEngine) -> None:
    eng.session_meta = _session_meta_snapshot(eng)


def _auto_record_run(status: str) -> None:
    try:
        eng = _get_engine()
        cfg = eng.model_config
        train_steps = max(0, eng.step - eng.run_start_step) if eng.run_start_step else 0
        goal = S.run_goal_snapshot(eng, int(S.target_gb * (1024**3)))
        record_training_finish(
            DATA_DIR,
            status=status,
            checkpoint_name=S.checkpoint_name or _checkpoint_basename(eng),
            step=eng.step,
            loss=eng.last_loss,
            data_source=S.data_source,
            target_gb=S.target_gb,
            model_params=eng.param_count,
            n_embd=cfg.n_embd,
            vocab_size=cfg.vocab_size,
            jsonl_epochs=S.jsonl_epoch,
            jsonl_file=S.jsonl_file,
            learning_rate=eng.learning_rate,
            base_step=eng.run_start_step if eng.step > eng.run_start_step else None,
            train_steps=train_steps,
            agent=S.run_goal_agent or None,
            run_goal_title=goal.get("title") or None,
            run_goal_outcome=goal.get("outcome_label") or None,
            auto=True,
        )
    except Exception:
        pass


def _save_checkpoint(force: bool = False) -> None:
    cos = _cos_client()
    if not cos:
        return
    cos.put_object(
        Bucket=COS_BUCKET,
        Key=CHECKPOINT_KEY,
        Body=json.dumps(S.checkpoint_dict(), indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _save_model(force: bool = False) -> None:
    eng = _get_engine()
    if not force and not should_save_model(eng.step, eng.save_every_n_steps):
        return
    if eng.step == 0:
        return
    _attach_session_meta(eng)
    eng.checkpoint_root = _ensure_checkpoint_dir()
    payload = eng.state_bytes()
    local_name = _checkpoint_basename(eng)
    local_path = os.path.join(_ensure_checkpoint_dir(), local_name)
    with open(local_path, "wb") as out:
        out.write(payload)
    S.checkpoint_name = local_name
    S.loaded_local_checkpoint = local_name
    cos = _cos_client()
    if cos:
        cos.put_object(
            Bucket=COS_BUCKET,
            Key=MODEL_KEY,
            Body=payload,
            ContentType="application/octet-stream",
        )
    if upcloud_configured():
        try:
            mirror = upload_checkpoint_file(local_path, name=local_name)
            if mirror.get("ok"):
                S.add_log(f"[CYAN] archived to UpCloud · {mirror.get('key')}", "cyan")
        except Exception as exc:
            S.add_log(f"[YELLOW] UpCloud archive failed: {exc}", "yellow")
    S.add_log(f"[CYAN] model checkpoint saved step {eng.step} · {local_name}", "cyan")
    _prune_old_checkpoints()


MAX_CHECKPOINTS_KEEP = int(os.environ.get("MAX_CHECKPOINTS_KEEP", "5"))


def _prune_old_checkpoints() -> None:
    """Auto-rotate local .pt files after each save — keep latest N + loaded.

    Prevents Errno 28 disk-full during long runs (app8 hit 100% at 31 ckpts).
    Safe while training: weights live in GPU memory; never deletes S.checkpoint_name.
    """
    if MAX_CHECKPOINTS_KEEP <= 0:
        return
    try:
        root = _ensure_checkpoint_dir()
        ckpts = []
        for name in os.listdir(root):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(root, name)
            if os.path.isfile(path):
                ckpts.append((os.path.getmtime(path), name, path))
        ckpts.sort(reverse=True)  # newest first
        keep_names = {n for _, n, _ in ckpts[:MAX_CHECKPOINTS_KEEP]}
        if S.checkpoint_name:
            keep_names.add(S.checkpoint_name)
        if S.loaded_local_checkpoint:
            keep_names.add(S.loaded_local_checkpoint)
        freed = 0
        deleted = []
        for _, name, path in ckpts:
            if name in keep_names:
                continue
            try:
                freed += os.path.getsize(path)
                os.remove(path)
                deleted.append(name)
            except OSError:
                pass
        if deleted:
            S.add_log(
                f"[CYAN] auto-pruned {len(deleted)} old checkpoint(s) · freed {freed / (1024**2):.0f} MB · keeping {len(keep_names)}",
                "cyan",
            )
    except Exception as exc:
        S.add_log(f"[YELLOW] checkpoint auto-prune failed: {exc}", "yellow")
    if AUTO_EVAL_ON_SAVE:
        if S.running:
            S.pending_save_eval = (local_name, eng.step)
        else:
            threading.Thread(
                target=_run_save_eval,
                args=(local_name, eng.step, False),
                daemon=True,
            ).start()


def _load_model_from_cos() -> bool:
    cos = _cos_client()
    if not cos:
        return False
    try:
        obj = cos.get_object(Bucket=COS_BUCKET, Key=MODEL_KEY)
        _get_engine().load_state_bytes(obj["Body"].read())
        return True
    except Exception:
        return False


def _hydrate_from_cos() -> None:
    if S.skip_cos_hydrate or S.running:
        return
    if S.checkpoint_name.endswith(".pt"):
        return
    ckpt = _load_checkpoint()
    if ckpt and not S.running:
        S.apply_checkpoint(ckpt)
        if _load_model_from_cos():
            S.checkpoint_name = "cos:" + MODEL_KEY


def _safe_checkpoint_name(name: str) -> str:
    base = os.path.basename(name.strip())
    if not base or base in {".", ".."}:
        raise HTTPException(400, "Invalid checkpoint filename")
    if not base.endswith(".pt"):
        raise HTTPException(400, "Checkpoint must be a .pt file")
    if any(c in base for c in ("/", "\\", "\0")):
        raise HTTPException(400, "Invalid checkpoint filename")
    return base


def _checkpoint_path(name: str) -> str:
    safe = _safe_checkpoint_name(name)
    root = os.path.abspath(CHECKPOINT_DIR)
    path = os.path.abspath(os.path.join(root, safe))
    if not path.startswith(root + os.sep) and path != root:
        raise HTTPException(400, "Invalid checkpoint path")
    return path


def _ensure_checkpoint_dir() -> str:
    root = os.path.abspath(CHECKPOINT_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def _import_checkpoint_bytes(payload: bytes, filename: str, *, skip_running_check: bool = False) -> dict[str, Any]:
    if S.running and not skip_running_check:
        raise HTTPException(409, "Stop training before loading a checkpoint")
    if len(payload) > MAX_CHECKPOINT_MB * 1024 * 1024:
        raise HTTPException(413, f"Checkpoint too large (max {MAX_CHECKPOINT_MB} MB)")
    if len(payload) < 1024:
        raise HTTPException(400, "Checkpoint file is too small")

    global _engine
    with _engine_lock:
        _engine = TrainEngine(device=DEVICE)
        eng = _engine
        try:
            eng.load_state_bytes(payload, checkpoint_root=_ensure_checkpoint_dir())
        except Exception as exc:
            _engine = None
            raise HTTPException(400, f"Failed to load checkpoint: {exc}") from exc

    if DEVICE == "cuda":
        import torch

        torch.cuda.empty_cache()

    S.checkpoint_name = _safe_checkpoint_name(filename)
    S.loaded_local_checkpoint = S.checkpoint_name
    S.stop_reason = ""
    S.stop_reason_label = ""
    S.train_step = eng.step
    S.last_loss = eng.last_loss
    S.learning_rate = eng.learning_rate
    S.batch_size = eng.batch_size
    S.model_ready = eng.step > 0
    S.model_params = eng.param_count
    S.bytes_written = 0
    S.rows = 0
    S.skipped = 0
    S.stream_index = 0
    S.jsonl_hits = 0
    S.jsonl_epoch = 0
    S.skip_cos_hydrate = True
    if not S.running and S.status not in ("streaming", "stopping"):
        S.status = "loaded"
    S.add_log(
        f"[GREEN] loaded {S.checkpoint_name} · step {eng.step:,} · loss {eng.last_loss:.4f} · lr={eng.learning_rate:g} · batch={eng.batch_label()}",
        "green",
    )
    try:
        cfg = eng.model_config
        if not has_run_record(DATA_DIR, S.checkpoint_name):
            record_training_finish(
            DATA_DIR,
            status="import",
            checkpoint_name=S.checkpoint_name,
            step=eng.step,
            loss=eng.last_loss,
            data_source=S.data_source or "import",
            target_gb=S.target_gb,
            model_params=eng.param_count,
            n_embd=cfg.n_embd,
            vocab_size=cfg.vocab_size,
            learning_rate=eng.learning_rate,
            auto=True,
        )
    except Exception:
        pass
    return {
        "ok": True,
        "checkpoint_name": S.checkpoint_name,
        "step": eng.step,
        "loss": eng.last_loss,
        "learning_rate": eng.learning_rate,
        "batch_size": eng.batch_size,
    }


def _list_local_checkpoints() -> list[dict[str, Any]]:
    root = _ensure_checkpoint_dir()
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        out.append(
            {
                "name": name,
                "source": "local",
                "size_mb": round(os.path.getsize(path) / (1024**2), 2),
                "loaded": name == S.checkpoint_name,
            }
        )
    return out


def _infer_model_catalog() -> list[dict[str, Any]]:
    """Unified list for inference model switcher (local disk + UpCloud archive)."""
    local_names = {c["name"] for c in _list_local_checkpoints()}
    catalog: list[dict[str, Any]] = []
    for ckpt in _list_local_checkpoints():
        catalog.append(
            {
                "id": f"local:{ckpt['name']}",
                "name": ckpt["name"],
                "source": "local",
                "size_mb": ckpt["size_mb"],
                "loaded": ckpt["loaded"],
                "label": f"{ckpt['name']} (local · {ckpt['size_mb']} MB)",
            }
        )
    for remote in list_remote_checkpoints():
        name = remote["name"]
        if name in local_names:
            continue
        catalog.append(
            {
                "id": f"upcloud:{name}",
                "name": name,
                "source": "upcloud",
                "key": remote.get("key"),
                "size_mb": remote.get("size_mb"),
                "loaded": False,
                "label": f"{name} (UpCloud · {remote.get('size_mb')} MB)",
            }
        )
    catalog.sort(key=lambda r: r["name"], reverse=True)
    return catalog


def _checkpoint_disk_summary() -> dict[str, Any]:
    ckpts = _list_local_checkpoints()
    return {
        "count": len(ckpts),
        "total_mb": round(sum(c["size_mb"] for c in ckpts), 2),
    }


def _delete_checkpoint_files(names: list[str]) -> dict[str, Any]:
    if S.running:
        raise HTTPException(409, "Stop training before deleting checkpoints")
    deleted: list[str] = []
    skipped: list[str] = []
    freed = 0
    for name in names:
        safe = _safe_checkpoint_name(name)
        if safe == S.checkpoint_name:
            skipped.append(safe)
            continue
        path = _checkpoint_path(safe)
        if not os.path.isfile(path):
            skipped.append(safe)
            continue
        freed += os.path.getsize(path)
        os.remove(path)
        deleted.append(safe)
    return {
        "deleted": deleted,
        "skipped": skipped,
        "freed_mb": round(freed / (1024**2), 2),
    }


def _release_gpu_memory() -> None:
    """Drop CUDA allocator cache after OOM, reset, or stop."""
    import gc

    gc.collect()
    if DEVICE != "cuda":
        return
    import torch

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def _find_local_checkpoint_for_step(step: int) -> str:
    if step <= 0:
        return ""
    root = _ensure_checkpoint_dir()
    needle = f"step{step}.pt"
    for name in sorted(os.listdir(root), reverse=True):
        if name.endswith(".pt") and needle in name:
            return name
    return ""


def _reset_training_session() -> dict[str, Any]:
    global _engine
    if S.running:
        raise HTTPException(409, "Stop training before resetting session")
    with _engine_lock:
        _engine = TrainEngine(device=DEVICE)
    S.checkpoint_name = ""
    S.loaded_local_checkpoint = ""
    S.stop_reason = ""
    S.stop_reason_label = ""
    S.status = "idle"
    S.train_step = 0
    S.last_loss = 0.0
    S.model_ready = False
    S.bytes_written = 0
    S.rows = 0
    S.skipped = 0
    S.stream_index = 0
    S.jsonl_hits = 0
    S.fable_hits = 0
    S.preference_hits = 0
    S.sol_hits = 0
    S.kimi_hits = 0
    S.math_hits = 0
    S.nemotron_hits = 0
    S.swe_hits = 0
    S.error = ""
    S.skip_cos_hydrate = True
    _release_gpu_memory()
    S.add_log("[YELLOW] session reset — in-memory weights cleared (local .pt files kept)", "yellow")
    return {"ok": True, "status": S.status}


def _presign(key: str, hours: int = 24) -> str:
    cos = _cos_client()
    if not cos:
        return ""
    return cos.generate_presigned_url(
        "get_object",
        Params={"Bucket": COS_BUCKET, "Key": key},
        ExpiresIn=hours * 3600,
    )


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        return "—"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _mb_per_hour(bytes_written: int, started_at: float) -> float:
    if started_at <= 0:
        return 0.0
    elapsed = time.time() - started_at
    if elapsed < 1.0:
        return 0.0
    return (bytes_written / (1024**2)) / (elapsed / 3600.0)


def _speed_suffix() -> str:
    parts: list[str] = []
    if S.last_step_interval_s > 0:
        parts.append(f"{_fmt_duration(S.last_step_interval_s)}/step")
    mb_hr = _mb_per_hour(S.bytes_written, S.train_wall_started_at)
    if mb_hr > 0:
        parts.append(f"{mb_hr:.1f} MB/hr")
    return (" · " + " · ".join(parts)) if parts else ""


def _fmt_consumed(n_bytes: int, target_gb: float) -> str:
    mb = n_bytes / (1024**2)
    gb = n_bytes / (1024**3)
    target_mb = target_gb * 1024
    if target_gb < 1:
        return f"{mb:.2f} MB / {target_mb:.1f} MB"
    if gb < 0.01:
        return f"{mb:.2f} MB / {target_gb:g} GB"
    return f"{gb:.3f}/{target_gb:g} GB"


def _fmt_target(target_gb: float) -> str:
    if target_gb < 1:
        return f"{target_gb * 1024:.1f} MB"
    return f"{target_gb:g} GB"


def _training_target_reached(eng: TrainEngine, target_bytes: int) -> bool:
    if S.bytes_written >= target_bytes or S.bytes_written >= int(target_bytes * 0.999):
        return True
    if eng.total_train_steps > 0 and eng.step >= eng.run_start_step + eng.total_train_steps:
        return True
    chinchilla_gb = chinchilla_target_gb(eng.param_count)
    if S.target_gb >= chinchilla_gb * 0.5:
        chinchilla_tokens = chinchilla_target_tokens(eng.param_count)
        if eng.total_tokens >= chinchilla_tokens:
            return True
    return False


def _accept_row(row: dict[str, Any], english_only: bool, min_token_count: int) -> bool:
    text = row.get("text")
    if not isinstance(text, str) or len(text.strip()) < 80:
        return False
    if english_only:
        lang = str(row.get("language") or "").lower()
        score = row.get("language_score")
        if lang and lang != "en":
            return False
        if isinstance(score, (int, float)) and score < 0.85:
            return False
    tok = row.get("token_count")
    if isinstance(tok, int) and tok < min_token_count:
        return False
    return True


def _safe_jsonl_name(name: str) -> str:
    base = os.path.basename(name.strip())
    if not base or base in {".", ".."}:
        raise HTTPException(400, "Invalid data filename")
    if not base.endswith(".jsonl"):
        raise HTTPException(400, "Data file must be a .jsonl file")
    if any(c in base for c in ("/", "\\", "\0")):
        raise HTTPException(400, "Invalid data filename")
    return base


def _ensure_data_dir() -> str:
    root = os.path.abspath(DATA_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def _data_path(name: str) -> str:
    safe = _safe_jsonl_name(name)
    root = os.path.abspath(DATA_DIR)
    path = os.path.abspath(os.path.join(root, safe))
    if not path.startswith(root + os.sep) and path != root:
        raise HTTPException(400, "Invalid data path")
    return path


def _list_local_data_files() -> list[dict[str, Any]]:
    root = _ensure_data_dir()
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        out.append(
            {
                "name": name,
                "size_mb": round(os.path.getsize(path) / (1024**2), 2),
                "rows": _count_jsonl_rows(path),
            }
        )
    return out


def _count_jsonl_rows(path: str) -> int:
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _model_config_from_body(body: StartIn) -> ModelConfig:
    if body.n_embd % body.n_head != 0:
        raise ValueError(f"n_embd ({body.n_embd}) must be divisible by n_head ({body.n_head})")
    ffn = body.ffn_dim if body.ffn_dim is not None else 4 * body.n_embd
    return ModelConfig(
        n_layer=body.n_layer,
        n_head=body.n_head,
        n_embd=body.n_embd,
        block_size=body.block_size,
        vocab_size=body.vocab_size,
        dropout=body.dropout,
        ffn_dim=ffn,
        use_flash_attention=body.use_flash_attention,
        scalar_input_projection=body.scalar_input_projection,
        gradient_checkpointing=body.gradient_checkpointing,
    )


def _merge_start_body(body: StartIn) -> StartIn:
    """When apply.* is false, keep values from the loaded engine / stream state."""
    eng = _get_engine()
    cfg = eng.model_config
    a = body.apply
    data = body.model_dump()
    if getattr(eng, "model_backend", "gpt") == "hf":
        data["n_layer"] = cfg.n_layer
        data["n_head"] = cfg.n_head
        data["n_embd"] = cfg.n_embd
        data["block_size"] = cfg.block_size
        data["dropout"] = cfg.dropout
        data["ffn_dim"] = cfg.ffn_dim
        data["vocab_size"] = cfg.vocab_size
        data["use_flash_attention"] = cfg.use_flash_attention
        data["scalar_input_projection"] = False
        data["gradient_checkpointing"] = cfg.gradient_checkpointing
        data["tokenizer_id"] = eng.tokenizer_id
        data["model_params"] = eng.param_count
        return StartIn(**data)
    if not a.n_layer:
        data["n_layer"] = cfg.n_layer
    if not a.n_head:
        data["n_head"] = cfg.n_head
    if not a.n_embd:
        data["n_embd"] = cfg.n_embd
    if not a.block_size:
        data["block_size"] = cfg.block_size
    if not a.dropout:
        data["dropout"] = cfg.dropout
    if not a.ffn_dim:
        data["ffn_dim"] = cfg.ffn_dim
    if not a.vocab_size:
        data["vocab_size"] = cfg.vocab_size
    if not a.use_flash_attention:
        data["use_flash_attention"] = cfg.use_flash_attention
    if not a.scalar_input_projection:
        data["scalar_input_projection"] = cfg.scalar_input_projection
    if not a.gradient_checkpointing:
        data["gradient_checkpointing"] = cfg.gradient_checkpointing
    if not a.learning_rate:
        data["learning_rate"] = eng.learning_rate
    if not a.batch_size:
        data["batch_size"] = eng.batch_size
    if not a.micro_batch_size:
        data["micro_batch_size"] = None
    if not a.grad_accum_steps:
        data["grad_accum_steps"] = None
    if not a.weight_decay:
        data["weight_decay"] = eng.weight_decay
    if not a.warmup_steps:
        data["warmup_steps"] = eng.warmup_steps
    if not a.mixed_precision:
        data["mixed_precision"] = eng.mixed_precision
    if not a.save_every_n_steps:
        data["save_every_n_steps"] = eng.save_every_n_steps
    if not a.lr_schedule:
        data["lr_schedule"] = eng.lr_schedule
    if not a.total_train_steps:
        data["total_train_steps"] = eng.total_train_steps
    if not a.target_gb:
        data["target_gb"] = S.target_gb or DEFAULT_TARGET_GB
    if not a.epoch_cycles:
        data["epoch_cycles"] = S.epoch_cycles
        data["epochs"] = S.epochs
    if not a.mix_local_jsonl:
        data["mix_local_jsonl"] = S.mix_local_jsonl
    if not a.jsonl_mix_ratio:
        data["jsonl_mix_ratio"] = S.jsonl_mix_ratio
    if not a.shuffle_hf_datasets:
        data["shuffle_hf_datasets"] = S.shuffle_hf_datasets
    if not a.mix_local_corpora:
        data["mix_local_corpora"] = S.mix_local_corpora
    if not a.jsonl_file:
        data["jsonl_file"] = S.jsonl_file
    if not a.hf_streams:
        data["hf_stream_fable"] = S.hf_stream_fable
        data["hf_stream_sol"] = S.hf_stream_sol
        data["hf_stream_kimi"] = S.hf_stream_kimi
        data["hf_stream_nemotron"] = S.hf_stream_nemotron
        data["hf_stream_unsolved_math"] = S.hf_stream_unsolved_math
        data["hf_stream_preference"] = S.hf_stream_preference
        data["hf_stream_fineweb"] = S.hf_stream_fineweb
    return StartIn(**data)


def _structural_model_config(cfg: ModelConfig) -> dict[str, Any]:
    data = cfg.to_dict()
    data.pop("gradient_checkpointing", None)
    return data


def _sync_gradient_checkpointing(eng: TrainEngine, enabled: bool) -> None:
    from model import Block

    flag = bool(enabled)
    eng.model_config.gradient_checkpointing = flag
    for module in eng.model.modules():
        if isinstance(module, Block):
            module.cfg.gradient_checkpointing = flag


def _arch_mismatch_msg(loaded: ModelConfig, requested: ModelConfig) -> str:
    def _brief(c: ModelConfig) -> str:
        ffn = c.ffn_dim or c.n_embd * 4
        return (
            f"{c.n_layer}L×{c.n_embd}d×block{c.block_size} "
            f"vocab{c.vocab_size} ffn{ffn} scalar={c.scalar_input_projection}"
        )

    return (
        f"Architecture mismatch: checkpoint is {_brief(loaded)} "
        f"but request is {_brief(requested)}"
    )


def _min_accept_chars(body: StartIn) -> int:
    if body.data_source == "vjepa":
        return 20
    if body.data_source == "multimodal":
        return max(24, min(80, body.min_token_count or 32))
    return 80


def _apply_train_hparams(eng: TrainEngine, body: StartIn) -> TrainEngine:
    eng.base_learning_rate = body.learning_rate
    eng.learning_rate = body.learning_rate
    eng.weight_decay = body.weight_decay
    eng.warmup_steps = body.warmup_steps
    eng.mixed_precision = body.mixed_precision
    eng.save_every_n_steps = body.save_every_n_steps
    eng.lr_schedule = body.lr_schedule
    eng.total_train_steps = body.total_train_steps
    if body.apply.tokenizer_id and body.tokenizer_id:
        eng.set_tokenizer(body.tokenizer_id)
    for group in eng.optimizer.param_groups:
        group["lr"] = body.learning_rate
        group["weight_decay"] = body.weight_decay
    if body.batch_size != eng.batch_size:
        eng.batch_size = body.batch_size
    if body.apply.micro_batch_size and body.micro_batch_size:
        eng.micro_batch_size = min(body.micro_batch_size, eng.batch_size, MAX_CUDA_MICRO_BATCH)
    else:
        eng.micro_batch_size = min(eng._resolve_micro_batch(eng.batch_size), MAX_CUDA_MICRO_BATCH)
    if body.apply.grad_accum_steps and body.grad_accum_steps:
        eng.grad_accum_steps = body.grad_accum_steps
    else:
        eng.grad_accum_steps = max(
            1, (eng.batch_size + eng.micro_batch_size - 1) // eng.micro_batch_size
        )
    return eng


def _prepare_engine(body: StartIn) -> TrainEngine:
    global _engine
    body = _merge_start_body(body)
    cfg = _model_config_from_body(body)
    if body.resume:
        eng = _get_engine()
        if eng.step > 0 and _structural_model_config(eng.model_config) != _structural_model_config(cfg):
            raise ValueError(_arch_mismatch_msg(eng.model_config, cfg))
        eng = _apply_train_hparams(eng, body)
        _sync_gradient_checkpointing(eng, cfg.gradient_checkpointing)
        return eng
    eng = _get_engine()
    if eng.step == 0:
        if eng.is_hf_backend:
            return _apply_train_hparams(eng, body)
        with _engine_lock:
            _engine = TrainEngine(
                device=DEVICE,
                batch_size=body.batch_size,
                learning_rate=body.learning_rate,
                model_config=cfg,
                weight_decay=body.weight_decay,
                warmup_steps=body.warmup_steps,
                mixed_precision=body.mixed_precision,
                micro_batch_size=body.micro_batch_size if body.apply.micro_batch_size else None,
                grad_accum_steps=body.grad_accum_steps if body.apply.grad_accum_steps else None,
                save_every_n_steps=body.save_every_n_steps,
                lr_schedule=body.lr_schedule,
                total_train_steps=body.total_train_steps,
                tokenizer_id=body.tokenizer_id,
            )
        eng = _apply_train_hparams(_engine, body)
        _sync_gradient_checkpointing(eng, cfg.gradient_checkpointing)
        return eng
    if _structural_model_config(eng.model_config) != _structural_model_config(cfg):
        raise ValueError(
            _arch_mismatch_msg(eng.model_config, cfg) + " — match layers/dim/block or load a fresh checkpoint"
        )
    eng = _apply_train_hparams(eng, body)
    _sync_gradient_checkpointing(eng, cfg.gradient_checkpointing)
    return eng


def _feed_steps(
    eng: TrainEngine,
    text: str,
    body: StartIn,
    label: str,
    row: dict[str, Any] | None = None,
    scalars: list[float] | None = None,
) -> None:
    text_bytes = len(text.encode("utf-8"))
    if scalars is None and row is not None:
        scalars = row_scalar_features(row)
        if scalars is None and body.data_source == "vjepa":
            scalars = vjepa_scalar_features(row)
    steps = eng.feed_text(text, scalars=scalars)
    S.bytes_written += text_bytes
    S.rows += 1
    for st in steps:
        now_mono = time.monotonic()
        if S.last_step_monotonic > 0:
            S.last_step_interval_s = now_mono - S.last_step_monotonic
        S.last_step_monotonic = now_mono
        S.train_step = st.step
        S.last_loss = st.loss
        S.last_tok_s = st.tok_s
        if st.step % 5 == 0:
            _touch_training_heartbeat()
        if S.data_source == "jsonl":
            pool_rows = S.jsonl_pool_rows or S.jsonl_total_rows or 1
            row_in_epoch = ((S.rows - 1) % pool_rows) + 1 if S.rows else 0
            epoch_note = f"ep{S.jsonl_epoch} · " if S.jsonl_epoch > 1 else ""
            progress = f"{epoch_note}{row_in_epoch:,}/{pool_rows:,} · {_fmt_consumed(S.bytes_written, body.target_gb)}"
        else:
            progress = _fmt_consumed(S.bytes_written, body.target_gb)
        S.add_log(
            f"[GREEN] step {st.step:,} loss {st.loss:.4f} · {progress} · {label} · {st.tok_s:.0f} tok/s{_speed_suffix()}",
            "green",
        )
        if should_save_model(st.step, eng.save_every_n_steps):
            _save_model(force=True)
    _flush_pending_save_eval(eng)


def _flush_pending_save_eval(eng: TrainEngine) -> None:
    pending = S.pending_save_eval
    if not pending:
        return
    name, step = pending
    if eng.step != step:
        return
    S.pending_save_eval = None
    threading.Thread(
        target=_run_save_eval,
        args=(name, step, True),
        daemon=True,
    ).start()


def _local_corpus_files(body: StartIn) -> list[str]:
    if body.mix_local_corpora:
        return list(LOCAL_CORPUS_MIX_FILES)
    return [body.jsonl_file]


def _local_corpus_label(body: StartIn) -> str:
    if body.mix_local_corpora:
        return LOCAL_CORPUS_MIX_LABEL
    return body.jsonl_file


def _count_local_corpus_rows(body: StartIn) -> int:
    return sum(_count_jsonl_rows(_data_path(fname)) for fname in _local_corpus_files(body))


def _validate_local_corpus(body: StartIn) -> str | None:
    for fname in _local_corpus_files(body):
        if not os.path.isfile(_data_path(fname)):
            return f"JSONL not found on server: {fname}"
    return None


def _ensure_distill_file(data_source: str) -> str | None:
    key = DISTILL_TOPIC_SOURCES.get(data_source)
    if not key:
        return None
    data_dir = _ensure_data_dir()
    fname = merged_distill_filename() if key == "merged" else distill_jsonl_filename(key)
    path = os.path.join(data_dir, fname)
    if os.path.isfile(path):
        return None
    return f"Distill corpus missing ({fname}) — run POST /api/distill/generate first"


def _ensure_sft_topic_file(data_source: str) -> str | None:
    topic = SFT_TOPIC_SOURCES.get(data_source)
    if not topic:
        return None
    data_dir = _ensure_data_dir()
    fname = topic_jsonl_filename(topic)
    path = os.path.join(data_dir, fname)
    if os.path.isfile(path):
        return None
    try:
        generate_topics_sft([topic], data_dir, hf_token=HF_TOKEN or None)
        S.add_log(f"[GREEN] generated {fname} ({topic} SFT)", "green")
    except Exception as exc:
        return f"SFT generate failed for {topic}: {exc}"
    if not os.path.isfile(path):
        return f"JSONL not found on server: {fname}"
    return None


def _wants_local_mix(body: StartIn) -> bool:
    if body.data_source in ("jsonl", "local-corpus-mix", "sft-intelligence", "distill-live", *SFT_TOPIC_SOURCES, *DISTILL_TOPIC_SOURCES):
        return False
    return bool(body.mix_local_jsonl)


def _normalize_training_body(body: StartIn) -> StartIn:
    """Apply data-source aliases and honor Stage-4 apply flags."""
    body = _merge_start_body(body)
    orig_source = body.data_source
    data = body.model_dump()
    if body.data_source == "local-corpus-mix":
        data["data_source"] = "jsonl"
        data["mix_local_corpora"] = True
    if body.data_source == "sft-intelligence":
        data["data_source"] = "jsonl"
        data["jsonl_file"] = "sft-full-merged.jsonl"
        data["mix_local_corpora"] = False
        data["mix_local_jsonl"] = False
        data["epoch_cycles"] = True
        data["epochs"] = min(max(int(body.epochs or 2), 1), 3)
        if not body.apply.learning_rate:
            data["learning_rate"] = 1e-5
        if not body.apply.target_gb:
            data["target_gb"] = 0.05
        if not body.apply.total_train_steps:
            data["total_train_steps"] = min(body.total_train_steps or 300, 500)
        if not body.apply.save_every_n_steps:
            data["save_every_n_steps"] = min(body.save_every_n_steps or 100, 200)
        data["hf_stream_fable"] = False
        data["hf_stream_sol"] = False
        data["hf_stream_kimi"] = False
        data["hf_stream_nemotron"] = False
        data["hf_stream_unsolved_math"] = False
        data["hf_stream_preference"] = False
        data["hf_stream_fineweb"] = False
    if body.data_source in DISTILL_TOPIC_SOURCES:
        key = DISTILL_TOPIC_SOURCES[body.data_source]
        fname = merged_distill_filename() if key == "merged" else distill_jsonl_filename(key)
        data["data_source"] = "jsonl"
        data["jsonl_file"] = fname
        data["mix_local_corpora"] = False
        data["mix_local_jsonl"] = False
        data["epoch_cycles"] = True
        data["epochs"] = min(max(int(body.epochs or 3), 1), 6)
        if not body.apply.learning_rate:
            data["learning_rate"] = 2e-5
        if not body.apply.target_gb:
            data["target_gb"] = 0.2
        if not body.apply.total_train_steps:
            data["total_train_steps"] = min(body.total_train_steps or 600, 1200)
        if not body.apply.save_every_n_steps:
            data["save_every_n_steps"] = min(body.save_every_n_steps or 50, 100)
        data["hf_stream_fable"] = False
        data["hf_stream_sol"] = False
        data["hf_stream_kimi"] = False
        data["hf_stream_nemotron"] = False
        data["hf_stream_unsolved_math"] = False
        data["hf_stream_preference"] = False
        data["hf_stream_fineweb"] = False
    if body.data_source == "distill-live":
        data["mix_local_corpora"] = False
        data["mix_local_jsonl"] = False
        data["epoch_cycles"] = False
        data["epochs"] = 1
        if not body.apply.learning_rate:
            data["learning_rate"] = 1e-5
        if not body.apply.total_train_steps:
            data["total_train_steps"] = 0
        if not body.apply.save_every_n_steps:
            data["save_every_n_steps"] = min(body.save_every_n_steps or 50, 200)
        data["hf_stream_fable"] = False
        data["hf_stream_sol"] = False
        data["hf_stream_kimi"] = False
        data["hf_stream_nemotron"] = False
        data["hf_stream_unsolved_math"] = False
        data["hf_stream_preference"] = False
        data["hf_stream_fineweb"] = False
    if body.data_source in SFT_TOPIC_SOURCES:
        topic = SFT_TOPIC_SOURCES[body.data_source]
        data["data_source"] = "jsonl"
        data["jsonl_file"] = topic_jsonl_filename(topic)
        data["mix_local_corpora"] = False
        data["mix_local_jsonl"] = False
        data["epoch_cycles"] = True
        data["epochs"] = min(max(int(body.epochs or 2), 1), 3)
        if not body.apply.learning_rate:
            data["learning_rate"] = 1e-5
        if not body.apply.target_gb:
            data["target_gb"] = 0.08
        if not body.apply.total_train_steps:
            data["total_train_steps"] = min(body.total_train_steps or 300, 500)
        if not body.apply.save_every_n_steps:
            data["save_every_n_steps"] = min(body.save_every_n_steps or 50, 200)
        data["hf_stream_fable"] = False
        data["hf_stream_sol"] = False
        data["hf_stream_kimi"] = False
        data["hf_stream_nemotron"] = False
        data["hf_stream_unsolved_math"] = False
        data["hf_stream_preference"] = False
        data["hf_stream_fineweb"] = False
    if body.apply.shuffle_hf_datasets:
        data["shuffle_hf_datasets"] = bool(body.shuffle_hf_datasets)
    if body.apply.mix_local_corpora and orig_source not in ("sft-intelligence", *SFT_TOPIC_SOURCES):
        data["mix_local_corpora"] = bool(body.mix_local_corpora)
    if body.apply.mix_local_jsonl and orig_source not in ("sft-intelligence", *SFT_TOPIC_SOURCES):
        data["mix_local_jsonl"] = bool(body.mix_local_jsonl)
    if body.apply.jsonl_mix_ratio:
        data["jsonl_mix_ratio"] = float(body.jsonl_mix_ratio)
    if body.apply.hf_streams and orig_source not in ("sft-intelligence", *SFT_TOPIC_SOURCES):
        data["hf_stream_fable"] = bool(body.hf_stream_fable)
        data["hf_stream_sol"] = bool(body.hf_stream_sol)
        data["hf_stream_kimi"] = bool(body.hf_stream_kimi)
        data["hf_stream_nemotron"] = bool(body.hf_stream_nemotron)
        data["hf_stream_unsolved_math"] = bool(body.hf_stream_unsolved_math)
        data["hf_stream_preference"] = bool(body.hf_stream_preference)
        data["hf_stream_fineweb"] = bool(body.hf_stream_fineweb)
    if body.data_source == "vjepa" and not body.apply.scalar_input_projection:
        data["scalar_input_projection"] = True
    if body.data_source == "multimodal" and not body.hf_dataset_id:
        data["hf_dataset_id"] = DEFAULT_MULTIMODAL_DATASET
    if body.data_source == "vjepa" and not body.hf_dataset_id:
        data["hf_dataset_id"] = DEFAULT_VJEPA_DATASET
    if not body.tokenizer_id:
        data["tokenizer_id"] = DEFAULT_TOKENIZER_ID
    return StartIn(**data)


def _init_local_mix_pool(body: StartIn) -> tuple[list[dict[str, Any]], list[int]]:
    if not _wants_local_mix(body):
        return [], [0]
    pool = _load_local_corpus_pool(body)
    random.shuffle(pool)
    return pool, [0]


def _take_local_mix_row(
    body: StartIn,
    pool: list[dict[str, Any]],
    local_idx: list[int],
) -> tuple[str, str, dict[str, Any]] | None:
    if not pool or random.random() >= body.jsonl_mix_ratio:
        return None
    row = pool[local_idx[0] % len(pool)]
    local_idx[0] += 1
    if local_idx[0] % len(pool) == 0:
        random.shuffle(pool)
    S.jsonl_hits += 1
    fname = str(row.get("_corpus_file") or body.jsonl_file)
    return row_training_text(row), f"local:{fname}", row


def _load_local_corpus_pool(body: StartIn) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    total_rows = 0
    for fname in _local_corpus_files(body):
        path = _data_path(fname)
        rows = load_jsonl_lines(path)
        total_rows += len(rows)
        added = 0
        for row in rows:
            text = row_training_text(row)
            if accept_training_text(text, min_chars=80):
                tagged = dict(row)
                tagged["_corpus_file"] = fname
                pool.append(tagged)
                added += 1
        skipped = len(rows) - added
        if skipped:
            S.add_log(f"[YELLOW] skipped {skipped} short/empty rows in {fname}", "yellow")
        S.add_log(f"[CYAN] loaded {added:,} rows from {fname}", "cyan")
    if not pool:
        label = _local_corpus_label(body)
        raise ValueError(f"No usable rows in local corpus ({label})")
    random.shuffle(pool)
    if body.mix_local_corpora:
        S.add_log(
            f"[CYAN] local corpus mix: {len(pool):,} rows shuffled across "
            + " + ".join(LOCAL_CORPUS_MIX_FILES),
            "cyan",
        )
    return pool


def _load_local_jsonl_pool(path: str) -> list[dict[str, Any]]:
    rows = load_jsonl_lines(path)
    pool: list[dict[str, Any]] = []
    for row in rows:
        text = row_training_text(row)
        if accept_training_text(text, min_chars=80):
            pool.append(row)
    if not pool:
        raise ValueError(f"No usable rows in {os.path.basename(path)}")
    skipped = len(rows) - len(pool)
    if skipped:
        S.add_log(f"[YELLOW] skipped {skipped} short/empty rows in {os.path.basename(path)}", "yellow")
    return pool


def _iter_shuffle_buffer(rows, buffer_size: int = HF_SHUFFLE_BUFFER):
    buf: list[Any] = []
    for row in rows:
        buf.append(row)
        if len(buf) >= buffer_size:
            random.shuffle(buf)
            yield from buf
            buf = []
    if buf:
        random.shuffle(buf)
        yield from buf


class _HfShufflePool:
    """Materialize finite HF streams for full shuffle; fall back to buffer shuffle if huge."""

    def __init__(self) -> None:
        self.rows: list[Any] | None = None
        self.use_buffer = False

    def epoch_rows(self, stream_factory, shuffle: bool):
        if not shuffle:
            yield from stream_factory()
            return
        if self.use_buffer:
            yield from _iter_shuffle_buffer(stream_factory(), HF_SHUFFLE_BUFFER)
            return
        if self.rows is None:
            self.rows = []
            tail = stream_factory()
            for row in tail:
                self.rows.append(row)
                if len(self.rows) >= HF_SHUFFLE_MATERIALIZE_MAX:
                    self.use_buffer = True
                    S.add_log(
                        f"[YELLOW] HF dataset >{HF_SHUFFLE_MATERIALIZE_MAX:,} rows — buffer shuffle only",
                        "yellow",
                    )
                    yield from _iter_shuffle_buffer(iter(self.rows), HF_SHUFFLE_BUFFER)
                    yield from _iter_shuffle_buffer(tail, HF_SHUFFLE_BUFFER)
                    return
            S.add_log(f"[CYAN] HF shuffle pool: {len(self.rows):,} rows", "cyan")
        random.shuffle(self.rows)
        yield from self.rows


def _fable_stream():
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files=FABLE_MERGED_URL,
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )


def _hh_rlhf_stream():
    from datasets import load_dataset

    for subset in ("helpful-base", "harmless-base"):
        stream = load_dataset(
            HH_RLHF_DATASET,
            data_dir=subset,
            split="train",
            streaming=True,
            token=HF_TOKEN or None,
        )
        for row in stream:
            yield row


def _sol_traces_stream():
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files=SOL_TRACES_URL,
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )


def _kimi_k3_traces_stream():
    from datasets import load_dataset

    return load_dataset(
        KIMI_K3_TRACES_DATASET_ID,
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )


def _unsolved_math_stream():
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files=UNSOLVED_MATH_URL,
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )


def _nemotron_math_stream():
    from datasets import load_dataset

    return load_dataset(
        NEMOTRON_MATH_DATASET_ID,
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )


def _nemotron_swe_stream():
    from datasets import load_dataset

    token = (HF_TOKEN or "").strip() or None
    attempts = [
        {
            "path": "json",
            "data_files": f"hf://datasets/{NEMOTRON_SWE_DATASET_ID}/data/*.jsonl.gz",
            "split": "train",
        },
        {
            "path": "json",
            "data_files": f"hf://datasets/{NEMOTRON_SWE_DATASET_ID}/data/train-00000-of-00010.jsonl.gz",
            "split": "train",
        },
    ]
    errors: list[str] = []
    for spec in attempts:
        try:
            return load_dataset(
                spec["path"],
                data_files=spec["data_files"],
                split=spec["split"],
                streaming=True,
                token=token,
            )
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("nemotron-swe stream failed: " + " | ".join(errors[:2]))


def _resolve_hf_dataset_id(body: StartIn) -> str:
    if body.data_source == "multimodal":
        ds = body.hf_dataset_id or DEFAULT_MULTIMODAL_DATASET
        if ds not in MULTIMODAL_DATASETS:
            raise ValueError(f"Unknown multimodal dataset '{ds}'")
        return MULTIMODAL_DATASETS[ds]
    if body.data_source == "vjepa":
        ids = _resolve_vjepa_dataset_ids(body)
        return ids[0]
    return body.hf_dataset_id or ""


def _resolve_vjepa_dataset_ids(body: StartIn) -> list[str]:
    raw = [x.strip() for x in (body.vjepa_dataset_ids or []) if x and x.strip()]
    if not raw:
        ds = body.hf_dataset_id or DEFAULT_VJEPA_DATASET
        raw = [ds]
    out: list[str] = []
    for ds in raw:
        resolved = VJEPA_DATASETS.get(ds, ds)
        if resolved not in VJEPA_DATASETS.values() and ds not in VJEPA_DATASETS:
            raise ValueError(f"Unknown V-JEPA dataset '{ds}'")
        if resolved not in out:
            out.append(resolved)
    return out


def _hf_dataset_stream(dataset_id: str, *, kind: str = "multimodal"):
    return open_hf_train_stream(dataset_id, kind=kind, token=HF_TOKEN or None)


def _hf_vjepa_stream(body: StartIn):
    ids = _resolve_vjepa_dataset_ids(body)
    use_local = body.vjepa_local_cache or os.environ.get("VJEPA_LOCAL_CACHE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if use_local:
        from vjepa_local_cache import cache_ready, cached_datasets, open_hybrid_vjepa_interleaved, open_local_vjepa_interleaved

        if cache_ready(ids):
            return open_local_vjepa_interleaved(ids)
        partial = cached_datasets(ids)
        if partial:
            return open_hybrid_vjepa_interleaved(ids, token=HF_TOKEN or None)
        S.add_log(
            "[YELLOW] vjepa_local_cache enabled but no cache files — falling back to HF stream",
            "yellow",
        )
    if len(ids) == 1:
        return _hf_dataset_stream(ids[0], kind="vjepa")
    return open_hf_train_stream_interleaved(ids, kind="vjepa", token=HF_TOKEN or None)


def _vjepa_row_text(row: dict[str, Any], *, min_chars: int = 20) -> str:
    if isinstance(row, dict) and row.get("text"):
        return str(row["text"])
    return vjepa_training_text(row, min_chars=min_chars)


def _vjepa_packed_stream_factory(body: StartIn):
    pack_rows = int(body.vjepa_pack_rows or 0)
    if pack_rows <= 1:
        return lambda: _hf_vjepa_stream(body)

    from vjepa_pack import pack_vjepa_rows

    min_chars = int(body.vjepa_pack_min_chars or 4000)

    def _gen():
        for text, meta in pack_vjepa_rows(
            _hf_vjepa_stream(body),
            pack_rows=pack_rows,
            min_chars=min_chars,
        ):
            row: dict[str, Any] = dict(meta or {})
            row["text"] = text
            row["packed"] = True
            yield row

    return _gen


class _HfStreamSlot:
    """One weighted HF source in mixed training."""

    def __init__(
        self,
        name: str,
        factory,
        text_fn,
        label: str,
        on_hit,
        *,
        filter_row=None,
    ) -> None:
        self.name = name
        self.factory = factory
        self.text_fn = text_fn
        self.label = label
        self.on_hit = on_hit
        self.filter_row = filter_row
        self.shuffle_pool = _HfShufflePool()
        self._row_iter = None

    def reset_epoch(self) -> None:
        self._row_iter = None

    def next_row(self, body: StartIn) -> tuple[str, str, dict[str, Any] | None]:
        while True:
            if self._row_iter is None:
                self._row_iter = iter(
                    self.shuffle_pool.epoch_rows(self.factory, body.shuffle_hf_datasets)
                )
            try:
                row = next(self._row_iter)
            except StopIteration:
                self._row_iter = None
                continue
            if self.filter_row and not self.filter_row(row):
                S.skipped += 1
                continue
            text = self.text_fn(row)
            if not accept_training_text(text, min_chars=80):
                S.skipped += 1
                continue
            self.on_hit()
            return text, self.label, row if isinstance(row, dict) else None


def _fineweb_stream_factory(config: str):
    def _factory():
        from datasets import load_dataset

        return load_dataset(
            DATASET_ID,
            name=config,
            split="train",
            streaming=True,
            token=HF_TOKEN,
        )

    return _factory


def _build_hf_stream_slots(body: StartIn) -> list[_HfStreamSlot]:
    slots: list[_HfStreamSlot] = []
    if body.hf_stream_fable:
        slots.append(
            _HfStreamSlot(
                "fable",
                _fable_stream,
                row_training_text,
                "fable",
                lambda: setattr(S, "fable_hits", S.fable_hits + 1),
            )
        )
    if body.hf_stream_sol:
        slots.append(
            _HfStreamSlot(
                "sol",
                _sol_traces_stream,
                row_training_text,
                "sol",
                lambda: setattr(S, "sol_hits", S.sol_hits + 1),
            )
        )
    if body.hf_stream_kimi:
        slots.append(
            _HfStreamSlot(
                "kimi",
                _kimi_k3_traces_stream,
                row_training_text,
                "kimi",
                lambda: setattr(S, "kimi_hits", S.kimi_hits + 1),
            )
        )
    if body.hf_stream_nemotron:
        slots.append(
            _HfStreamSlot(
                "nemotron",
                _nemotron_math_stream,
                nemotron_math_training_text,
                "nemotron",
                lambda: setattr(S, "nemotron_hits", S.nemotron_hits + 1),
            )
        )
    if body.hf_stream_unsolved_math:
        slots.append(
            _HfStreamSlot(
                "math",
                _unsolved_math_stream,
                unsolved_math_training_text,
                "math",
                lambda: setattr(S, "math_hits", S.math_hits + 1),
            )
        )
    if body.hf_stream_preference:
        slots.append(
            _HfStreamSlot(
                "pref",
                _hh_rlhf_stream,
                preference_training_text,
                "hh-rlhf",
                lambda: setattr(S, "preference_hits", S.preference_hits + 1),
            )
        )
    if body.hf_stream_fineweb:
        slots.append(
            _HfStreamSlot(
                "fineweb",
                _fineweb_stream_factory(body.config),
                lambda row: str(row.get("text") or "").strip(),
                "fineweb",
                lambda: None,
                filter_row=lambda row: _accept_row(row, body.english_only, body.min_token_count),
            )
        )
    return slots


def _stream_mixed_train(
    eng: TrainEngine,
    body: StartIn,
    target_bytes: int,
    skip_to: int,
    local_pool: list[dict[str, Any]],
    local_idx: list[int],
) -> None:
    slots = _build_hf_stream_slots(body)
    if not slots:
        raise ValueError("Enable at least one HF stream (Stage 4 → HF stream mix)")
    names = ", ".join(s.name for s in slots)
    S.add_log(f"[CYAN] HF stream mix: {names} · equal weight", "cyan")
    idx = 0
    mix_epoch = 0
    while S.running and not _training_target_reached(eng, target_bytes):
        mix_epoch += 1
        if body.epoch_cycles and mix_epoch > body.epochs:
            break
        if mix_epoch > 1:
            S.add_log(
                f"[CYAN] mixed epoch {mix_epoch} — {_fmt_consumed(S.bytes_written, body.target_gb)} "
                f"toward {_fmt_target(body.target_gb)}",
                "cyan",
            )
        for slot in slots:
            slot.reset_epoch()
        while S.running and not _training_target_reached(eng, target_bytes):
            while S.paused and S.running:
                time.sleep(0.2)
            if not S.running or _training_target_reached(eng, target_bytes):
                break
            idx += 1
            S.stream_index = idx
            if idx <= skip_to:
                continue
            mixed = _take_local_mix_row(body, local_pool, local_idx)
            if mixed:
                text, label, active_row = mixed
            else:
                slot = random.choice(slots)
                text, label, active_row = slot.next_row(body)
            text_bytes = len(text.encode("utf-8"))
            if S.bytes_written + text_bytes > target_bytes:
                break
            _feed_steps(eng, text, body, label, row=active_row)
        skip_to = 0
        if _training_target_reached(eng, target_bytes):
            break


def _get_reward_scorer():
    global _reward_model, _reward_tokenizer
    with _reward_lock:
        if _reward_model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _reward_tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_ID, token=HF_TOKEN or None)
            _reward_model = AutoModelForSequenceClassification.from_pretrained(
                RM_MODEL_ID,
                token=HF_TOKEN or None,
            )
            _reward_model.to("cpu")
            _reward_model.eval()
    return _reward_model, _reward_tokenizer


def _stream_finite_rows(
    eng: TrainEngine,
    body: StartIn,
    target_bytes: int,
    skip_to: int,
    stream_factory,
    text_fn,
    label: str,
    epoch_name: str,
    on_row=None,
    local_pool: list[dict[str, Any]] | None = None,
    local_idx: list[int] | None = None,
    hf_shuffle_pool: _HfShufflePool | None = None,
) -> None:
    idx = 0
    epoch = 0
    min_chars = _min_accept_chars(body)
    mix_pool = local_pool or []
    mix_idx = local_idx if local_idx is not None else [0]
    shuffle_pool = hf_shuffle_pool or _HfShufflePool()
    while S.running and not _training_target_reached(eng, target_bytes):
        epoch += 1
        steps_before_epoch = eng.step
        bytes_before_epoch = S.bytes_written
        epoch_rows = 0
        epoch_start_mono = time.monotonic()
        if epoch > 1:
            S.add_log(f"[CYAN] {epoch_name} epoch {epoch} — continuing toward {_fmt_target(body.target_gb)}", "cyan")
        for row in shuffle_pool.epoch_rows(stream_factory, body.shuffle_hf_datasets):
            while S.paused and S.running:
                time.sleep(0.2)
            if not S.running or _training_target_reached(eng, target_bytes):
                break
            idx += 1
            epoch_rows += 1
            S.stream_index = idx
            if idx <= skip_to:
                continue
            row_label = label
            active_row: dict[str, Any] | None = row if isinstance(row, dict) else None
            mixed = _take_local_mix_row(body, mix_pool, mix_idx)
            if mixed:
                text, row_label, active_row = mixed
            else:
                if on_row:
                    row, row_label = on_row(row, label)
                    active_row = row if isinstance(row, dict) else active_row
                text = text_fn(row, min_chars=min_chars) if body.data_source in ("multimodal", "vjepa") else text_fn(row)
            if not accept_training_text(text, min_chars=min_chars):
                S.skipped += 1
                if epoch_rows >= 2500 and eng.step == steps_before_epoch and S.skipped > epoch_rows - 5:
                    raise RuntimeError(
                        f"{epoch_name}: no trainable text after {epoch_rows} rows "
                        f"({S.skipped} skipped) — HF fields/config may be wrong for this dataset"
                    )
                continue
            text_bytes = len(text.encode("utf-8"))
            if S.bytes_written + text_bytes > target_bytes:
                break
            _feed_steps(eng, text, body, row_label, row=active_row)
        epoch_dur = time.monotonic() - epoch_start_mono
        steps_delta = eng.step - steps_before_epoch
        mb_delta = (S.bytes_written - bytes_before_epoch) / (1024**2)
        S.last_epoch_duration_s = epoch_dur
        S.stream_epoch = epoch
        if epoch_rows > 0 and (steps_delta > 0 or mb_delta > 0):
            epoch_mb_hr = (mb_delta / (epoch_dur / 3600.0)) if epoch_dur >= 1.0 and mb_delta > 0 else 0.0
            S.add_log(
                f"[CYAN] {epoch_name} epoch {epoch} done · {_fmt_duration(epoch_dur)}"
                f" · +{steps_delta} steps · +{mb_delta:.2f} MB"
                + (f" · {epoch_mb_hr:.1f} MB/hr (epoch)" if epoch_mb_hr > 0 else ""),
                "cyan",
            )
        if _training_target_reached(eng, target_bytes):
            break
        skip_to = 0


def _compute_stop_reason(eng: TrainEngine, body: StartIn, target_bytes: int) -> tuple[str, str]:
    byte_met = S.bytes_written >= int(target_bytes * 0.999)
    step_cap = int(eng.total_train_steps or 0)
    steps_done = max(0, eng.step - eng.run_start_step) if eng.run_start_step else eng.step
    step_met = step_cap > 0 and steps_done >= step_cap
    pool = S.jsonl_pool_rows or 0
    if byte_met:
        return "byte_target", "Byte target reached"
    if body.data_source == "jsonl" and body.epoch_cycles and body.epochs and S.jsonl_epoch >= body.epochs:
        mb = S.bytes_written / (1024**2)
        return (
            "epoch_cycles_complete",
            f"Epoch cycles ×{body.epochs} complete · {mb:.2f} MB consumed"
            + (f" (target {_fmt_target(body.target_gb)} not reached)" if not byte_met else ""),
        )
    if body.data_source == "jsonl" and pool > 0 and not byte_met and not body.epoch_cycles:
        mb = S.bytes_written / (1024**2)
        return (
            "data_exhausted",
            f"Local JSONL exhausted ({pool} rows · {mb:.2f} MB) — file smaller than byte target",
        )
    if step_met:
        return "step_cap", f"Step cap reached ({steps_done}/{step_cap} steps this run)"
    if S.bytes_written > 0 and not S.running:
        return "stopped", "Training stopped"
    return "complete", "Run complete"


def _finish_training(eng: TrainEngine, body: StartIn, target_bytes: int) -> None:
    done = _training_target_reached(eng, target_bytes)
    if S.data_source == "jsonl" and body.epoch_cycles and S.jsonl_total_rows > 0:
        done = done or S.rows >= S.jsonl_total_rows
        if body.epochs and S.jsonl_epoch >= body.epochs:
            done = True
    if done:
        reason, label = _compute_stop_reason(eng, body, target_bytes)
        S.stop_reason = reason
        S.stop_reason_label = label
        S.status = "complete"
        _save_model(force=True)
        _save_checkpoint(force=True)
        _auto_record_run("complete")
        if S.data_source == "jsonl":
            epoch_note = f" · {S.jsonl_epoch} epoch(s)" if S.jsonl_epoch else ""
            S.add_log(
                f"[GREEN] complete — step {eng.step:,} · {_fmt_consumed(S.bytes_written, body.target_gb)}"
                f"{epoch_note} · model in COS",
                "green",
            )
        elif S.data_source == "mixed":
            S.add_log(
                f"[GREEN] complete — step {eng.step:,} · {_fmt_consumed(S.bytes_written, body.target_gb)} · "
                f"local {S.jsonl_hits:,} / fable {S.fable_hits:,} · model in COS",
                "green",
            )
        else:
            S.add_log(
                f"[GREEN] complete — step {eng.step:,} · {_fmt_consumed(S.bytes_written, body.target_gb)} · model in COS",
                "green",
            )
        goal = S.run_goal_snapshot(eng, target_bytes)
        color = "green" if goal.get("goal_met") else "yellow"
        S.add_log(f"[{color.upper()}] STOP · {S.stop_reason_label}", color)
        S.add_log(f"[{color.upper()}] GOAL outcome · {goal.get('outcome_label')}", color)
    elif S.running:
        S.status = "stopped"
        _attach_session_meta(_get_engine())
        _save_checkpoint(force=True)
        _auto_record_run("stopped")
        if S.data_source == "jsonl" and not _training_target_reached(eng, target_bytes):
            S.add_log(
                f"[YELLOW] paused at {_fmt_consumed(S.bytes_written, body.target_gb)}"
                f" · ep {S.jsonl_epoch} · resume to continue toward {_fmt_target(body.target_gb)}",
                "yellow",
            )
        else:
            S.add_log(
                f"[YELLOW] stopped at {_fmt_consumed(S.bytes_written, body.target_gb)} — no mid-run COS save",
                "yellow",
            )
    else:
        S.status = "interrupted"
        _attach_session_meta(_get_engine())
        _save_checkpoint(force=True)
        _auto_record_run("interrupted")
        S.add_log("[YELLOW] interrupted — no mid-run COS save", "yellow")


def _run_distill_live(eng: TrainEngine, body: StartIn, target_bytes: int) -> None:
    """Generate teacher responses on-the-fly and train student until byte target."""
    benchmark_limit = int(os.environ.get("DISTILL_LIVE_BENCHMARK_LIMIT", "0"))
    data_dir = _ensure_data_dir()
    teacher = TopicTeacherSession(hf_token=HF_TOKEN or None, prefer_device="cuda")
    S.jsonl_file = LIVE_DISTILL_LOG
    S.add_log(
        f"[CYAN] distill-live · topics {list(DISTILL_TOPICS)} · benchmark_limit={benchmark_limit or 'all'} · "
        f"append {LIVE_DISTILL_LOG} · stop {_fmt_target(body.target_gb)}",
        "cyan",
    )
    generated = 0
    try:
        for epoch, topic, item in iter_distill_live_prompts(list(DISTILL_TOPICS), benchmark_limit=benchmark_limit):
            while S.paused and S.running:
                time.sleep(0.2)
            if not S.running or _training_target_reached(eng, target_bytes):
                break
            user = str(item.get("user") or "").strip()
            if not user:
                continue
            teacher.ensure_topic(topic)
            if epoch != S.jsonl_epoch:
                S.jsonl_epoch = epoch
                S.add_log(
                    f"[CYAN] distill-live epoch {epoch} · {_fmt_consumed(S.bytes_written, body.target_gb)}",
                    "cyan",
                )
            try:
                answer = teacher.generate(user)
            except Exception as exc:
                S.add_log(f"[YELLOW] teacher {topic} failed: {exc}", "yellow")
                continue
            if not accept_training_text(answer, min_chars=16):
                S.skipped += 1
                continue
            row = make_distill_row(
                topic,
                user=user,
                assistant=answer,
                teacher=teacher.label,
                source=str(item.get("source") or "distill-live"),
                prompt_id=item.get("id"),
            )
            append_live_distill_row(data_dir, row)
            text = row_training_text(row)
            if not accept_training_text(text, min_chars=80):
                S.skipped += 1
                continue
            _feed_steps(eng, text, body, f"distill-live:{topic}", row=row)
            S.jsonl_hits += 1
            generated += 1
            S.stream_index = generated
    finally:
        teacher.close()
    S.add_log(f"[GREEN] distill-live done · generated {generated} rows · log {LIVE_DISTILL_LOG}", "green")


def _stream_and_train(body: StartIn) -> None:
    body = _normalize_training_body(body)
    if body.mix_local_corpora:
        S.add_log(
            f"[CYAN] local corpus mix: {' + '.join(_local_corpus_files(body))}",
            "cyan",
        )
    if body.shuffle_hf_datasets and body.data_source not in ("jsonl",):
        S.add_log("[CYAN] HF dataset shuffle enabled", "cyan")

    if body.data_source not in DATA_SOURCES:
        S.error = f"Unknown data_source '{body.data_source}'"
        S.status = "error"
        S.add_log(f"[RED] {S.error}", "red")
        return

    needs_hf = body.data_source in (
        "fineweb",
        "fable-traces",
        "mixed",
        "openassistant-rm",
        "sol-traces",
        "kimi-k3-traces",
        "unsolved-math",
        "nemotron-math",
        "nemotron-swe",
        "multimodal",
        "vjepa",
    )
    if body.data_source == "distill-live" and not HF_TOKEN:
        S.error = "HF_TOKEN missing (required for live teacher distill)"
        S.status = "error"
        S.add_log("[RED] HF_TOKEN not mounted — distill-live needs teacher models", "red")
        return
    if needs_hf and not HF_TOKEN:
        S.error = "HF_TOKEN missing"
        S.status = "error"
        S.add_log("[RED] HF_TOKEN not mounted", "red")
        return
    needs_jsonl = body.data_source in ("jsonl", "local-corpus-mix") or _wants_local_mix(body)
    if needs_jsonl:
        missing = _validate_local_corpus(body)
        if missing:
            S.error = missing
            S.status = "error"
            S.add_log(f"[RED] {S.error} — scp to /opt/app2-nextaura-us/data/", "red")
            return

    cos = _cos_client()
    if not cos:
        S.error = "IBM_CLOUD_API_KEY missing"
        S.status = "error"
        S.add_log("[RED] IBM_CLOUD_API_KEY not mounted", "red")
        return

    try:
        eng = _prepare_engine(body)
    except Exception as exc:
        S.error = str(exc)
        S.status = "error"
        S.add_log(f"[RED] engine setup failed: {exc}", "red")
        return

    target_bytes = int(body.target_gb * (1024**3))
    ckpt = _load_checkpoint() if body.resume else None
    resume_ok = (
        ckpt
        and ckpt.get("status") in ("interrupted", "stopped", "streaming")
        and ckpt.get("data_source") == body.data_source
        and float(ckpt.get("target_gb") or 0) == body.target_gb
        and (
            body.data_source != "jsonl"
            or (
                str(ckpt.get("jsonl_file") or "") == body.jsonl_file
                and bool(ckpt.get("mix_local_corpora")) == bool(body.mix_local_corpora)
            )
        )
        and int(ckpt.get("rows") or 0) > 0
        and int(ckpt.get("bytes_written") or 0) < target_bytes
    )

    S.data_source = body.data_source
    S.hf_dataset_id = body.hf_dataset_id
    S.tokenizer_id = body.tokenizer_id
    S.jsonl_file = body.jsonl_file
    S.epochs = body.epochs if body.epoch_cycles else 1
    S.epoch_cycles = body.epoch_cycles
    S.jsonl_mix_ratio = body.jsonl_mix_ratio
    S.mix_local_jsonl = _wants_local_mix(body)
    S.mix_local_corpora = body.mix_local_corpora
    S.shuffle_hf_datasets = body.shuffle_hf_datasets
    S.hf_stream_fable = body.hf_stream_fable
    S.hf_stream_sol = body.hf_stream_sol
    S.hf_stream_kimi = body.hf_stream_kimi
    S.hf_stream_nemotron = body.hf_stream_nemotron
    S.hf_stream_unsolved_math = body.hf_stream_unsolved_math
    S.hf_stream_preference = body.hf_stream_preference
    S.hf_stream_fineweb = body.hf_stream_fineweb
    S.config = body.config
    S.target_gb = body.target_gb
    S.model_params = body.model_params
    S.learning_rate = eng.learning_rate
    S.batch_size = eng.batch_size
    S.running = True
    _touch_training_heartbeat()
    S.paused = False
    S.status = "streaming"
    S.error = ""
    if body.run_goal_title or body.run_goal_detail or body.run_goal_agent:
        S.run_goal_title = body.run_goal_title.strip()
        S.run_goal_detail = body.run_goal_detail.strip()
        S.run_stop_primary = body.run_stop_primary or "bytes"
        S.run_goal_agent = (body.run_goal_agent or "operator").strip()
        S.run_goal_started_at = datetime.now(timezone.utc).isoformat()
        detail_preview = S.run_goal_detail[:140] + ("…" if len(S.run_goal_detail) > 140 else "")
        S.add_log(
            f"[CYAN] GOAL · {S.run_goal_title or 'training run'} · stop={S.run_stop_primary}"
            f" · agent={S.run_goal_agent}"
            + (f" · {detail_preview}" if detail_preview else ""),
            "cyan",
        )

    skip_to = 0
    if resume_ok and ckpt:
        S.apply_checkpoint(ckpt)
        _load_model_from_cos()
        skip_to = S.stream_index
        if S.train_wall_started_at <= 0:
            S.train_wall_started_at = time.time()
        S.add_log(
            f"[CYAN] resume {body.data_source} row={skip_to:,} step={eng.step}",
            "cyan",
        )
    else:
        eng.run_start_step = eng.step
        S.rows = 0
        S.skipped = 0
        S.bytes_written = 0
        S.stream_index = 0
        S.jsonl_hits = 0
        S.fable_hits = 0
        S.preference_hits = 0
        S.sol_hits = 0
        S.kimi_hits = 0
        S.math_hits = 0
        S.nemotron_hits = 0
        S.swe_hits = 0
        S.multimodal_hits = 0
        S.vjepa_hits = 0
        S.jsonl_epoch = 0
        S.jsonl_pool_rows = 0
        S.train_wall_started_at = time.time()
        S.last_step_monotonic = 0.0
        S.last_step_interval_s = 0.0
        S.last_epoch_duration_s = 0.0
        S.stream_epoch = 0
        skip_to = int(body.stream_skip_rows or 0)
        if skip_to > 0:
            S.add_log(f"[CYAN] stream skip first {skip_to:,} rows", "cyan")
        if body.data_source == "jsonl":
            corpus_label = _local_corpus_label(body)
            S.add_log(
                f"[CYAN] jsonl train · {corpus_label} · target {_fmt_target(body.target_gb)} · "
                f"step={eng.step:,} · lr={eng.learning_rate:g} batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "mixed":
            corpus_label = _local_corpus_label(body)
            local_pct = int(body.jsonl_mix_ratio * 100) if _wants_local_mix(body) else 0
            hf_names = [n for n, on in (
                ("fable", body.hf_stream_fable),
                ("sol", body.hf_stream_sol),
                ("kimi", body.hf_stream_kimi),
                ("nemotron", body.hf_stream_nemotron),
                ("math", body.hf_stream_unsolved_math),
                ("pref", body.hf_stream_preference),
                ("fineweb", body.hf_stream_fineweb),
            ) if on]
            S.jsonl_total_rows = _count_local_corpus_rows(body)
            S.add_log(
                f"[CYAN] mixed train · local {local_pct}% · {corpus_label} + HF [{', '.join(hf_names) or 'none'}] · "
                f"{_fmt_target(body.target_gb)} · step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "fable-traces":
            S.add_log(
                f"[CYAN] fable stream · {FABLE_DATASET_ID} · {body.target_gb} GB · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "openassistant-rm":
            S.add_log(
                f"[CYAN] preference stream · {HH_RLHF_DATASET} · RM {RM_MODEL_ID} · "
                f"{_fmt_target(body.target_gb)} · {eng.model_config.n_layer}L×{eng.model_config.n_embd}d · "
                f"batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "sol-traces":
            S.add_log(
                f"[CYAN] sol traces · {SOL_TRACES_DATASET_ID} · {_fmt_target(body.target_gb)} · "
                f"step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "kimi-k3-traces":
            mix_note = f" + local {int(body.jsonl_mix_ratio * 100)}%" if _wants_local_mix(body) else ""
            S.add_log(
                f"[CYAN] kimi k3 traces · {KIMI_K3_TRACES_DATASET_ID}{mix_note} · "
                f"{_fmt_target(body.target_gb)} · step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "unsolved-math":
            S.add_log(
                f"[CYAN] unsolved math · {UNSOLVED_MATH_DATASET_ID} · {_fmt_target(body.target_gb)} · "
                f"step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "nemotron-math":
            mix_note = f" + local {int(body.jsonl_mix_ratio * 100)}%" if _wants_local_mix(body) else ""
            S.add_log(
                f"[CYAN] nemotron math · {NEMOTRON_MATH_DATASET_ID}{mix_note} · "
                f"{_fmt_target(body.target_gb)} · step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "nemotron-swe":
            mix_note = f" + local {int(body.jsonl_mix_ratio * 100)}%" if _wants_local_mix(body) else ""
            S.add_log(
                f"[CYAN] nemotron SWE v3.5 · {NEMOTRON_SWE_DATASET_ID}{mix_note} · "
                f"{_fmt_target(body.target_gb)} · step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "multimodal":
            ds_id = _resolve_hf_dataset_id(body)
            S.add_log(
                f"[CYAN] multimodal · {ds_id} · tokenizer {body.tokenizer_id} · "
                f"{_fmt_target(body.target_gb)} · step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        elif body.data_source == "vjepa":
            ds_id = _resolve_hf_dataset_id(body)
            S.add_log(
                f"[CYAN] v-jepa · {ds_id} · scalar proj on · tokenizer {body.tokenizer_id} · "
                f"{_fmt_target(body.target_gb)} · step={eng.step:,} · batch={eng.batch_label()}",
                "cyan",
            )
        else:
            S.add_log(
                f"[CYAN] fresh pretrain · {DATASET_ID}/{body.config} · {body.target_gb} GB · "
                f"{eng.param_count:,} params · {DEVICE} lr={eng.learning_rate:g} batch={eng.batch_label()}",
                "cyan",
            )

    try:
        try:
            cos.head_bucket(Bucket=COS_BUCKET)
        except Exception:
            cos.create_bucket(Bucket=COS_BUCKET)

        local_pool, local_idx = _init_local_mix_pool(body)
        fable_shuffle = _HfShufflePool()

        if body.data_source == "fineweb":
            from datasets import load_dataset

            def _fineweb_stream():
                return load_dataset(
                    DATASET_ID,
                    name=body.config,
                    split="train",
                    streaming=True,
                    token=HF_TOKEN,
                )

            idx = 0
            row_iter = (
                _iter_shuffle_buffer(_fineweb_stream(), HF_SHUFFLE_BUFFER)
                if body.shuffle_hf_datasets
                else _fineweb_stream()
            )
            for row in row_iter:
                while S.paused and S.running:
                    time.sleep(0.2)
                if not S.running or _training_target_reached(eng, target_bytes):
                    break
                idx += 1
                S.stream_index = idx
                if idx <= skip_to:
                    continue
                mixed = _take_local_mix_row(body, local_pool, local_idx)
                active_row: dict[str, Any] | None = None
                if mixed:
                    text, label, active_row = mixed
                else:
                    if not _accept_row(row, body.english_only, body.min_token_count):
                        S.skipped += 1
                        continue
                    text = str(row.get("text") or "").strip()
                    label = "fineweb"
                if S.bytes_written + len(text.encode("utf-8")) > target_bytes:
                    break
                _feed_steps(eng, text, body, label, row=active_row)

        elif body.data_source == "distill-live":
            _run_distill_live(eng, body, target_bytes)

        elif body.data_source == "jsonl":
            pool = _load_local_corpus_pool(body)
            pool_rows = len(pool)
            S.jsonl_pool_rows = pool_rows
            if body.epoch_cycles:
                S.jsonl_total_rows = pool_rows * body.epochs
            else:
                S.jsonl_total_rows = pool_rows
            if not resume_ok:
                S.add_log(
                    f"[CYAN] jsonl pool: {pool_rows:,} usable rows · "
                    f"will cycle until {_fmt_target(body.target_gb)}",
                    "cyan",
                )
            idx = 0
            epoch = S.jsonl_epoch or 0
            max_epochs = body.epochs if body.epoch_cycles else None
            while S.running and not _training_target_reached(eng, target_bytes):
                epoch += 1
                S.jsonl_epoch = epoch
                epoch_start_mono = time.monotonic()
                bytes_before_epoch = S.bytes_written
                steps_before_epoch = eng.step
                if max_epochs is not None and epoch > max_epochs:
                    break
                if epoch > 1:
                    S.add_log(
                        f"[CYAN] jsonl epoch {epoch} — {_fmt_consumed(S.bytes_written, body.target_gb)} "
                        f"toward {_fmt_target(body.target_gb)}",
                        "cyan",
                    )
                random.shuffle(pool)
                for row in pool:
                    while S.paused and S.running:
                        time.sleep(0.2)
                    if not S.running or _training_target_reached(eng, target_bytes):
                        break
                    idx += 1
                    S.stream_index = idx
                    if idx <= skip_to:
                        continue
                    text = row_training_text(row)
                    fname = str(row.get("_corpus_file") or body.jsonl_file)
                    _feed_steps(eng, text, body, f"jsonl:{fname}", row=row)
                    S.jsonl_hits += 1
                epoch_dur = time.monotonic() - epoch_start_mono
                steps_delta = eng.step - steps_before_epoch
                mb_delta = (S.bytes_written - bytes_before_epoch) / (1024**2)
                S.last_epoch_duration_s = epoch_dur
                S.stream_epoch = epoch
                if steps_delta > 0 or mb_delta > 0:
                    epoch_mb_hr = (mb_delta / (epoch_dur / 3600.0)) if epoch_dur >= 1.0 and mb_delta > 0 else 0.0
                    S.add_log(
                        f"[CYAN] jsonl epoch {epoch} done · {_fmt_duration(epoch_dur)}"
                        f" · +{steps_delta} steps · +{mb_delta:.2f} MB"
                        + (f" · {epoch_mb_hr:.1f} MB/hr (epoch)" if epoch_mb_hr > 0 else ""),
                        "cyan",
                    )
                skip_to = 0
                if _training_target_reached(eng, target_bytes):
                    break
                if max_epochs is not None and epoch >= max_epochs:
                    break

        elif body.data_source == "openassistant-rm":

            def _track_pref(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.preference_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _hh_rlhf_stream,
                preference_training_text,
                "hh-rlhf",
                "preference stream",
                on_row=_track_pref,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "sol-traces":

            def _track_sol(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.sol_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _sol_traces_stream,
                row_training_text,
                "sol",
                "sol trace stream",
                on_row=_track_sol,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "kimi-k3-traces":

            def _track_kimi(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.kimi_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _kimi_k3_traces_stream,
                row_training_text,
                "kimi",
                "kimi trace stream",
                on_row=_track_kimi,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "unsolved-math":

            def _track_math(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.math_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _unsolved_math_stream,
                unsolved_math_training_text,
                "math",
                "unsolved math stream",
                on_row=_track_math,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "nemotron-math":

            def _track_nemotron(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.nemotron_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _nemotron_math_stream,
                nemotron_math_training_text,
                "nemotron",
                "nemotron math stream",
                on_row=_track_nemotron,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "nemotron-swe":

            def _track_swe(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.swe_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _nemotron_swe_stream,
                nemotron_swe_training_text,
                "swe",
                "nemotron SWE v3.5 stream",
                on_row=_track_swe,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "fable-traces":

            def _track_fable(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.fable_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                _fable_stream,
                row_training_text,
                "fable",
                "fable stream",
                on_row=_track_fable,
                local_pool=local_pool,
                local_idx=local_idx,
                hf_shuffle_pool=fable_shuffle,
            )

        elif body.data_source == "mixed":
            _stream_mixed_train(
                eng,
                body,
                target_bytes,
                skip_to,
                local_pool,
                local_idx,
            )

        elif body.data_source == "multimodal":
            ds_id = _resolve_hf_dataset_id(body)
            try:
                probe = probe_hf_train_stream(ds_id, kind="multimodal", token=HF_TOKEN or None, max_rows=150)
                if probe.get("ok"):
                    S.add_log(
                        f"[CYAN] HF probe OK · {ds_id} · keys {probe.get('keys', [])[:6]} · "
                        f"{str(probe.get('text_preview', ''))[:72]}…",
                        "cyan",
                    )
                else:
                    S.add_log(f"[YELLOW] HF probe: {probe.get('error')} — will try stream anyway", "yellow")
            except Exception as exc:
                S.add_log(f"[YELLOW] HF probe failed: {exc}", "yellow")

            def _track_mm(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.multimodal_hits += 1
                return row, label

            _stream_finite_rows(
                eng,
                body,
                target_bytes,
                skip_to,
                lambda: _hf_dataset_stream(ds_id, kind="multimodal"),
                multimodal_training_text,
                "multimodal",
                f"multimodal:{ds_id}",
                on_row=_track_mm,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        elif body.data_source == "vjepa":
            vjepa_ids = _resolve_vjepa_dataset_ids(body)
            ds_label = "+".join(vjepa_ids[:3]) + (f"+{len(vjepa_ids)-3}" if len(vjepa_ids) > 3 else "")
            if len(vjepa_ids) > 1:
                S.add_log(
                    f"[CYAN] V-JEPA interleaved mix ({len(vjepa_ids)} datasets): {', '.join(vjepa_ids)}",
                    "cyan",
                )
            else:
                ds_id = vjepa_ids[0]
                try:
                    probe = probe_hf_train_stream(ds_id, kind="vjepa", token=HF_TOKEN or None, max_rows=150)
                    if probe.get("ok"):
                        S.add_log(
                            f"[CYAN] HF probe OK · {ds_id} · keys {probe.get('keys', [])[:6]} · "
                            f"{str(probe.get('text_preview', ''))[:72]}…",
                            "cyan",
                        )
                    else:
                        S.add_log(f"[YELLOW] HF probe: {probe.get('error')} — will try stream anyway", "yellow")
                except Exception as exc:
                    S.add_log(f"[YELLOW] HF probe failed: {exc}", "yellow")

            def _track_vjepa(row: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
                S.vjepa_hits += 1
                return row, label

            vjepa_body = body
            if len(vjepa_ids) > 1 and body.shuffle_hf_datasets:
                vjepa_body = body.model_copy(update={"shuffle_hf_datasets": False})
                S.add_log(
                    "[YELLOW] multi V-JEPA interleave — HF shuffle disabled (avoids materializing huge latent corpora)",
                    "yellow",
                )
            pack_rows = int(body.vjepa_pack_rows or 0)
            if pack_rows > 1:
                S.add_log(
                    f"[CYAN] MCF v2 latent packing ON — {pack_rows} frames/block · "
                    f"min {body.vjepa_pack_min_chars} chars (GCT scalar bridge)",
                    "cyan",
                )
            if body.vjepa_local_cache:
                from vjepa_local_cache import cache_dir, cache_ready, cached_datasets

                local_ds = cached_datasets(vjepa_ids)
                if cache_ready(vjepa_ids):
                    S.add_log(
                        f"[CYAN] V-JEPA local NVMe cache · {cache_dir()} · "
                        f"{len(vjepa_ids)} datasets (IBM COS backup · no HF streaming)",
                        "cyan",
                    )
                elif local_ds:
                    missing = [d for d in vjepa_ids if d not in local_ds]
                    S.add_log(
                        f"[CYAN] V-JEPA hybrid cache · local {len(local_ds)}/{len(vjepa_ids)} · "
                        f"HF stream for {', '.join(missing)}",
                        "cyan",
                    )
                else:
                    S.add_log(
                        "[YELLOW] vjepa_local_cache requested but no cache files — HF streaming",
                        "yellow",
                    )

            _stream_finite_rows(
                eng,
                vjepa_body,
                target_bytes,
                skip_to,
                _vjepa_packed_stream_factory(body),
                _vjepa_row_text,
                "vjepa",
                f"vjepa:{ds_label}",
                on_row=_track_vjepa,
                local_pool=local_pool,
                local_idx=local_idx,
            )

        _finish_training(eng, body, target_bytes)

    except Exception as exc:
        S.error = str(exc)
        S.status = "interrupted"
        S.add_log(f"[RED] error: {exc}", "red")
        S.add_log(f"[RED] {traceback.format_exc()[-600:]}", "red")
        S.running = False
        _release_gpu_memory()
    finally:
        S.running = False
        _clear_training_heartbeat()
        if S.status in ("streaming", "stopping"):
            S.status = "interrupted"


@app.on_event("startup")
def startup_hydrate():
    global _hf_probe_started
    _hydrate_from_cos()
    if not S.running and S.status in ("stopping", "streaming"):
        S.status = "interrupted"
    S.add_log(f"[CYAN] {PUBLIC_URL} · device={DEVICE} · target {S.target_gb:g} GB", "cyan")
    if not _hf_probe_started and HF_TOKEN:
        _hf_probe_started = True
        threading.Thread(target=_hf_probe_background, daemon=True).start()
    if os.environ.get("AUTO_RESUME", "0") != "1":
        return
    if S.bytes_written > 0 and S.status in ("interrupted", "stopped", "streaming") and not S.running:
        body = StartIn(config=S.config or DEFAULT_CONFIG, target_gb=S.target_gb or DEFAULT_TARGET_GB, resume=True)
        threading.Thread(target=_stream_and_train, args=(body,), daemon=True).start()
        S.add_log("[GREEN] auto-resume CUDA pretrain", "green")


@app.get("/")
def root():
    return FileResponse("public/index.html")


def _agent_live_snapshot() -> dict[str, Any]:
    eng = _get_engine()
    snap = S.snapshot() if S.bytes_written or S.running else {}
    return {
        "hf_token_loaded": bool(HF_TOKEN),
        "cos_configured": bool(IBM_API_KEY),
        "device": DEVICE,
        "running": S.running,
        "train_step": eng.step,
        "last_loss": eng.last_loss,
        "target_gb": S.target_gb,
        "data_source": S.data_source,
        "checkpoint_name": S.checkpoint_name,
        "bytes_written": S.bytes_written,
        "can_resume": snap.get("can_resume"),
        "origin_url": "https://app7-nextaura-us.284w7l87aq94.us-south.codeengine.appdomain.cloud",
    }


@app.get("/agent.json")
def agent_json():
    contract = build_agent_contract(live=_agent_live_snapshot())
    return JSONResponse(contract)


@app.get("/agent.txt")
def agent_txt():
    contract = build_agent_contract(live=_agent_live_snapshot())
    return PlainTextResponse(render_agent_txt(contract))


@app.get("/.well-known/agent.json")
def well_known_agent_json():
    return agent_json()


@app.get("/agent-graph.json")
def agent_graph_json():
    return FileResponse("public/agent-graph.json", media_type="application/json")


@app.get("/llms.txt")
def llms_txt():
    contract = build_agent_contract(live=_agent_live_snapshot())
    return PlainTextResponse(render_llms_txt(contract), media_type="text/plain; charset=utf-8")


@app.get("/mm-vjepa-recipes.txt")
def mm_vjepa_recipes_txt():
    return FileResponse("public/mm-vjepa-recipes.txt", media_type="text/plain; charset=utf-8")


@app.get("/robots.txt")
def robots_txt():
    return PlainTextResponse(render_robots_txt(), media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml")
def sitemap_xml():
    return PlainTextResponse(render_sitemap_xml(), media_type="application/xml; charset=utf-8")


@app.get("/api/agent/index")
def agent_index():
    """Alias for /agent.json — full scrape index."""
    return agent_json()


DRIVE_STATE_FILE = os.path.join(DATA_DIR, "agent-drive-state.json")
MASTER_STATE_FILE = os.path.join(DATA_DIR, "master-agent-state.json")


@app.get("/api/agent/drive-status")
def agent_drive_status():
    """Autonomous drive_agent.py ladder state (legacy) or master-agent if active."""
    if os.path.isfile(MASTER_STATE_FILE):
        try:
            with open(MASTER_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            data["source"] = "master-agent"
            return data
        except (json.JSONDecodeError, OSError):
            pass
    if os.path.isfile(DRIVE_STATE_FILE):
        try:
            with open(DRIVE_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            data["source"] = "drive-agent"
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "agent": "master-agent",
        "phase": "idle",
        "phase_index": 0,
        "running": False,
        "enabled": MASTER_AGENT_ENABLED,
        "distill_live_enabled": DISTILL_LIVE_ENABLED,
        "note": "dormant" if not MASTER_AGENT_ENABLED else "active",
        "updated_at": None,
        "source": "default",
    }


@app.get("/api/agent/master-status")
def agent_master_status():
    """Master orchestrator state (51M distill-live ladder)."""
    return agent_drive_status()


def _build_agent_models_payload() -> dict[str, Any]:
    eng = _get_engine()
    data_dir = _ensure_data_dir()
    health_tok = TOKENIZER_OPTIONS
    return build_local_models_payload(
        host=HOST,
        checkpoints=_list_local_checkpoints(),
        remote_checkpoints=list_remote_checkpoints(),
        infer_catalog=_infer_model_catalog(),
        teachers=distill_catalog(data_dir),
        tokenizers=health_tok,
        loaded=S.checkpoint_name,
        loaded_step=eng.step,
    )


@app.get("/api/agent/models")
def agent_models_index():
    """All checkpoints, teachers, and tokenizers on this host — for agent scrapers."""
    return _build_agent_models_payload()


@app.get("/api/agent/platform")
def agent_platform_index(refresh: bool = False):
    """Platform-wide scrape index: all NextAura hosts + aggregated model catalog."""
    if not refresh:
        cached = load_cached_platform_index()
        if cached:
            age_sec = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(cached["scraped_at"].replace("Z", "+00:00"))
            ).total_seconds()
            if age_sec < 300:
                return cached
    payload = build_platform_index(
        self_url=PUBLIC_URL,
        local_models=_build_agent_models_payload(),
    )
    try:
        save_platform_index(payload)
    except OSError:
        pass
    return payload


@app.post("/api/agent/scrape")
def agent_scrape_refresh():
    """Force refresh platform index (all hosts) and persist cache."""
    payload = build_platform_index(
        self_url=PUBLIC_URL,
        local_models=_build_agent_models_payload(),
    )
    path = save_platform_index(payload)
    return {"ok": True, "path": path, "model_count": payload.get("model_count"), "hosts_reachable": payload.get("hosts_reachable")}


@app.get("/api/health")
def health():
    eng = _get_engine()
    return {
        "status": "online",
        "host": HOST,
        "public_url": PUBLIC_URL,
        "stage1": "pretrain",
        "stage2": "infer",
        "dataset": DATASET_ID,
        "default_config": DEFAULT_CONFIG,
        "target_gb": DEFAULT_TARGET_GB,
        "chinchilla_tokens_per_param": CHINCHILLA_TOKENS_PER_PARAM,
        "chinchilla_bytes_per_token": BYTES_PER_TRAINING_TOKEN,
        "chinchilla_target_tokens": chinchilla_target_tokens(eng.param_count),
        "chinchilla_target_gb": chinchilla_target_gb(eng.param_count),
        "chinchilla_train_steps": chinchilla_train_steps(
            eng.param_count,
            block_size=eng.model_config.block_size,
            batch_size=eng.batch_size,
        ),
        "default_total_train_steps": DEFAULT_TOTAL_TRAIN_STEPS,
        "default_warmup_steps": DEFAULT_WARMUP_STEPS,
        "default_save_every_n_steps": DEFAULT_SAVE_EVERY_N_STEPS,
        "default_mixed_precision": DEFAULT_MIXED_PRECISION,
        "default_use_flash_attention": True,
        "default_lr_schedule": "cosine",
        "default_micro_batch_size": DEFAULT_MICRO_BATCH,
        "default_grad_accum_steps": 4,
        "max_cuda_micro_batch": MAX_CUDA_MICRO_BATCH,
        "local_corpus_mix_label": LOCAL_CORPUS_MIX_LABEL,
        "local_corpus_mix_files": list(LOCAL_CORPUS_MIX_FILES),
        "model_params": eng.param_count,
        "learning_rate": eng.learning_rate,
        "batch_size": eng.batch_size,
        "default_learning_rate": DEFAULT_LR,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "hf_token_loaded": bool(HF_TOKEN),
        "distill_live_enabled": DISTILL_LIVE_ENABLED,
        "master_agent_enabled": MASTER_AGENT_ENABLED,
        "cos_configured": bool(IBM_API_KEY),
        "storage": storage_status(),
        "torch": True,
        "device": DEVICE,
        "cos_bucket": COS_BUCKET,
        "voice_to_plan_url": VOICE_TO_PLAN_URL,
        "configs": FINEWEB_CONFIGS,
        "data_sources": list(DATA_SOURCES),
        "default_data_source": "multimodal",
        "multimodal_datasets": list(MULTIMODAL_DATASETS.keys()),
        "vjepa_datasets": list(VJEPA_DATASETS.keys()),
        "multimodal_datasets_verified": _verified_hf_ids("multimodal"),
        "vjepa_datasets_verified": _verified_hf_ids("vjepa"),
        "hf_dataset_probe": _hf_probe_snapshot(),
        "default_multimodal_dataset": DEFAULT_MULTIMODAL_DATASET,
        "default_vjepa_dataset": DEFAULT_VJEPA_DATASET,
        "tokenizer_options": TOKENIZER_OPTIONS,
        "default_tokenizer_id": DEFAULT_TOKENIZER_ID,
        "tokenizer_id": eng.tokenizer_id,
        "default_jsonl_file": "4563-curated.jsonl",
        "sft_topics": topic_catalog(_ensure_data_dir()),
        "distill_topics": topic_catalog(_ensure_data_dir()),
        "distill_teachers": TEACHER_REGISTRY,
        "distill_target_teachers": {
            "coding": "tjarvis91/Q-Coder-50M-Sovereign",
            "math": "nkthebass/tinybrainbot-100m-v3-math",
            "physics": "AlexWortega/moe100m-physics-tinybpe",
        },
        "default_jsonl_mix_ratio": 0.30,
        "default_shuffle_hf_datasets": False,
        "default_mix_local_corpora": True,
        "default_hf_stream_fable": True,
        "default_hf_stream_sol": True,
        "default_hf_stream_kimi": True,
        "default_hf_stream_nemotron": True,
        "default_hf_stream_unsolved_math": True,
        "default_hf_stream_preference": True,
        "default_hf_stream_fineweb": False,
        "min_target_gb": MIN_TARGET_GB,
        "default_n_layer": DEFAULT_N_LAYER,
        "default_n_head": DEFAULT_N_HEAD,
        "default_n_embd": DEFAULT_N_EMBD,
        "default_vocab_size": DEFAULT_VOCAB_SIZE,
        "default_ffn_dim": DEFAULT_FFN_DIM,
        "default_model_params": FOOTPRINT_10M_PARAMS,
        "footprint_ladder": {
            "10m": {
                "params": FOOTPRINT_10M_PARAMS,
                "chinchilla_gb": chinchilla_target_gb(FOOTPRINT_10M_PARAMS),
                "smoke_gb": DEFAULT_SMOKE_TARGET_GB,
                "smoke_quarter_gb": round(chinchilla_target_gb(FOOTPRINT_10M_PARAMS) * 0.25, 3),
            },
            "50m": {
                "params": FOOTPRINT_50M_PARAMS,
                "chinchilla_gb": chinchilla_target_gb(FOOTPRINT_50M_PARAMS),
                "smoke_quarter_gb": round(chinchilla_target_gb(FOOTPRINT_50M_PARAMS) * 0.25, 3),
            },
            "25m": {
                "params": FOOTPRINT_25M_PARAMS,
                "chinchilla_gb": chinchilla_target_gb(FOOTPRINT_25M_PARAMS),
                "smoke_quarter_gb": round(chinchilla_target_gb(FOOTPRINT_25M_PARAMS) * 0.25, 3),
                "config": FOOTPRINT_25M_CFG.to_dict(),
            },
        },
        "ui_defaults": {
            "n_layer": FOOTPRINT_10M_CFG.n_layer,
            "n_head": FOOTPRINT_10M_CFG.n_head,
            "n_embd": FOOTPRINT_10M_CFG.n_embd,
            "block_size": FOOTPRINT_10M_CFG.block_size,
            "vocab_size": FOOTPRINT_10M_CFG.vocab_size,
            "ffn_dim": FOOTPRINT_10M_CFG.ffn_dim,
            "tokenizer_id": DEFAULT_TOKENIZER_ID,
            "target_gb": DEFAULT_SMOKE_TARGET_GB,
            "model_params": FOOTPRINT_10M_PARAMS,
        },
        "n_layer": eng.model_config.n_layer,
        "n_head": eng.model_config.n_head,
        "n_embd": eng.model_config.n_embd,
        "block_size": eng.model_config.block_size,
        "dropout": eng.model_config.dropout,
        "weight_decay": eng.weight_decay,
        "warmup_steps": eng.warmup_steps,
        "mixed_precision": eng.mixed_precision,
        "save_every_n_steps": eng.save_every_n_steps,
        "micro_batch_size": eng.micro_batch_size,
        "grad_accum_steps": eng.grad_accum_steps,
        "vocab_size": eng.model_config.vocab_size,
        "ffn_dim": eng.model_config.ffn_dim,
        "use_flash_attention": eng.model_config.use_flash_attention,
        "scalar_input_projection": eng.model_config.scalar_input_projection,
        "gradient_checkpointing": eng.model_config.gradient_checkpointing,
        "lr_schedule": eng.lr_schedule,
        "total_train_steps": eng.total_train_steps,
        "flash_attention_available": flash_attention_available(),
        "capabilities": {
            "gpt_trainer": True,
            "flash_attention": flash_attention_available(),
            "scalar_projection": True,
            "gradient_checkpointing": True,
            "hf_import": True,
            "configurable_ffn": True,
            "cosine_lr": True,
        },
        "reward_model_id": RM_MODEL_ID,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/model/estimate")
def model_estimate(
    n_layer: int = DEFAULT_N_LAYER,
    n_head: int = DEFAULT_N_HEAD,
    n_embd: int = DEFAULT_N_EMBD,
    block_size: int = DEFAULT_BLOCK_SIZE,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    ffn_dim: int | None = None,
    use_flash_attention: bool = False,
    scalar_input_projection: bool = False,
    gradient_checkpointing: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mode: str = "pretrain",
):
    try:
        cfg = ModelConfig(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            vocab_size=vocab_size,
            ffn_dim=ffn_dim,
            use_flash_attention=use_flash_attention,
            scalar_input_projection=scalar_input_projection,
            gradient_checkpointing=gradient_checkpointing,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    model_params = estimate_params(cfg)
    return {
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "model_params": model_params,
        "block_size": cfg.block_size,
        "vocab_size": cfg.vocab_size,
        "ffn_dim": cfg.ffn_dim,
        "use_flash_attention": cfg.use_flash_attention,
        "scalar_input_projection": cfg.scalar_input_projection,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "chinchilla": chinchilla_scaling_bundle(
            model_params,
            block_size=cfg.block_size,
            batch_size=batch_size,
            n_embd=cfg.n_embd,
            vocab_size=cfg.vocab_size,
            mode=mode if mode in ("pretrain", "finetune") else "pretrain",
        ),
    }


@app.get("/api/scaling/chinchilla")
def chinchilla_scaling(
    model_params: int = DEFAULT_MODEL_PARAMS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_embd: int = DEFAULT_N_EMBD,
    vocab_size: int = 50257,
    mode: str = "pretrain",
):
    if model_params < 1_000_000:
        return JSONResponse({"error": "model_params must be >= 1M"}, status_code=400)
    return chinchilla_scaling_bundle(
        model_params,
        block_size=block_size,
        batch_size=batch_size,
        n_embd=n_embd,
        vocab_size=vocab_size,
        mode=mode if mode in ("pretrain", "finetune") else "pretrain",
    )


class HfConvertIn(BaseModel):
    repo_id: str = Field(default="Qwen/Qwen2.5-3B-Instruct", max_length=260)
    backend: str = Field(default="hf", max_length=16)
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None
    block_size: int | None = None


@app.post("/api/convert/hf")
def convert_hf_checkpoint(body: HfConvertIn):
    if S.running:
        raise HTTPException(409, "Stop training before importing a model")
    backend = (body.backend or "hf").strip().lower()
    slug = body.repo_id.replace("/", "-").replace("\\", "-")[:120]
    try:
        if backend in ("gpt2", "gpt2_map", "custom_gpt"):
            eng = _get_engine()
            if eng.step > 0:
                cfg = eng.model_config
                if cfg.n_layer != 12 or cfg.n_embd != 768 or cfg.vocab_size != 50257:
                    raise HTTPException(
                        409,
                        "Architecture must be 12L×768d · vocab 50257 (gpt2) before legacy GPT-2 weight map — "
                        "Reset session, apply 124M footprint, then import with backend=gpt2_map.",
                    )
            payload, meta = import_hf_gpt2(
                body.repo_id,
                token=HF_TOKEN or None,
                n_layer=body.n_layer,
                n_head=body.n_head,
                n_embd=body.n_embd,
                block_size=body.block_size,
            )
            meta["model_backend"] = "gpt"
        else:
            checkpoint_root = _ensure_checkpoint_dir()
            payload, meta = import_hf_causal(
                body.repo_id,
                weights_root=checkpoint_root,
                token=HF_TOKEN or None,
                block_size=body.block_size,
                device=DEVICE,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"HF import failed: {exc}") from exc
    name = f"imported-{slug}.pt"
    path = _checkpoint_path(name)
    with open(path, "wb") as out:
        out.write(payload)
    result = _import_checkpoint_bytes(payload, name, skip_running_check=True)
    return {"ok": True, "import": meta, **result}


@app.get("/api/capabilities")
def capabilities():
    return {
        "gpt_trainer": True,
        "flash_attention_sdp": flash_attention_available(),
        "scalar_input_projection": True,
        "gradient_checkpointing": True,
        "hf_gpt2_import": True,
        "hf_causal_import": True,
        "hf_native_training": True,
        "model_backends": ["gpt", "hf"],
        "configurable_ffn_dim": True,
        "configurable_vocab_size": True,
        "lr_schedules": ["constant", "warmup", "cosine"],
        "hf_dataset_shuffle": True,
        "mixed_precision": ["fp32", "fp16", "bf16"],
        "unsloth": False,
        "distill_live_enabled": DISTILL_LIVE_ENABLED,
        "master_agent_enabled": MASTER_AGENT_ENABLED,
        "note": "Native HF causal LM (Qwen, Llama, …) via backend=hf. Legacy GPT-2 weight map: backend=gpt2_map. Unsloth LoRA not wired yet — use gradient_checkpointing + micro_batch=1 for large models.",
    }


class RewardScoreIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1, max_length=16000)


@app.post("/api/eval/reward")
def eval_reward(body: RewardScoreIn):
    import torch

    model, tokenizer = _get_reward_scorer()
    device = next(model.parameters()).device
    inputs = tokenizer(body.question, body.answer, return_tensors="pt").to(device)
    with torch.no_grad():
        score = float(model(**inputs).logits[0].cpu().item())
    return {
        "ok": True,
        "score": score,
        "model": RM_MODEL_ID,
        "question": body.question[:200],
    }


@app.get("/api/gpu/vram")
def gpu_vram(peers: int = 1):
    return _gpu_vram_snapshot(include_peers=bool(peers))


_muse_last_reported_step = -1
MUSE_INFER_URL = os.environ.get("MUSE_INFER_URL", "http://127.0.0.1:8766")


def _muse_infer_health() -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(MUSE_INFER_URL + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _muse_infer_ask(question: str, max_new_tokens: int = 256) -> dict[str, Any]:
    payload = json.dumps({"question": question, "max_new_tokens": max_new_tokens}).encode()
    req = urllib.request.Request(
        MUSE_INFER_URL + "/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=exc.code, detail=body[:800]) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Muse infer sidecar unreachable: {exc}") from exc


def _external_muse_status() -> dict[str, Any] | None:
    """Expose the separately managed Muse QLoRA run through the main GUI."""
    active = False
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
            except OSError:
                continue
            if b"sft_qlora.py" in cmd:
                active = True
                break
    except OSError:
        pass

    log_path = os.environ.get("MUSE_TRAIN_LOG")
    if not log_path:
        candidates = ["/tmp/muse_phase2.log", "/tmp/muse_phase1.log"]
        existing = [p for p in candidates if os.path.isfile(p)]
        log_path = max(existing, key=os.path.getmtime) if existing else "/tmp/muse_phase2.log"
    try:
        with open(log_path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(log_path) - 2_000_000))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    progress = re.findall(r"(\d+)/(\d+)\s+\[[^\]]*<([^,]+),\s*([\d.]+)s/it\]", text)
    losses = re.findall(r"'loss':\s*'([0-9.eE+-]+)'", text)
    complete = "SFT_DONE" in text or "PHASE1_EXIT_OK" in text
    if "/muse_phase1.log" in log_path.replace("\\", "/") and os.path.isfile("/tmp/muse_phase2.log"):
        complete = "SFT_DONE" in text
    if not active and not complete:
        return None

    step, total, eta, seconds_per_step = (0, 3000, "", 0.0)
    if progress:
        step_s, total_s, eta, seconds_s = progress[-1]
        step, total, seconds_per_step = int(step_s), int(total_s), float(seconds_s)
    loss = float(losses[-1]) if losses else None
    tok_s = (8192 * 16 / seconds_per_step) if seconds_per_step else 0.0
    status_value = "streaming" if active else "complete"
    sidecar = _muse_infer_health() or {}
    return {
        "running": active,
        "training": active,
        "status": status_value,
        "session_status": status_value,
        "complete": complete and not active,
        "data_source": "muse-sft",
        "dataset": "Nemotron-SFT-SWE-v3.5 + OpenCodeReasoning",
        "hf_dataset_id": "nvidia/Nemotron-SFT-SWE-v3.5 + nvidia/OpenCodeReasoning",
        "hf_stream_label": "Muse coding SFT",
        "model": "Muse-Glimmer-30B-QLoRA",
        "model_label": "Muse-Glimmer-30B · QLoRA coding SFT",
        "model_params": 30_000_000_000,
        "model_params_m": 30_000.0,
        "model_backend": "huggingface-qlora",
        "n_layer": 52,
        "n_head": 32,
        "n_embd": 6656,
        "ffn_dim": 19968,
        "vocab_size": 202048,
        "block_size": 8192,
        "tokenizer_id": "Muse-Glimmer-30B",
        "training_mode": "sft",
        "training_mode_label": "Supervised fine-tuning",
        "train_step": step,
        "run_steps_done": step,
        "total_train_steps": total,
        "total_train_steps_cap": total,
        "progress_pct": round(step * 100 / total, 2) if total else 0.0,
        "last_loss": loss,
        "last_tok_s": round(tok_s, 1),
        "step_interval_s": seconds_per_step or None,
        "model_ready": True,
        "inference_available": bool(sidecar.get("ready") or sidecar.get("ok")),
        "weights_source": sidecar.get("adapter") or "Muse-Glimmer-30B base + live QLoRA adapter",
        "run_goal_title": "Muse-Glimmer coding agent · Phase 1 SFT",
        "run_goal": {
            "title": "Muse-Glimmer coding agent · Phase 1 SFT",
            "detail": "sft-v2 · Nemotron SWE + OpenCodeReasoning · tools+messages format · ≤2 epochs · collapse stop",
            "stop_primary": "steps",
            "step_cap": total,
            "step_progress_pct": round(step * 100 / total, 2) if total else 0.0,
            "outcome": status_value,
            "outcome_label": f"{status_value} · ETA {eta}" if eta else status_value,
            "goal_met": complete and not active,
        },
    }


@app.get("/api/status")
def status():
    global _muse_last_reported_step
    if not S.running and S.bytes_written == 0:
        _hydrate_from_cos()
    snap = S.snapshot()
    try:
        snap.update(_infer_context(_get_engine()))
    except Exception:
        pass
    muse = _external_muse_status()
    if muse:
        snap.update(muse)
        step = int(muse["train_step"])
        if muse["running"] and step != _muse_last_reported_step:
            S.add_log(
                f"[GREEN] Muse SFT step {step:,}/{muse['total_train_steps']:,} "
                f"loss {muse['last_loss'] if muse['last_loss'] is not None else '—'} · "
                f"{muse['step_interval_s'] or 0:.2f}s/step · {muse['last_tok_s']:.0f} tok/s",
                "green",
            )
            _muse_last_reported_step = step
    return snap


@app.get("/api/log")
def log(since: int = 0):
    return {"lines": S.log_lines[since:], "index": len(S.log_lines)}


@app.post("/api/start")
def start(body: StartIn = Body(default_factory=StartIn)):
    with _run_lock:
        if S.running:
            return {"ok": True, "running": True}
        orig_source = body.data_source
        if body.resume and S.data_source and orig_source == StartIn().data_source:
            # Resume with default body: inherit the interrupted session's config
            # instead of validating against factory defaults (app8 disk-full
            # resume hit a false preflight error on the default jsonl path).
            body.data_source = S.data_source
            orig_source = S.data_source
            if S.hf_dataset_id:
                body.hf_dataset_id = S.hf_dataset_id
            body.target_gb = body.target_gb or (S.target_bytes / (1024**3) if S.target_bytes else body.target_gb)
            if S.target_bytes:
                body.target_gb = max(body.target_gb, S.target_bytes / (1024**3))
        if orig_source == "distill-live" and not DISTILL_LIVE_ENABLED:
            return JSONResponse(
                {
                    "error": "distill-live is dormant on this host — set DISTILL_LIVE_ENABLED=1 to enable manual runs",
                    "distill_live_enabled": False,
                },
                status_code=403,
            )
        if orig_source not in DATA_SOURCES:
            return JSONResponse({"error": f"Unknown data_source '{orig_source}'"}, status_code=400)
        norm = _normalize_training_body(body)
        if norm.data_source == "fineweb" and norm.config not in FINEWEB_CONFIGS:
            return JSONResponse({"error": f"Unknown config '{norm.config}'"}, status_code=400)
        if orig_source == "multimodal":
            ds = norm.hf_dataset_id or DEFAULT_MULTIMODAL_DATASET
            if ds not in MULTIMODAL_DATASETS:
                return JSONResponse({"error": f"Unknown multimodal dataset '{ds}'"}, status_code=400)
        if orig_source == "vjepa":
            ds = norm.hf_dataset_id or DEFAULT_VJEPA_DATASET
            if ds not in VJEPA_DATASETS:
                return JSONResponse({"error": f"Unknown V-JEPA dataset '{ds}'"}, status_code=400)
        if orig_source in SFT_TOPIC_SOURCES:
            missing = _ensure_sft_topic_file(orig_source)
            if missing:
                return JSONResponse({"error": missing}, status_code=400)
        if orig_source in DISTILL_TOPIC_SOURCES:
            missing = _ensure_distill_file(orig_source)
            if missing:
                return JSONResponse({"error": missing}, status_code=400)
        eng = _get_engine()
        if eng.is_hf_backend and body.data_source == "vjepa":
            return JSONResponse(
                {"error": "V-JEPA / scalar 12d requires custom NextAura GPT — use data_source=mixed on HF models"},
                status_code=400,
            )
        pf = run_preflight(
            norm,
            data_dir=_ensure_data_dir(),
            status=S.snapshot(),
            loaded_local=S.loaded_local_checkpoint,
            train_step=eng.step,
            last_loss=eng.last_loss,
            running=S.running,
            session_status=S.status,
        )
        if not pf.get("ok"):
            return JSONResponse(
                {"error": "Preflight failed", "preflight": pf},
                status_code=400,
            )
        if orig_source not in DISTILL_TOPIC_SOURCES and orig_source not in SFT_TOPIC_SOURCES:
            if norm.data_source in ("jsonl", "local-corpus-mix") or _wants_local_mix(norm):
                missing = _validate_local_corpus(norm)
                if missing:
                    return JSONResponse(
                        {"error": f"{missing} — scp to data/ first"},
                        status_code=400,
                    )
        elif orig_source in DISTILL_TOPIC_SOURCES:
            missing = _ensure_distill_file(orig_source)
            if missing:
                return JSONResponse({"error": missing}, status_code=400)
        S.status = "starting"
        S.stop_reason = ""
        S.stop_reason_label = ""
        S.data_source = orig_source
        S.jsonl_file = norm.jsonl_file
        if pf.get("warnings"):
            S.add_log("[YELLOW] Preflight warnings: " + "; ".join(pf["warnings"][:3]), "yellow")
        threading.Thread(target=_stream_and_train, args=(body,), daemon=True).start()
    return {"ok": True, "preflight": pf, **S.snapshot()}


@app.post("/api/preflight")
def preflight_check(body: StartIn = Body(default_factory=StartIn)):
    norm = _normalize_training_body(body)
    eng = _get_engine()
    pf = run_preflight(
        norm,
        data_dir=_ensure_data_dir(),
        status=S.snapshot(),
        loaded_local=S.loaded_local_checkpoint,
        train_step=eng.step,
        last_loss=eng.last_loss,
        running=S.running,
        session_status=S.status,
    )
    return pf


@app.post("/api/resume")
def resume(body: StartIn = Body(default_factory=StartIn)):
    body.resume = True
    return start(body)


@app.post("/api/stop")
def stop():
    S.running = False
    S.status = "stopping"
    return {"ok": True}


@app.post("/api/pause")
def pause():
    S.paused = not S.paused
    return {"paused": S.paused}


@app.post("/api/import/checkpoint")
async def import_checkpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    safe = _safe_checkpoint_name(file.filename)
    root = _ensure_checkpoint_dir()
    dest = os.path.join(root, safe)
    size = 0
    max_bytes = MAX_CHECKPOINT_MB * 1024 * 1024
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Checkpoint too large (max {MAX_CHECKPOINT_MB} MB)")
                out.write(chunk)
        with open(dest, "rb") as inp:
            payload = inp.read()
    except HTTPException:
        if os.path.exists(dest):
            os.remove(dest)
        raise
    except Exception as exc:
        if os.path.exists(dest):
            os.remove(dest)
        raise HTTPException(500, f"Upload failed: {exc}") from exc
    return _import_checkpoint_bytes(payload, safe)


class LoadCheckpointIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@app.get("/api/runs/history")
def runs_history():
    return run_history_bundle(
        DATA_DIR,
        _ensure_checkpoint_dir(),
        loaded=S.checkpoint_name or "",
    )


class RunRecordIn(BaseModel):
    event: str = "note"
    agent: str = ""
    lesson: str = ""
    checkpoint: str = ""
    step: int | None = None
    jsonl_file: str = ""
    jsonl_rows: int | None = None
    phase: str = ""
    verdict: str = ""
    reward_mean_post: float | None = None
    reward_mean_baseline: float | None = None
    reward_wins: int | None = None
    reward_n: int | None = None


@app.post("/api/runs/record")
def runs_record(body: RunRecordIn):
    fields = body.model_dump(exclude_none=True, exclude={"event", "agent", "lesson"})
    if body.reward_mean_post is not None and body.checkpoint:
        row = record_reward_eval(
            DATA_DIR,
            checkpoint=body.checkpoint,
            step=int(body.step or 0),
            mean_reward=float(body.reward_mean_post),
            n_prompts=int(body.reward_n or 20),
            agent=body.agent or "agent",
            baseline_mean=body.reward_mean_baseline,
            wins=body.reward_wins,
        )
    else:
        row = record_engineering(
            DATA_DIR,
            event=body.event,
            agent=body.agent,
            lesson=body.lesson,
            **fields,
        )
    return {"ok": True, "record": row}


def _reward_generate_score(eng: TrainEngine, body: RewardPanelIn):
    def _gen(question: str) -> str:
        q = wrap_chat_prompt(question.strip())
        ans = eng.generate(
            q,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            repetition_penalty=body.repetition_penalty,
        )
        return truncate_chat_answer(ans)

    def _score(question: str, answer: str) -> float:
        import torch

        model, tokenizer = _get_reward_scorer()
        device = next(model.parameters()).device
        inputs = tokenizer(question, answer, return_tensors="pt").to(device)
        with torch.no_grad():
            return float(model(**inputs).logits[0].cpu().item())

    return _gen, _score


def _apply_eval_result(panel: dict[str, Any], *, record: bool, agent: str, baseline_mean: float | None, live: bool) -> None:
    if not panel.get("ok"):
        return
    mean = float(panel["mean_reward"])
    ckpt = str(panel.get("checkpoint") or S.checkpoint_name or "")
    step = int(panel.get("step") or 0)
    S.last_eval_mean = mean
    S.last_eval_checkpoint = ckpt
    S.last_eval_at = datetime.now(timezone.utc).isoformat()
    if S.best_eval_mean is None or mean > S.best_eval_mean:
        S.best_eval_mean = mean
        S.best_eval_checkpoint = ckpt
    if record:
        record_reward_eval(
            DATA_DIR,
            checkpoint=ckpt,
            step=step,
            mean_reward=mean,
            n_prompts=int(panel.get("n_prompts") or 0),
            agent=agent,
            baseline_mean=baseline_mean,
        )
    tag = "live" if live else "eval"
    delta = ""
    if baseline_mean is not None:
        delta = f" · Δ{baseline_mean - mean:+.3f}" if False else f" · Δ{mean - baseline_mean:+.3f}"
    S.add_log(
        f"[CYAN] RM {tag} · {ckpt} · mean {mean:.4f}{delta} · n={panel.get('n_prompts')}",
        "cyan",
    )


def _run_save_eval(checkpoint_name: str, step: int, live: bool) -> None:
    try:
        eng = _get_engine()
        if eng.step != step:
            return
        body = RewardPanelIn(limit=SAVE_EVAL_LIMIT, record=True, agent="save-eval", live=live)
        gen, score = _reward_generate_score(eng, body)
        panel = build_panel_result(
            eng=eng,
            checkpoint_name=checkpoint_name,
            generate=gen,
            score=score,
            limit=body.limit,
            live=live,
        )
        panel["model"] = RM_MODEL_ID
        baseline = S.best_eval_mean if S.best_eval_checkpoint and S.best_eval_checkpoint != checkpoint_name else None
        panel["baseline_mean"] = baseline
        if baseline is not None:
            panel["delta"] = round(panel["mean_reward"] - baseline, 4)
        _apply_eval_result(panel, record=body.record, agent=body.agent, baseline_mean=baseline, live=live)
    except Exception as exc:
        S.add_log(f"[YELLOW] save eval failed: {exc}", "yellow")


class RewardPanelIn(BaseModel):
    limit: int = Field(default=20, ge=1, le=60)
    max_new_tokens: int = Field(default=100, ge=16, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    baseline_mean: float | None = None
    record: bool = False
    agent: str = "agent"
    live: bool = False


@app.post("/api/eval/reward-panel")
def eval_reward_panel(body: RewardPanelIn = Body(default_factory=RewardPanelIn)):
    if S.running and not body.live:
        return JSONResponse(
            {"error": "Training running — pass live=true for on-save eval, or stop first"},
            status_code=409,
        )
    eng = _get_engine()
    if eng.step == 0:
        _hydrate_from_cos()
    if eng.step == 0:
        return JSONResponse({"error": "No checkpoint loaded"}, status_code=503)

    gen, score = _reward_generate_score(eng, body)
    panel = build_panel_result(
        eng=eng,
        checkpoint_name=S.checkpoint_name or f"step{eng.step}",
        generate=gen,
        score=score,
        limit=body.limit,
        live=body.live,
    )
    panel["model"] = RM_MODEL_ID
    baseline = body.baseline_mean
    if baseline is None and S.best_eval_mean is not None:
        baseline = S.best_eval_mean
    panel["baseline_mean"] = baseline
    if baseline is not None:
        panel["delta"] = round(panel["mean_reward"] - baseline, 4)

    if body.record or body.live:
        _apply_eval_result(panel, record=True, agent=body.agent, baseline_mean=baseline, live=body.live)

    panel = compare_to_timemoe({**panel, "modality": "text_rm"})
    panel["benchmark"] = benchmark_summary_for_api(DATA_DIR)
    return panel


class CompareCheckpointsIn(BaseModel):
    names: list[str] = Field(min_length=1, max_length=12)
    limit: int = Field(default=COMPARE_EVAL_LIMIT, ge=3, le=30)
    record: bool = True
    restore_loaded: bool = True


@app.get("/api/eval/leaderboard")
def eval_leaderboard_api(limit: int = 40):
    board = eval_leaderboard(DATA_DIR, limit=limit)
    board["loaded"] = S.checkpoint_name
    board["benchmark"] = benchmark_summary_for_api(DATA_DIR)
    board["live_eval"] = {
        "last_mean": S.last_eval_mean,
        "last_checkpoint": S.last_eval_checkpoint or None,
        "best_mean": S.best_eval_mean,
        "best_checkpoint": S.best_eval_checkpoint or None,
    }
    return board


@app.post("/api/eval/timemoe-baseline")
def eval_timemoe_baseline():
    if S.running:
        raise HTTPException(409, "Stop training before running TimeMoE benchmark")
    try:
        result = run_timemoe_benchmark(hf_token=HF_TOKEN or None)
    except Exception as exc:
        raise HTTPException(500, f"TimeMoE benchmark failed: {exc}") from exc
    if not result.get("ok"):
        raise HTTPException(500, result.get("error") or "TimeMoE benchmark failed")
    path = save_baseline_state(result, os.path.join(DATA_DIR, "timemoe-baseline.json"))
    if upcloud_configured():
        try:
            upload_checkpoint_file(
                str(path),
                name="timemoe-baseline.json",
            )
        except Exception:
            pass
    S.add_log(
        f"[CYAN] TimeMoE-50M baseline · MSE {result['mse']:.4f} · MAE {result['mae']:.4f} · n={result['n_windows']}",
        "cyan",
    )
    return {"ok": True, "result": result, "path": str(path), "benchmark": benchmark_summary_for_api(DATA_DIR)}


@app.get("/api/eval/benchmark")
def eval_benchmark_status():
    state = load_baseline_state(os.path.join(DATA_DIR, "timemoe-baseline.json"))
    data_dir = _ensure_data_dir()
    return {
        "reference_model": TIMEMOE_MODEL_ID,
        "reference_url": "https://huggingface.co/Maple728/TimeMoE-50M",
        "timemoe": (state or {}).get("reference") or benchmark_summary_for_api(DATA_DIR),
        "distill": {
            "teachers": TEACHER_REGISTRY,
            "target_teachers": {
                "coding": "tjarvis91/Q-Coder-50M-Sovereign",
                "math": "nkthebass/tinybrainbot-100m-v3-math",
                "physics": "AlexWortega/moe100m-physics-tinybpe",
            },
            "catalog": distill_catalog(data_dir),
        },
        "our_checkpoints": eval_leaderboard(DATA_DIR, limit=20),
        "loaded": S.checkpoint_name,
        "eval": {
            "last_mean": S.last_eval_mean,
            "best_mean": S.best_eval_mean,
            "best_checkpoint": S.best_eval_checkpoint or None,
        },
    }


@app.post("/api/eval/compare")
def eval_compare_checkpoints(body: CompareCheckpointsIn):
    if S.running:
        raise HTTPException(409, "Stop training before comparing checkpoints")
    original = S.checkpoint_name
    original_path: str | None = None
    if original and original.endswith(".pt"):
        try:
            original_path = _checkpoint_path(original)
        except HTTPException:
            original_path = None

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    baseline_mean: float | None = S.best_eval_mean

    for name in body.names:
        safe = _safe_checkpoint_name(name)
        path = _checkpoint_path(safe)
        if not os.path.isfile(path):
            errors.append({"name": safe, "error": "not on disk"})
            continue
        try:
            with open(path, "rb") as inp:
                payload = inp.read()
            _import_checkpoint_bytes(payload, safe)
            eng = _get_engine()
            panel_body = RewardPanelIn(limit=body.limit, record=body.record, agent="compare-eval")
            gen, score = _reward_generate_score(eng, panel_body)
            panel = build_panel_result(
                eng=eng,
                checkpoint_name=safe,
                generate=gen,
                score=score,
                limit=panel_body.limit,
            )
            panel["model"] = RM_MODEL_ID
            if baseline_mean is not None:
                panel["delta"] = round(panel["mean_reward"] - baseline_mean, 4)
            if body.record:
                _apply_eval_result(
                    panel,
                    record=True,
                    agent="compare-eval",
                    baseline_mean=baseline_mean,
                    live=False,
                )
            if baseline_mean is None and panel.get("mean_reward") is not None:
                baseline_mean = float(panel["mean_reward"])
            row = compare_to_timemoe(
                {
                    "checkpoint": safe,
                    "step": eng.step,
                    "mean_reward": panel.get("mean_reward"),
                    "delta": panel.get("delta"),
                    "n_prompts": panel.get("n_prompts"),
                    "ok": panel.get("ok", True),
                    "modality": "text_rm",
                }
            )
            results.append(row)
        except HTTPException as exc:
            errors.append({"name": safe, "error": str(exc.detail)})
        except Exception as exc:
            errors.append({"name": safe, "error": str(exc)})

    if body.restore_loaded and original_path and os.path.isfile(original_path):
        with open(original_path, "rb") as inp:
            _import_checkpoint_bytes(inp.read(), original)

    ranked = sorted(
        [r for r in results if r.get("mean_reward") is not None],
        key=lambda r: float(r["mean_reward"]),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    if best:
        S.best_eval_mean = float(best["mean_reward"])
        S.best_eval_checkpoint = str(best["checkpoint"])

    return {
        "ok": bool(results),
        "results": results,
        "ranked": ranked,
        "best": best,
        "errors": errors,
        "restored": original if body.restore_loaded else None,
        "benchmark": benchmark_summary_for_api(DATA_DIR),
        "leaderboard": eval_leaderboard(DATA_DIR, limit=20),
    }


@app.get("/api/checkpoints")
def list_checkpoints():
    ckpts = _list_local_checkpoints()
    summary = _checkpoint_disk_summary()
    remote = list_remote_checkpoints()
    return {
        "dir": os.path.abspath(_ensure_checkpoint_dir()),
        "checkpoints": ckpts,
        "remote_checkpoints": remote,
        "infer_models": _infer_model_catalog(),
        "loaded": S.checkpoint_name,
        "max_mb": MAX_CHECKPOINT_MB,
        "count": summary["count"],
        "total_mb": summary["total_mb"],
        "storage": storage_status(),
    }


class DeleteCheckpointsIn(BaseModel):
    names: list[str] | None = None
    all_except_loaded: bool = False
    keep_latest: int | None = Field(default=None, ge=1, le=500)


@app.delete("/api/checkpoints/{name}")
def delete_checkpoint(name: str):
    result = _delete_checkpoint_files([name])
    if name in result["skipped"] and name == S.checkpoint_name:
        raise HTTPException(409, "Cannot delete loaded checkpoint — reset session first")
    if not result["deleted"]:
        raise HTTPException(404, f"Checkpoint not found or skipped: {name}")
    return {"ok": True, **result}


@app.post("/api/checkpoints/delete")
def delete_checkpoints(body: DeleteCheckpointsIn):
    names: list[str] = []
    if body.names:
        names = body.names
    elif body.all_except_loaded:
        names = [c["name"] for c in _list_local_checkpoints() if c["name"] != S.checkpoint_name]
    elif body.keep_latest is not None:
        ckpts = sorted(_list_local_checkpoints(), key=lambda c: c["name"], reverse=True)
        keep = {c["name"] for c in ckpts[: body.keep_latest]}
        if S.checkpoint_name:
            keep.add(S.checkpoint_name)
        names = [c["name"] for c in ckpts if c["name"] not in keep]
    else:
        return JSONResponse({"error": "Provide names, all_except_loaded, or keep_latest"}, status_code=400)
    return {"ok": True, **_delete_checkpoint_files(names)}


@app.post("/api/checkpoints/reset-session")
def reset_training_session():
    return _reset_training_session()


@app.post("/api/import/checkpoint/local")
def import_checkpoint_local(body: LoadCheckpointIn):
    path = _checkpoint_path(body.name)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Checkpoint not found: {body.name}")
    with open(path, "rb") as inp:
        payload = inp.read()
    return _import_checkpoint_bytes(payload, body.name)


class LoadRemoteCheckpointIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source: str = Field(default="upcloud", pattern="^(upcloud)$")


@app.post("/api/import/checkpoint/remote")
def import_checkpoint_remote(body: LoadRemoteCheckpointIn):
    if not upcloud_configured():
        raise HTTPException(503, "UpCloud object storage not configured on this host")
    safe = _safe_checkpoint_name(body.name)
    local_path = _checkpoint_path(safe)
    if not os.path.isfile(local_path):
        try:
            payload = download_checkpoint_bytes(name=safe)
        except Exception as exc:
            raise HTTPException(404, f"Remote checkpoint not found: {safe}") from exc
        with open(local_path, "wb") as out:
            out.write(payload)
    else:
        with open(local_path, "rb") as inp:
            payload = inp.read()
    return _import_checkpoint_bytes(payload, safe)


@app.post("/api/checkpoints/sync-upcloud")
def sync_checkpoints_upcloud():
    if not upcloud_configured():
        raise HTTPException(503, "UpCloud object storage not configured")
    if S.running:
        raise HTTPException(409, "Stop training before syncing checkpoints")
    uploaded: list[str] = []
    errors: list[dict[str, str]] = []
    for ckpt in _list_local_checkpoints():
        path = _checkpoint_path(ckpt["name"])
        try:
            mirror = upload_checkpoint_file(path, name=ckpt["name"])
            if mirror.get("ok"):
                uploaded.append(ckpt["name"])
            else:
                errors.append({"name": ckpt["name"], "error": mirror.get("error") or "upload failed"})
        except Exception as exc:
            errors.append({"name": ckpt["name"], "error": str(exc)})
    return {
        "ok": not errors,
        "uploaded": uploaded,
        "errors": errors,
        "remote_count": len(list_remote_checkpoints()),
        "storage": storage_status(),
    }


@app.get("/api/infer/models")
def infer_models():
    eng = _get_engine()
    muse = _external_muse_status()
    if muse:
        live = {
            "id": "live:muse-glimmer-30b",
            "name": "Muse-Glimmer-30B-QLoRA",
            "label": f"Muse-Glimmer-30B · live QLoRA · step {muse['train_step']:,}",
            "source": "live",
            "loaded": True,
            "step": muse["train_step"],
            "params": muse["model_params"],
        }
        return {
            "loaded": live["name"],
            "loaded_step": muse["train_step"],
            "models": [live, *_infer_model_catalog()],
            "storage": storage_status(),
        }
    return {
        "loaded": S.checkpoint_name,
        "loaded_step": eng.step,
        "models": _infer_model_catalog(),
        "storage": storage_status(),
    }


class InferLoadIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source: str = Field(default="local", pattern="^(local|upcloud)$")


@app.post("/api/infer/load")
def infer_load(body: InferLoadIn):
    if S.running:
        raise HTTPException(409, "Stop training before switching inference model")
    if body.source == "upcloud":
        return import_checkpoint_remote(LoadRemoteCheckpointIn(name=body.name, source="upcloud"))
    return import_checkpoint_local(LoadCheckpointIn(name=body.name))


class SftGenerateIn(BaseModel):
    topics: list[str] = Field(default_factory=lambda: list(SFT_TOPICS))
    hf_augment: bool = False
    hf_limit: int = Field(default=40, ge=0, le=500)
    write_merged: bool = True


class DistillGenerateIn(BaseModel):
    topics: list[str] = Field(default_factory=lambda: list(DISTILL_TOPICS))
    benchmark_limit: int = Field(default=12, ge=0, le=40)
    write_merged: bool = True


@app.get("/api/distill/teachers")
def distill_teachers_list():
    data_dir = _ensure_data_dir()
    return {
        "topics": list(DISTILL_TOPICS),
        "teachers": TEACHER_REGISTRY,
        "target_teachers": {
            "coding": "tjarvis91/Q-Coder-50M-Sovereign",
            "math": "nkthebass/tinybrainbot-100m-v3-math",
            "physics": "AlexWortega/moe100m-physics-tinybpe",
        },
        "catalog": distill_catalog(data_dir),
        "merged_file": merged_distill_filename(),
        "per_topic_files": {t: distill_jsonl_filename(t) for t in DISTILL_TOPICS},
        "dir": os.path.abspath(data_dir),
    }


@app.post("/api/distill/generate")
def distill_generate(body: DistillGenerateIn):
    if S.running:
        raise HTTPException(409, "Stop training before generating distill corpus")
    chosen = [t for t in body.topics if t in DISTILL_TOPICS]
    if not chosen:
        raise HTTPException(400, f"No valid topics — choose from {list(DISTILL_TOPICS)}")
    try:
        result = generate_distill_corpus(
            chosen,
            _ensure_data_dir(),
            hf_token=HF_TOKEN or None,
            benchmark_limit=body.benchmark_limit,
            write_merged=body.write_merged,
        )
    except Exception as exc:
        raise HTTPException(500, f"Distill generation failed: {exc}") from exc
    S.add_log(
        "[GREEN] Distill corpus: "
        + ", ".join(f"{r['topic']} ({r.get('kept', 0)} rows · {r.get('teacher', '?')})" for r in result.get("topics", [])),
        "green",
    )
    return {**result, "data_dir": os.path.abspath(_ensure_data_dir())}



@app.get("/api/sft/topics")
def sft_topics_list():
    data_dir = _ensure_data_dir()
    return {
        "topics": topic_catalog(data_dir),
        "available": list(SFT_TOPICS),
        "dir": os.path.abspath(data_dir),
    }


@app.post("/api/sft/generate")
def sft_topics_generate(body: SftGenerateIn):
    if S.running:
        raise HTTPException(409, "Stop training before regenerating SFT corpus")
    chosen = [t for t in body.topics if t in SFT_TOPICS]
    if not chosen:
        raise HTTPException(400, f"No valid topics — choose from {list(SFT_TOPICS)}")
    try:
        result = generate_topics_sft(
            chosen,
            _ensure_data_dir(),
            hf_augment=body.hf_augment,
            hf_limit=body.hf_limit,
            hf_token=HF_TOKEN or None,
            write_merged=body.write_merged,
        )
    except Exception as exc:
        raise HTTPException(500, f"SFT generation failed: {exc}") from exc
    S.add_log(
        f"[GREEN] SFT generated: "
        + ", ".join(f"{b['topic']} ({b['kept']} rows)" for b in result.get("topics", [])),
        "green",
    )
    return {**result, "catalog": topic_catalog(_ensure_data_dir())}


class LoadDataIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@app.get("/api/data")
def list_data_files():
    return {
        "dir": os.path.abspath(_ensure_data_dir()),
        "files": _list_local_data_files(),
        "max_mb": MAX_JSONL_MB,
    }


@app.post("/api/import/data")
async def import_data(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    safe = _safe_jsonl_name(file.filename)
    root = _ensure_data_dir()
    dest = os.path.join(root, safe)
    size = 0
    max_bytes = MAX_JSONL_MB * 1024 * 1024
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"JSONL too large (max {MAX_JSONL_MB} MB)")
                out.write(chunk)
    except HTTPException:
        if os.path.exists(dest):
            os.remove(dest)
        raise
    except Exception as exc:
        if os.path.exists(dest):
            os.remove(dest)
        raise HTTPException(500, f"Upload failed: {exc}") from exc
    rows = _count_jsonl_rows(dest)
    S.add_log(f"[GREEN] uploaded data {safe} · {rows:,} rows · {round(size/(1024**2),2)} MB", "green")
    return {"ok": True, "name": safe, "rows": rows, "size_mb": round(size / (1024**2), 2)}


@app.get("/api/export/checkpoint")
def export_checkpoint():
    eng = _get_engine()
    if eng.step == 0:
        raise HTTPException(404, "No weights in memory yet — wait for a train step")
    local_name = S.checkpoint_name if S.checkpoint_name and S.checkpoint_name.endswith(".pt") else None
    if local_name:
        path = _checkpoint_path(local_name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            return StreamingResponse(
                open(path, "rb"),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={local_name}",
                    "Content-Length": str(size),
                },
            )
    payload = eng.state_bytes()
    filename = _checkpoint_basename(eng)
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(payload)),
        },
    )


@app.get("/api/infer/status")
def infer_status():
    muse = _external_muse_status()
    if muse:
        sidecar = _muse_infer_health() or {}
        infer_ok = bool(sidecar.get("ready") or sidecar.get("ok"))
        if infer_ok:
            err = None
        else:
            err = sidecar.get("error") or (
                "Muse infer sidecar is not ready. Training is untouched; "
                "Ask unlocks when leftover-VRAM generate is loaded (v1 adapter until v2 checkpoint-200)."
            )
        return {
            "ready": True,
            "step": muse["train_step"],
            "loss": muse["last_loss"],
            "params": muse["model_params"],
            "device": sidecar.get("device") or "cuda:0 (training)",
            **muse,
            "inference_available": infer_ok,
            "error": err,
        }
    eng = _get_engine()
    if eng.step == 0 and not S.running:
        _hydrate_from_cos()
    ready = eng.step > 0
    ctx = _infer_context(eng)
    return {
        "ready": ready,
        "step": eng.step,
        "loss": eng.last_loss,
        "params": eng.param_count,
        "device": DEVICE,
        **ctx,
        "error": None if ready else "No checkpoint yet — load a .pt file or train a few steps first",
    }


@app.post("/api/infer/ask")
def infer_ask(body: dict = Body(default_factory=dict)):
    body = body or {}
    q = (body.get("question") or "").strip()
    if not q:
        return JSONResponse({"error": "Type a question first."}, status_code=400)
    muse = _external_muse_status()
    if muse:
        sidecar = _muse_infer_health() or {}
        if sidecar.get("ready") or sidecar.get("ok"):
            out = _muse_infer_ask(q, int(body.get("max_new_tokens") or 128))
            if out.get("error"):
                return JSONResponse(out, status_code=502)
            return {
                "answer": out.get("answer"),
                "step": muse["train_step"],
                "device": out.get("device"),
                "latency_ms": out.get("latency_ms"),
                "adapter": out.get("adapter"),
                "weights_source": out.get("adapter"),
            }
        return JSONResponse(
            {
                "error": sidecar.get("error")
                or (
                    f"Muse-Glimmer is training at step {muse['train_step']:,}. "
                    "Start leftover-VRAM infer sidecar on :8766 to Ask without stopping SFT."
                )
            },
            status_code=409,
        )
    q = wrap_chat_prompt(q)
    eng = _get_engine()
    if eng.step == 0:
        _hydrate_from_cos()
    if eng.step == 0:
        return JSONResponse({"error": "No checkpoint yet"}, status_code=503)
    t0 = _now()
    use_rep = bool(body.get("use_repetition_penalty"))
    rep_penalty = float(body.get("repetition_penalty") or 1.1) if use_rep else 1.0
    rep_penalty = max(1.0, min(rep_penalty, 2.0))
    try:
        answer = eng.generate(
            q,
            max_new_tokens=int(body.get("max_new_tokens") or 120),
            temperature=float(body.get("temperature") if body.get("temperature") is not None else 0.8),
            repetition_penalty=rep_penalty,
        )
        answer = truncate_chat_answer(answer)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    ms = int((_now() - t0) * 1000)
    meta = _infer_meta(eng)
    return {
        "answer": answer,
        "step": eng.step,
        "device": DEVICE,
        "latency_ms": ms,
        **meta,
    }


@app.exception_handler(HTTPException)
def http_exc(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

