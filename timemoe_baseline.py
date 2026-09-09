"""Maple728/TimeMoE-50M — OSS 50M benchmark reference for progress tracking.

Paper: https://huggingface.co/papers/2409.16040
Model:  https://huggingface.co/Maple728/TimeMoE-50M
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TIMEMOE_MODEL_ID = "Maple728/TimeMoE-50M"
TIMEMOE_HF_URL = "https://huggingface.co/Maple728/TimeMoE-50M"
DEFAULT_CONTEXT_LEN = 12
DEFAULT_HORIZON = 6
BENCHMARK_WINDOWS_PATH = Path(__file__).resolve().parent / "data" / "timemoe-benchmark-windows.jsonl"
BASELINE_STATE_PATH = Path(__file__).resolve().parent / "data" / "timemoe-baseline.json"


def _default_windows() -> list[dict[str, Any]]:
    """Reproducible synthetic windows (context → held-out future)."""
    import random

    rng = random.Random(728)
    rows: list[dict[str, Any]] = []
    for i in range(24):
        ctx = [rng.uniform(-2, 2) for _ in range(DEFAULT_CONTEXT_LEN)]
        drift = rng.uniform(-0.15, 0.15)
        future = [ctx[-1] + drift * (j + 1) + rng.gauss(0, 0.05) for j in range(DEFAULT_HORIZON)]
        rows.append(
            {
                "id": f"ts-{i + 1:02d}",
                "category": "synthetic",
                "context": ctx,
                "future": future,
                "horizon": DEFAULT_HORIZON,
            }
        )
    return rows


def load_benchmark_windows(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else BENCHMARK_WINDOWS_PATH
    if not p.is_file():
        return _default_windows()
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows or _default_windows()


def _normalize_context(seq: list[float]) -> tuple[list[float], float, float]:
    import torch

    t = torch.tensor(seq, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std().item()) or 1.0
    normed = ((t - mean) / std).tolist()
    return normed, mean, std


def _timemoe_rollout(model, context, horizon: int):
    import torch

    seq = context
    if seq.dim() == 2:
        seq = seq.unsqueeze(-1)
    for _ in range(horizon):
        out = model(seq)
        logits = out.logits
        if logits is None:
            raise RuntimeError("TimeMoE forward returned no logits")
        next_step = logits[:, -1:, :]
        seq = torch.cat([seq, next_step], dim=1)
    return seq


def run_timemoe_benchmark(
    *,
    model_id: str = TIMEMOE_MODEL_ID,
    windows: list[dict[str, Any]] | None = None,
    device: str | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    windows = windows or load_benchmark_windows()
    if not windows:
        return {"ok": False, "error": "no benchmark windows"}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction

        model = TimeMoeForPrediction.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
    model.to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    samples: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in windows:
            ctx = [float(x) for x in row["context"]]
            future = [float(x) for x in row["future"]]
            horizon = int(row.get("horizon") or len(future) or DEFAULT_HORIZON)
            normed, mean, std = _normalize_context(ctx)
            inp = torch.tensor([normed], dtype=model_dtype, device=device).unsqueeze(-1)
            out = _timemoe_rollout(model, inp, horizon)
            preds = out[:, -horizon:, 0].float().cpu().tolist()[0]
            preds_denorm = [p * std + mean for p in preds]
            err2 = sum((p - t) ** 2 for p, t in zip(preds_denorm, future, strict=False)) / max(
                len(future), 1
            )
            err1 = sum(abs(p - t) for p, t in zip(preds_denorm, future, strict=False)) / max(
                len(future), 1
            )
            mse_sum += err2
            mae_sum += err1
            n += 1
            samples.append(
                {
                    "id": row.get("id"),
                    "mse": round(err2, 6),
                    "mae": round(err1, 6),
                }
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mse = mse_sum / max(n, 1)
    mae = mae_sum / max(n, 1)
    return {
        "ok": True,
        "model_id": model_id,
        "model_url": TIMEMOE_HF_URL,
        "modality": "time_series",
        "params_class": "50M",
        "n_windows": n,
        "context_length": DEFAULT_CONTEXT_LEN,
        "horizon": DEFAULT_HORIZON,
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "samples": samples[:8],
        "device": device,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_baseline_state(result: dict[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path) if path else BASELINE_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "TimeMoE-50M",
        "reference": result,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_baseline_state(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path) if path else BASELINE_STATE_PATH
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def benchmark_summary_for_api(data_dir: str | None = None) -> dict[str, Any]:
    if data_dir:
        state_path = Path(data_dir) / "timemoe-baseline.json"
    else:
        state_path = BASELINE_STATE_PATH
    state = load_baseline_state(state_path)
    ref = (state or {}).get("reference") or {}
    return {
        "benchmark_model": TIMEMOE_MODEL_ID,
        "benchmark_url": TIMEMOE_HF_URL,
        "benchmark_modality": "time_series",
        "params_class": "50M",
        "mse": ref.get("mse"),
        "mae": ref.get("mae"),
        "n_windows": ref.get("n_windows"),
        "updated_at": ref.get("updated_at") or (state or {}).get("updated_at"),
        "configured": bool(ref.get("mse") is not None),
    }


def compare_to_timemoe(
    our_metric: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach progress context vs TimeMoE (different modalities — label clearly)."""
    baseline = baseline or (load_baseline_state() or {}).get("reference") or {}
    out = dict(our_metric)
    out["benchmark_ref"] = TIMEMOE_MODEL_ID
    out["benchmark_url"] = TIMEMOE_HF_URL
    if our_metric.get("modality") == "time_series" and baseline.get("mse") is not None:
        our_mse = float(our_metric.get("mse") or math.inf)
        base_mse = float(baseline["mse"])
        out["vs_timemoe_mse_delta"] = round(base_mse - our_mse, 6)
        out["vs_timemoe_better"] = our_mse < base_mse
    elif our_metric.get("mean_reward") is not None:
        out["benchmark_timemoe_mse"] = baseline.get("mse")
        out["benchmark_timemoe_mae"] = baseline.get("mae")
        out["note"] = (
            "Text RM vs TimeMoE TS MSE — different tasks; TimeMoE is the 50M OSS reference bar."
        )
    return out
