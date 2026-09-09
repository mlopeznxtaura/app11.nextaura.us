"""Start preflight — enforce run-history lessons before training."""

from __future__ import annotations

import os
from typing import Any

SFT_SOURCES = frozenset(
    {
        "sft-intelligence",
        "sft-coding",
        "sft-physics",
        "sft-math",
    }
)
DISTILL_SOURCES = frozenset(
    {
        "distill-merged",
        "distill-coding",
        "distill-physics",
        "distill-math",
    }
)
JSONL_LIKE = frozenset({"jsonl", "local-corpus-mix", *SFT_SOURCES, *DISTILL_SOURCES})
STREAM_SOURCES = frozenset(
    {
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
    }
)
FINETUNE_STREAM_SOURCES = frozenset({"nemotron-swe"})
RECOMMENDED_PRETRAIN_BASE = "nextaura-51m-step3898.pt"
HEALTHY_PRETRAIN_LOSS_MAX = 4.5
HEALTHY_SFT_BASE_LOSS_MAX = 3.5
MIN_SFT_ROWS = 500
MIN_RM_PANEL = 20


def _jsonl_path(data_dir: str, filename: str) -> str:
    return os.path.join(data_dir, os.path.basename(filename))


def _file_row_count(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _file_size_bytes(path: str) -> int:
    return os.path.getsize(path) if os.path.isfile(path) else 0


def run_preflight(
    body: Any,
    *,
    data_dir: str,
    status: dict[str, Any],
    loaded_local: str,
    train_step: int,
    last_loss: float,
    running: bool,
    session_status: str,
) -> dict[str, Any]:
    """Return {ok, errors[], warnings[], hints[]}. errors block Start."""
    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []

    if running:
        errors.append("Training is already running.")
        return {"ok": False, "errors": errors, "warnings": warnings, "hints": hints}

    if session_status == "complete" and ds not in FINETUNE_STREAM_SOURCES:
        errors.append(
            "Last run is complete — Reset session or change recipe before Start "
            f"(stop: {status.get('stop_reason_label') or 'see run goal'})."
        )

    ds = str(getattr(body, "data_source", "") or "")
    batch = int(getattr(body, "batch_size", 1024) or 1024)
    target_gb = float(getattr(body, "target_gb", 0.08) or 0.08)
    target_bytes = int(target_gb * (1024**3))
    epoch_cycles = bool(getattr(body, "epoch_cycles", False))
    epochs = int(getattr(body, "epochs", 1) or 1)
    total_steps = int(getattr(body, "total_train_steps", 0) or 0)
    jsonl_file = str(getattr(body, "jsonl_file", "") or "")
    mix_local = bool(getattr(body, "mix_local_jsonl", False))
    lr_schedule = str(getattr(body, "lr_schedule", "cosine") or "cosine")

    jsonl_rows = 0
    jsonl_bytes = 0
    if ds == "distill-live":
        if os.environ.get("DISTILL_LIVE_ENABLED", "0").strip().lower() not in ("1", "true", "yes"):
            errors.append(
                "distill-live is dormant on this host — set DISTILL_LIVE_ENABLED=1 to enable (master-agent also off)"
            )
        else:
            hints.append(
                "distill-live: teachers generate training rows on-the-fly until byte target; "
                f"append log distill-live.jsonl. HF_TOKEN required."
            )
    if ds in JSONL_LIKE or (ds == "mixed" and mix_local):
        fname = jsonl_file
        if ds == "local-corpus-mix":
            jsonl_rows = int(status.get("jsonl_pool_rows") or 0)
        elif ds in SFT_SOURCES | DISTILL_SOURCES:
            topic_file = {
                "sft-coding": "sft-coding.jsonl",
                "sft-physics": "sft-physics.jsonl",
                "sft-math": "sft-math.jsonl",
                "sft-intelligence": "sft-full-merged.jsonl",
                "distill-merged": "distill-merged.jsonl",
                "distill-coding": "distill-coding.jsonl",
                "distill-physics": "distill-physics.jsonl",
                "distill-math": "distill-math.jsonl",
            }.get(ds, jsonl_file)
            path = _jsonl_path(data_dir, topic_file)
            jsonl_rows = _file_row_count(path)
            jsonl_bytes = _file_size_bytes(path)
        else:
            path = _jsonl_path(data_dir, fname)
            jsonl_rows = _file_row_count(path)
            jsonl_bytes = _file_size_bytes(path)

        if jsonl_rows == 0:
            errors.append(f"Local JSONL has no rows ({fname or ds}) — generate or upload data first.")
        elif jsonl_rows < batch:
            errors.append(
                f"jsonl_rows ({jsonl_rows}) < batch_size ({batch}) — shrink batch to ≤{jsonl_rows} "
                "or use a larger corpus."
            )

        if jsonl_bytes > 0 and target_bytes > jsonl_bytes * 10:
            warnings.append(
                f"Byte target {target_gb:.3f} GB is >10× local file size "
                f"({jsonl_bytes / (1024**2):.2f} MB) — run will stop when data is exhausted, not at target GB."
            )

        if epoch_cycles and epochs > 2 and jsonl_rows < 200:
            warnings.append(
                f"epoch_cycles ×{epochs} on only {jsonl_rows} rows — high overfit risk; history marks these as discard."
            )

    if ds == "nemotron-swe":
        hints.append(
            "nemotron-swe: streams nvidia/Nemotron-SFT-SWE-v3.5 (~5.1k OpenCode agent trajectories). "
            "Load a pretrain base first; use Fine-tuning mode and constant LR."
        )

    if ds in SFT_SOURCES | DISTILL_SOURCES | FINETUNE_STREAM_SOURCES:
        if jsonl_rows > 0 and jsonl_rows < MIN_SFT_ROWS:
            warnings.append(
                f"SFT/distill with {jsonl_rows} rows is below {MIN_SFT_ROWS} — treat as smoke only, not a real finetune."
            )
        if train_step > 0 and last_loss > HEALTHY_SFT_BASE_LOSS_MAX:
            warnings.append(
                f"Loaded loss {last_loss:.2f} is above healthy SFT base (<{HEALTHY_SFT_BASE_LOSS_MAX}) — "
                f"reload pretrain base ({RECOMMENDED_PRETRAIN_BASE}) before finetuning."
            )
        if loaded_local and RECOMMENDED_PRETRAIN_BASE not in loaded_local and "51m" not in loaded_local:
            if "8m" in loaded_local or train_step < 500:
                warnings.append(
                    f"SFT/distill should start from a pretrain base (e.g. {RECOMMENDED_PRETRAIN_BASE}), "
                    f"not {loaded_local or 'current weights'}."
                )

    if ds in STREAM_SOURCES and ds not in FINETUNE_STREAM_SOURCES and total_steps > 0 and target_gb >= 0.05:
        warnings.append(
            "Stream pretrain with total_train_steps > 0 — set total_train_steps=0 and stop on Target GB only."
        )

    if lr_schedule == "cosine" and total_steps > 0 and ds in JSONL_LIKE:
        warnings.append(
            f"Cosine schedule over {total_steps} steps on local JSONL — LR may hit zero before byte target; "
            "prefer constant LR for SFT or match steps to epoch budget."
        )

    goal_title = str(getattr(body, "run_goal_title", "") or "").lower()
    if ("smoke" in goal_title or "50m" in goal_title or "multimodal" in goal_title) and ds in JSONL_LIKE:
        warnings.append(
            "Run goal mentions HF smoke/scale but data_source is local JSONL — switch to multimodal/mixed/fineweb."
        )

    if ds == "jsonl" and mix_local and goal_title:
        pass

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "jsonl_rows": jsonl_rows,
        "jsonl_bytes": jsonl_bytes,
        "recommended_pretrain_base": RECOMMENDED_PRETRAIN_BASE,
        "min_rm_panel": MIN_RM_PANEL,
    }
