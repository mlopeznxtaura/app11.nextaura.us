"""Persist and serve model run history (pretrain / SFT / smoke) for the UI."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

RUN_HISTORY_FILE = os.environ.get("RUN_HISTORY_FILE", "data/run-history.jsonl")
STEP_RE = re.compile(r"step(\d+)", re.I)

# Curated lessons from real sessions — shown even if .pt was deleted.
SEED_RUNS: tuple[dict[str, Any], ...] = (
    {
        "id": "seed-3898",
        "ts": "2026-08-25T18:39:00+00:00",
        "event": "complete",
        "phase": "pretrain",
        "checkpoint": "nextaura-50m-step3898.pt",
        "step": 3898,
        "loss": 3.3583,
        "model_params": 51_082_752,
        "n_embd": 512,
        "vocab_size": 50257,
        "data_source": "mixed",
        "target_gb": 3.33,
        "verdict": "keep",
        "lesson": "1× Chinchilla pretrain base — use for SFT, not chat",
    },
    {
        "id": "seed-4191-fail",
        "ts": "2026-08-26T00:16:00+00:00",
        "event": "complete",
        "phase": "sft",
        "checkpoint": "nextaura-50m-step4191.pt",
        "step": 4191,
        "base_step": 3898,
        "loss": 4.4912,
        "model_params": 51_082_752,
        "data_source": "sft-intelligence",
        "target_gb": 0.25,
        "jsonl_epochs": 207,
        "verdict": "discard",
        "lesson": "256 MB on 954 rows = 207 epochs — GB hell, rolled back",
    },
    {
        "id": "seed-3915",
        "ts": "2026-08-26T06:00:00+00:00",
        "event": "complete",
        "phase": "sft",
        "agent": "ox-alpha",
        "checkpoint": "nextaura-50m-step3915.pt",
        "step": 3915,
        "base_step": 3898,
        "loss": 5.5754,
        "model_params": 51_082_752,
        "data_source": "sft-intelligence",
        "target_gb": 0.17,
        "jsonl_epochs": 3,
        "reward_mean_post": -4.107,
        "reward_mean_baseline": -4.177,
        "reward_delta": 0.07,
        "reward_wins": 2,
        "reward_n": 5,
        "verdict": "noise",
        "lesson": "17 updates @ batch 256 — pipeline OK, corpus too small (RM +0.07 noise)",
    },
    {
        "id": "seed-v2-corpus",
        "ts": "2026-08-26T11:30:00+00:00",
        "event": "corpus",
        "phase": "data",
        "agent": "ox-alpha",
        "jsonl_file": "nextaura-sft-v2.jsonl",
        "jsonl_rows": 7350,
        "verdict": "note",
        "lesson": "v2 corpus: ~94% ML-engineering docs; chat diversity only 1.3%",
    },
    {
        "id": "seed-3943-v2",
        "ts": "2026-08-26T12:00:00+00:00",
        "event": "complete",
        "phase": "sft",
        "agent": "ox-alpha",
        "checkpoint": "nextaura-50m-step3943.pt",
        "step": 3943,
        "base_step": 3898,
        "loss": 4.39,
        "loss_start": 5.06,
        "model_params": 51_082_752,
        "data_source": "jsonl",
        "jsonl_file": "nextaura-sft-v2.jsonl",
        "jsonl_rows": 7350,
        "target_gb": 0.17,
        "jsonl_epochs": 3,
        "train_steps": 57,
        "batch_size": 128,
        "learning_rate": 3e-5,
        "reward_mean_baseline": -3.913,
        "reward_mean_post": -4.474,
        "reward_delta": -0.561,
        "reward_wins": 3,
        "reward_n": 20,
        "verdict": "discard",
        "lesson": "Loss ↓ 5.06→4.39 but RM ↓ — templates/register not Q&A (3/20 wins)",
    },
    {
        "id": "seed-v3-corpus",
        "ts": "2026-08-26T12:45:00+00:00",
        "event": "corpus",
        "phase": "data",
        "agent": "ox-alpha",
        "jsonl_file": "nextaura-sft-v3.jsonl",
        "jsonl_rows": 666,
        "verdict": "note",
        "lesson": "v3 corpus: 666 rows, 32% direct-answer + verified math (rebalance after v2)",
    },
    {
        "id": "seed-3904-v3",
        "ts": "2026-08-26T13:00:00+00:00",
        "event": "complete",
        "phase": "sft",
        "agent": "ox-alpha",
        "checkpoint": "nextaura-50m-step3904.pt",
        "step": 3904,
        "base_step": 3898,
        "loss": 5.1521,
        "model_params": 51_082_752,
        "data_source": "jsonl",
        "jsonl_file": "nextaura-sft-v3.jsonl",
        "jsonl_rows": 666,
        "target_gb": 0.17,
        "jsonl_epochs": 3,
        "train_steps": 6,
        "batch_size": 128,
        "verdict": "discard",
        "lesson": "Only ~6 optimizer steps — batch 128 on 666 rows; epoch_cycles finished too early",
    },
    {
        "id": "seed-108-smoke",
        "ts": "2026-08-26T00:45:00+00:00",
        "event": "complete",
        "phase": "smoke",
        "checkpoint": "nextaura-50m-step108.pt",
        "step": 108,
        "loss": 5.9935,
        "model_params": 8_481_280,
        "n_embd": 256,
        "vocab_size": 8192,
        "data_source": "mixed",
        "target_gb": 0.08,
        "verdict": "floor",
        "lesson": "10M StorySupra shape floor smoke — compare vs StorySupra benchmark",
    },
    {
        "id": "seed-40103",
        "ts": "2026-08-25T14:50:00+00:00",
        "event": "note",
        "phase": "pretrain",
        "checkpoint": "nextaura-50m-step40103.pt",
        "step": 40103,
        "loss": 0.48,
        "model_params": 51_082_752,
        "verdict": "discard",
        "lesson": "Loss <1 on mixed = overfit/memorization — never use",
    },
    {
        "id": "seed-176",
        "ts": "2026-08-25T21:17:00+00:00",
        "event": "complete",
        "phase": "sft",
        "checkpoint": "nextaura-50m-step176.pt",
        "step": 176,
        "loss": 5.178,
        "model_params": 17_200_192,
        "n_embd": 192,
        "data_source": "sft-intelligence",
        "verdict": "discard",
        "lesson": "17M seed-only SFT from scratch — failed architecture",
    },
)


def _history_path(data_dir: str) -> str:
    root = os.path.abspath(data_dir)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "run-history.jsonl")


def _infer_phase(
    *,
    data_source: str,
    model_params: int,
    n_embd: int,
    vocab_size: int,
    step: int,
) -> str:
    if data_source in ("jsonl", "sft-intelligence", "local-corpus-mix"):
        return "sft"
    if data_source == "vjepa":
        return "mcf" if model_params >= 200_000_000 else "smoke"
    if data_source == "multimodal":
        return "pretrain"
    if model_params < 15_000_000 or n_embd <= 256 or vocab_size <= 8192:
        return "smoke"
    if step >= 3000 and model_params > 40_000_000:
        return "pretrain"
    return "pretrain"


def _infer_verdict(
    phase: str,
    loss: float,
    *,
    jsonl_epochs: int = 0,
    target_gb: float = 0,
    reward_delta: float | None = None,
    train_steps: int = 0,
) -> str:
    if reward_delta is not None and reward_delta < -0.15:
        return "discard"
    if loss > 0 and loss < 1.0 and phase == "pretrain":
        return "discard"
    if phase == "sft" and jsonl_epochs > 10 and target_gb > 0.1:
        return "discard"
    if phase == "sft" and train_steps > 0 and train_steps < 10:
        return "discard"
    if phase == "sft" and loss > 5.0:
        return "noise"
    if phase == "pretrain" and 2.5 <= loss <= 4.5:
        return "keep"
    if phase == "smoke":
        return "floor"
    if phase == "sft":
        if reward_delta is not None and reward_delta > 0.15:
            return "keep"
        return "noise"
    return "ok"


def _lesson_for(verdict: str, phase: str, row: dict[str, Any]) -> str:
    if row.get("lesson"):
        return str(row["lesson"])
    if verdict == "discard" and phase == "pretrain":
        return "Suspiciously low loss — likely overfit"
    if verdict == "discard" and phase == "sft":
        return "Too many epochs or wrong base — roll back to 3898"
    if verdict == "keep":
        return "Clean pretrain base"
    if verdict == "floor":
        return "Floor experiment — benchmark before scaling"
    if verdict == "noise":
        return "SFT ran but corpus/steps too small for measurable gain"
    return ""


def _row_key(row: dict[str, Any]) -> str:
    ck = row.get("checkpoint")
    if ck:
        return str(ck)
    jid = row.get("id")
    if jid:
        return str(jid)
    jf = row.get("jsonl_file")
    if jf and row.get("event") == "corpus":
        return f"corpus:{jf}"
    return str(row.get("ts") or id(row))


def _sort_ts(row: dict[str, Any]) -> str:
    return str(row.get("ts") or "")


def has_run_record(data_dir: str, checkpoint: str) -> bool:
    for row in _read_jsonl(_history_path(data_dir)):
        if row.get("checkpoint") == checkpoint and row.get("event") in (
            "complete",
            "stopped",
            "interrupted",
            "import",
            "backfill",
        ):
            return True
    return False


def read_checkpoint_meta(path: str) -> dict[str, Any]:
    """Read step/loss/arch/session from a .pt without loading onto GPU."""
    try:
        import torch
        from model import ModelConfig, estimate_params

        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {"checkpoint": os.path.basename(path)}
    if not isinstance(data, dict):
        return {"checkpoint": os.path.basename(path)}
    cfg_dict = data.get("model_config") or {}
    cfg = ModelConfig.from_dict(cfg_dict) if cfg_dict else None
    session = dict(data.get("session_meta") or {})
    step = int(data.get("step") or 0)
    run_start = int(data.get("run_start_step") or 0)
    meta: dict[str, Any] = {
        "checkpoint": os.path.basename(path),
        "step": step,
        "run_start_step": run_start,
        "base_step": run_start if step > run_start else None,
        "loss": float(data.get("last_loss") or 0.0) or None,
        "learning_rate": float(data.get("learning_rate") or 0.0) or None,
        "batch_size": int(data.get("batch_size") or 0) or None,
        "model_params": estimate_params(cfg) if cfg else None,
        "n_embd": cfg.n_embd if cfg else None,
        "n_layer": cfg.n_layer if cfg else None,
        "vocab_size": cfg.vocab_size if cfg else None,
        "train_steps": max(0, step - run_start) if run_start else None,
        **session,
    }
    return {k: v for k, v in meta.items() if v is not None}


def backfill_disk_checkpoints(data_dir: str, checkpoint_dir: str) -> list[dict[str, Any]]:
    """Auto-append history rows for on-disk .pt files not yet logged."""
    if not os.path.isdir(checkpoint_dir):
        return []
    added: list[dict[str, Any]] = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if not name.endswith(".pt"):
            continue
        if has_run_record(data_dir, name):
            continue
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            continue
        meta = read_checkpoint_meta(path)
        if not meta.get("step"):
            m = STEP_RE.search(name)
            if m:
                meta["step"] = int(m.group(1))
        row = record_training_finish(
            data_dir,
            status="backfill",
            checkpoint_name=name,
            step=int(meta.get("step") or 0),
            loss=float(meta.get("loss") or 0),
            data_source=str(meta.get("data_source") or ""),
            target_gb=float(meta.get("target_gb") or 0),
            model_params=int(meta.get("model_params") or 0),
            n_embd=int(meta.get("n_embd") or 0),
            vocab_size=int(meta.get("vocab_size") or 0),
            jsonl_epochs=int(meta.get("jsonl_epochs") or meta.get("jsonl_epoch") or 0),
            jsonl_file=str(meta.get("jsonl_file") or ""),
            learning_rate=float(meta.get("learning_rate") or 0),
            base_step=meta.get("base_step"),
            train_steps=int(meta.get("train_steps") or 0),
            auto=True,
        )
        added.append(row)
    return added


def append_run_record(data_dir: str, record: dict[str, Any]) -> dict[str, Any]:
    path = _history_path(data_dir)
    row = dict(record)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    if "id" not in row:
        ck = row.get("checkpoint")
        row["id"] = str(ck or row.get("jsonl_file") or row["ts"])
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def record_engineering(
    data_dir: str,
    *,
    event: str = "note",
    agent: str = "",
    lesson: str = "",
    **fields: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"event": event, "agent": agent or None, "lesson": lesson, **fields}
    row = {k: v for k, v in row.items() if v is not None}
    if "verdict" not in row:
        row["verdict"] = "note" if event in ("corpus", "note", "reward_eval") else "ok"
    return append_run_record(data_dir, row)


def record_reward_eval(
    data_dir: str,
    *,
    checkpoint: str,
    step: int,
    mean_reward: float,
    n_prompts: int,
    agent: str = "agent",
    baseline_mean: float | None = None,
    wins: int | None = None,
    record_path: str = "",
) -> dict[str, Any]:
    delta = None
    if baseline_mean is not None:
        delta = round(mean_reward - baseline_mean, 4)
    row: dict[str, Any] = {
        "event": "reward_eval",
        "phase": "eval",
        "agent": agent,
        "checkpoint": checkpoint,
        "step": step,
        "reward_mean_post": round(mean_reward, 4),
        "reward_n": n_prompts,
        "reward_eval_path": record_path or None,
    }
    if baseline_mean is not None:
        row["reward_mean_baseline"] = round(baseline_mean, 4)
        row["reward_delta"] = delta
    if wins is not None:
        row["reward_wins"] = wins
    if delta is not None:
        if delta < -0.15:
            row["verdict"] = "discard"
            row["lesson"] = row.get("lesson") or f"RM regressed {delta:+.3f} — roll back"
        elif delta > 0.15:
            row["verdict"] = "keep"
            row["lesson"] = row.get("lesson") or f"RM improved {delta:+.3f}"
        else:
            row["verdict"] = "noise"
            row["lesson"] = row.get("lesson") or f"RM noise-level ({delta:+.3f})"
    return append_run_record(data_dir, row)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def record_training_finish(
    data_dir: str,
    *,
    status: str,
    checkpoint_name: str,
    step: int,
    loss: float,
    data_source: str,
    target_gb: float,
    model_params: int,
    n_embd: int,
    vocab_size: int,
    jsonl_epochs: int = 0,
    jsonl_file: str = "",
    learning_rate: float = 0.0,
    base_step: int | None = None,
    agent: str = "",
    train_steps: int = 0,
    run_goal_title: str | None = None,
    run_goal_outcome: str | None = None,
    auto: bool = False,
) -> dict[str, Any]:
    phase = _infer_phase(
        data_source=data_source,
        model_params=model_params,
        n_embd=n_embd,
        vocab_size=vocab_size,
        step=step,
    )
    if train_steps <= 0 and base_step is not None and step > base_step:
        train_steps = step - base_step
    verdict = _infer_verdict(
        phase,
        loss,
        jsonl_epochs=jsonl_epochs,
        target_gb=target_gb,
        train_steps=train_steps,
    )
    row: dict[str, Any] = {
        "event": "complete" if status == "complete" else status,
        "phase": phase,
        "checkpoint": checkpoint_name,
        "step": step,
        "loss": round(loss, 4) if loss else None,
        "data_source": data_source or None,
        "jsonl_file": jsonl_file or None,
        "target_gb": target_gb or None,
        "model_params": model_params or None,
        "n_embd": n_embd or None,
        "vocab_size": vocab_size or None,
        "jsonl_epochs": jsonl_epochs or None,
        "learning_rate": learning_rate or None,
        "train_steps": train_steps or None,
        "agent": agent or None,
        "auto": auto,
        "verdict": verdict,
        "status": status,
    }
    if base_step is not None:
        row["base_step"] = base_step
    if run_goal_title:
        row["run_goal_title"] = run_goal_title
    if run_goal_outcome:
        row["run_goal_outcome"] = run_goal_outcome
    row["lesson"] = _lesson_for(verdict, phase, row)
    if run_goal_title or run_goal_outcome:
        extra = " · ".join(x for x in [run_goal_title, run_goal_outcome] if x)
        if extra:
            row["lesson"] = f"{row['lesson']} ({extra})"
    return append_run_record(data_dir, row)


def run_history_bundle(
    data_dir: str,
    checkpoint_dir: str,
    *,
    loaded: str = "",
    backfill: bool = True,
) -> dict[str, Any]:
    if backfill:
        try:
            backfill_disk_checkpoints(data_dir, checkpoint_dir)
        except Exception:
            pass
    path = _history_path(data_dir)
    logged = _read_jsonl(path)
    by_key: dict[str, dict[str, Any]] = {}

    for seed in SEED_RUNS:
        by_key[_row_key(seed)] = {**seed, "on_disk": False, "loaded": False}

    for row in logged:
        key = _row_key(row)
        prev = by_key.get(key, {})
        if str(prev.get("ts") or "") > str(row.get("ts") or ""):
            merged = {**row, **prev}
        else:
            merged = {**prev, **row}
        merged["on_disk"] = prev.get("on_disk", False)
        by_key[key] = merged

    if os.path.isdir(checkpoint_dir):
        for name in os.listdir(checkpoint_dir):
            if not name.endswith(".pt"):
                continue
            m = STEP_RE.search(name)
            step = int(m.group(1)) if m else 0
            key = name
            entry = by_key.get(key, {})
            entry.setdefault("checkpoint", name)
            entry.setdefault("step", step)
            entry["on_disk"] = True
            entry["size_mb"] = round(os.path.getsize(os.path.join(checkpoint_dir, name)) / (1024**2), 1)
            entry["loaded"] = name == loaded
            if "phase" not in entry or entry.get("phase") == "unknown":
                disk_meta = read_checkpoint_meta(os.path.join(checkpoint_dir, name))
                for k, v in disk_meta.items():
                    if k not in entry or entry[k] in (None, "", "unknown", "ok"):
                        entry[k] = v
                if entry.get("data_source") in ("jsonl", "sft-intelligence") or entry.get("jsonl_file"):
                    entry["phase"] = "sft"
                elif entry.get("model_params") and int(entry.get("model_params") or 0) > 40_000_000:
                    entry["phase"] = "pretrain"
            if "verdict" not in entry:
                entry["verdict"] = "ok"
            by_key[key] = entry

    rows = list(by_key.values())
    for row in rows:
        ck = row.get("checkpoint")
        row["loaded"] = ck == loaded and bool(loaded)

    rows.sort(key=lambda r: (_sort_ts(r), int(r.get("step") or 0)), reverse=True)

    return {
        "rows": rows,
        "count": len(rows),
        "log_path": path,
        "lessons": [
            "Loss 2.5–4.5 on mixed pretrain = healthy base (e.g. step 3898)",
            "Loss <1 on mixed = overfit — discard (e.g. step 40103)",
            "SFT loss ↓ does not mean chat ↑ — always run RM panel (20 prompts)",
            "94% ML-eng templates → register markers, not answers (v2 @ 3943)",
            "Shrink batch when jsonl_rows < batch_size (v3 died at 6 steps)",
            "Reload step 3898 before each SFT experiment",
        ],
    }
