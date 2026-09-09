"""Checkpoint eval orchestration — RM panel history, compare, best-checkpoint tracking."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from reward_eval import load_benchmark_prompts, run_reward_panel

AUTO_EVAL_ON_SAVE = os.environ.get("AUTO_EVAL_ON_SAVE", "1").strip().lower() in ("1", "true", "yes")
SAVE_EVAL_LIMIT = int(os.environ.get("SAVE_EVAL_LIMIT", "8"))
COMPARE_EVAL_LIMIT = int(os.environ.get("COMPARE_EVAL_LIMIT", "12"))


def _history_path(data_dir: str) -> str:
    return os.path.join(data_dir, "run-history.jsonl")


def _read_history(data_dir: str) -> list[dict[str, Any]]:
    path = _history_path(data_dir)
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def eval_leaderboard(data_dir: str, *, limit: int = 40) -> dict[str, Any]:
    """Latest RM mean per checkpoint, sorted best-first."""
    by_ckpt: dict[str, dict[str, Any]] = {}
    for row in _read_history(data_dir):
        if row.get("event") != "reward_eval":
            continue
        ckpt = str(row.get("checkpoint") or "").strip()
        if not ckpt:
            continue
        prev = by_ckpt.get(ckpt)
        ts = str(row.get("ts") or "")
        if prev and str(prev.get("ts") or "") > ts:
            continue
        by_ckpt[ckpt] = {
            "checkpoint": ckpt,
            "step": row.get("step"),
            "mean_reward": row.get("reward_mean_post"),
            "delta": row.get("reward_delta"),
            "baseline": row.get("reward_mean_baseline"),
            "n_prompts": row.get("reward_n"),
            "verdict": row.get("verdict"),
            "lesson": row.get("lesson"),
            "agent": row.get("agent"),
            "ts": ts,
            "live": bool(row.get("live_eval")),
        }
    ranked = sorted(
        by_ckpt.values(),
        key=lambda r: (r.get("mean_reward") is not None, float(r.get("mean_reward") or -999)),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    return {
        "best": best,
        "checkpoints": ranked[:limit],
        "evaluated_count": len(ranked),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def pick_best_checkpoint(data_dir: str) -> str | None:
    board = eval_leaderboard(data_dir, limit=1)
    best = board.get("best")
    if not best:
        return None
    return str(best.get("checkpoint") or "") or None


def build_panel_result(
    *,
    eng,
    checkpoint_name: str,
    generate,
    score,
    limit: int,
    live: bool = False,
) -> dict[str, Any]:
    prompts = load_benchmark_prompts(limit=limit)
    panel = run_reward_panel(
        prompts,
        generate=generate,
        score=score,
        checkpoint=checkpoint_name,
        step=int(getattr(eng, "step", 0) or 0),
    )
    panel["live_eval"] = live
    return panel
