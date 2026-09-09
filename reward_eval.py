"""Batch reward-panel eval for agent self-drive (OpenAssistant RM on fixed prompts)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "data" / "benchmark-prompts.jsonl"


def load_benchmark_prompts(path: str | Path | None = None, *, limit: int = 0) -> list[dict[str, Any]]:
    p = Path(path) if path else DEFAULT_PROMPTS_PATH
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def run_reward_panel(
    prompts: list[dict[str, Any]],
    *,
    generate: Callable[[str], str],
    score: Callable[[str, str], float],
    checkpoint: str = "",
    step: int = 0,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for row in prompts:
        question = str(row.get("prompt") or row.get("question") or "").strip()
        if not question:
            continue
        answer = generate(question)
        reward = score(question, answer)
        results.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "prompt": question[:200],
                "answer": answer[:500],
                "reward": round(reward, 4),
            }
        )
    if not results:
        return {"ok": False, "error": "no prompts scored", "n": 0}
    mean = sum(r["reward"] for r in results) / len(results)
    return {
        "ok": True,
        "checkpoint": checkpoint,
        "step": step,
        "n_prompts": len(results),
        "mean_reward": round(mean, 4),
        "results": results,
        "prompts_path": os.path.basename(str(DEFAULT_PROMPTS_PATH)),
    }
