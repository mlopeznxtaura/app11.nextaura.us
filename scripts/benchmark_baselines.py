#!/usr/bin/env python3
"""Floor benchmark: StorySupra-10M, Supra-50M-Instruct, optional NextAura .pt checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS_PATH = ROOT / "data" / "benchmark-prompts.jsonl"
OUT_DIR = ROOT / "data"

HF_MODELS = (
    ("SupraLabs/StorySupra-10M", "storysupra-10m"),
    ("SupraLabs/Supra-50M-Instruct", "supra-50m-instruct"),
)


def load_prompts() -> list[dict]:
    rows = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gen_hf(repo_id: str, prompts: list[dict], *, token: str | None, max_new_tokens: int) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    print(f"[bench] loading HF {repo_id}...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, token=token, trust_remote_code=True)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        if "TokenizersBackend" in str(exc) or "does not exist" in str(exc):
            tok = PreTrainedTokenizerFast.from_pretrained(repo_id, token=token)
        else:
            raise
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except Exception:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            repo_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    out: list[dict] = []
    for row in prompts:
        prompt = row["prompt"]
        t0 = time.perf_counter()
        inputs = tok(prompt, return_tensors="pt").to(device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(ids[0], skip_special_tokens=True)
        answer = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
        ms = int((time.perf_counter() - t0) * 1000)
        out.append({**row, "answer": answer, "latency_ms": ms})
        print(f"  {row['id']} {ms}ms", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def gen_nextaura(ckpt_path: Path, prompts: list[dict], *, max_new_tokens: int) -> list[dict]:
    import torch

    from train_engine import TrainEngine

    print(f"[bench] loading NextAura {ckpt_path.name}...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = TrainEngine(device=device)
    eng.load_state_bytes(ckpt_path.read_bytes())
    out: list[dict] = []
    for row in prompts:
        prompt = row["prompt"]
        t0 = time.perf_counter()
        try:
            answer = eng.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)
        except Exception as exc:
            answer = f"[error: {exc}]"
        ms = int((time.perf_counter() - t0) * 1000)
        out.append({**row, "answer": answer, "latency_ms": ms})
        print(f"  {row['id']} {ms}ms", flush=True)
    del eng
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def write_results(model_key: str, rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT_DIR / f"benchmark-{model_key}-{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"model": model_key, **row}, ensure_ascii=False) + "\n")
    print(f"[bench] wrote {path}", flush=True)
    return path


def summarize(path: Path) -> dict:
    cats: dict[str, int] = {}
    empty = 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            cat = row.get("category", "?")
            cats[cat] = cats.get(cat, 0) + 1
            ans = (row.get("answer") or "").strip()
            if len(ans) < 8:
                empty += 1
    return {"path": str(path), "total": total, "categories": cats, "short_or_empty": empty}


def main() -> int:
    parser = argparse.ArgumentParser(description="Floor benchmark vs public small LMs")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--our-checkpoint", default="checkpoints/nextaura-50m-step3898.pt")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip() or None
    prompts = load_prompts()
    print(f"[bench] {len(prompts)} prompts", flush=True)

    summary: list[dict] = []
    if not args.skip_hf:
        for repo_id, key in HF_MODELS:
            try:
                rows = gen_hf(repo_id, prompts, token=token, max_new_tokens=args.max_new_tokens)
                path = write_results(key, rows)
                summary.append(summarize(path))
            except Exception as exc:
                print(f"[bench] FAILED {repo_id}: {exc}", flush=True)
                summary.append({"model": key, "error": str(exc)})

    if not args.skip_ours:
        ckpt = ROOT / args.our_checkpoint
        if ckpt.is_file():
            try:
                rows = gen_nextaura(ckpt, prompts, max_new_tokens=args.max_new_tokens)
                path = write_results("nextaura-50m-step3898", rows)
                summary.append(summarize(path))
            except Exception as exc:
                print(f"[bench] FAILED our ckpt: {exc}", flush=True)
                summary.append({"model": "nextaura", "error": str(exc)})
        else:
            print(f"[bench] missing {ckpt}", flush=True)

    report = OUT_DIR / "benchmark-latest-summary.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[bench] summary {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
