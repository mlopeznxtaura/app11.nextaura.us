#!/usr/bin/env python3
"""Ingest intelligence seed JSONL (one object per line) into data/."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(ROOT, "data", "intelligence-seed-corpus.jsonl")
ML_DEST = os.path.join(ROOT, "data", "intelligence-seed-corpus-ml.jsonl")

_DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("tokenization", ("tokenizer", "bpe", "sentencepiece", "wordpiece", "subword", "byte-level", "vocab")),
    ("vision_language", ("llava", "clip", "blip", "flamingo", "multimodal", "image-text", "vq-vae")),
    ("vjepa", ("v-jepa", "v-jema", "jepa", "masked autoencoder", "mae")),
    ("training_systems", ("ddp", "fsdp", "zero", "distributed", "tensor parallelism", "pipeline parallelism")),
    ("optimization", ("adamw", "learning rate", "warmup", "cosine", "gradient", "optimizer", "weight decay")),
    ("attention", ("flashattention", "attention", "rope", "positional", "mqa", "gqa", "kv cache")),
    ("architecture", ("transformer", "ffn", "layernorm", "rmsnorm", "swiglu", "residual")),
    ("scaling", ("chinchilla", "scaling law", "compute-optimal", "tokens per param")),
    ("inference", ("quantization", "speculative decoding", "kv cache", "inference")),
    ("data_pipeline", ("deduplication", "minhash", "corpus", "jsonl", "webdataset")),
]


def _infer_domain(text: str) -> str:
    lower = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(k in lower for k in keywords):
            return domain
    return "ml_engineering"


def _normalize_seed(row: dict, *, next_id: int) -> dict | None:
    text = str(row.get("text") or "").strip()
    if len(text) < 80:
        return None
    seed_id = row.get("id")
    if not isinstance(seed_id, int):
        seed_id = next_id
    return {
        "id": int(seed_id),
        "tier": row.get("tier", 2),
        "domain": row.get("domain") or _infer_domain(text),
        "token_class": row.get("token_class") or "technical_note",
        "text": text,
    }


def ingest(src: str, dest: str, *, id_offset: int = 1) -> dict[str, int]:
    rows: list[dict] = []
    next_id = id_offset
    with open(src, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            normalized = _normalize_seed(row, next_id=next_id)
            if normalized:
                rows.append(normalized)
                if row.get("id") is None:
                    next_id = int(normalized["id"]) + 1
                else:
                    next_id = max(next_id, int(normalized["id"]) + 1)

    rows.sort(key=lambda r: int(r["id"]))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"input": len(rows), "kept": len(rows), "skipped": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="?", default="-", help="Source JSONL or - for stdin")
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--id-offset", type=int, default=1)
    parser.add_argument("--ml", action="store_true", help="Write to intelligence-seed-corpus-ml.jsonl with ids from 1001")
    args = parser.parse_args()
    if args.ml:
        args.dest = ML_DEST
        if args.id_offset == 1:
            args.id_offset = 1001

    if args.src == "-":
        tmp = os.path.join(ROOT, "data", "_seed_paste.tmp")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as out:
            out.write(sys.stdin.read())
        src = tmp
    else:
        src = args.src

    stats = ingest(src, args.dest, id_offset=args.id_offset)
    print(f"Wrote {stats['kept']} seeds → {args.dest}")


if __name__ == "__main__":
    main()
