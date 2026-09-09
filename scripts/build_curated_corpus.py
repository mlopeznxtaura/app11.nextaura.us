#!/usr/bin/env python3
"""Build curated local JSONL files for intelligence training."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_jsonl import accept_training_text, load_jsonl_lines, row_training_text, should_skip_local_row
from intelligence_expand import (
    INTELLIGENCE_SEED_FILE,
    INTELLIGENCE_SEED_FILES,
    INTELLIGENCE_SEED_ML_FILE,
    INTELLIGENCE_SFT_FILE,
    SFT_FULL_MERGED_FILE,
    write_intelligence_sft_expanded_all,
)


def _write_curated(rows: list[dict], dest: str) -> dict[str, int]:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    kept = 0
    with open(dest, "w", encoding="utf-8") as out:
        for row in rows:
            text = row_training_text(row)
            if accept_training_text(text):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    return {"input": len(rows), "kept": kept, "skipped": len(rows) - kept}


def build_2400_clean(src: str, dest: str) -> dict[str, int]:
    rows = load_jsonl_lines(src)
    return _write_curated(rows, dest)


def build_4563_curated(src: str, dest: str) -> dict[str, int]:
    rows = load_jsonl_lines(src)
    filtered = [row for row in rows if not should_skip_local_row(row)]
    return _write_curated(filtered, dest)


def build_extract_merged(src_glob: str, dest: str) -> dict[str, int]:
    rows: list[dict] = []
    for path in sorted(glob.glob(src_glob)):
        rows.extend(load_jsonl_lines(path))
    return _write_curated(rows, dest)


def _sft_row_key(row: dict) -> str:
    import hashlib

    text = row_training_text(row)
    if text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _curriculum_to_chat(row: dict) -> dict | None:
    action = str(row.get("action") or "").strip()
    desc = str(row.get("description") or row.get("analysis") or "").strip()
    if not action or len(desc) < 80:
        return None
    user = action if action.endswith("?") else f"Can you explain: {action}?"
    principle = str(row.get("why_this_matters") or row.get("final_statement") or "Explain clearly").strip()
    return {
        "principle": principle,
        "user": user,
        "assistant": desc[:2000],
        "source": "4563://curriculum",
    }


def build_sft_chat_merged(
    extract_glob: str,
    curriculum_src: str,
    identity_src: str,
    dest: str,
) -> dict[str, int]:
    """SFT-only chat corpus: identity seed + deduped extracts + 4563 curriculum Q/A."""
    candidates: list[dict] = []
    if os.path.isfile(identity_src):
        candidates.extend(load_jsonl_lines(identity_src))
    for path in sorted(glob.glob(extract_glob)):
        candidates.extend(load_jsonl_lines(path))
    if os.path.isfile(curriculum_src):
        for row in load_jsonl_lines(curriculum_src):
            if row.get("step") is not None and row.get("action"):
                chat = _curriculum_to_chat(row)
                if chat:
                    candidates.append(chat)

    seen: set[str] = set()
    kept_rows: list[dict] = []
    for row in candidates:
        text = row_training_text(row)
        if not accept_training_text(text):
            continue
        key = _sft_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept_rows.append(row)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as out:
        for row in kept_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"input": len(candidates), "kept": len(kept_rows), "skipped": len(candidates) - len(kept_rows)}


def build_sft_full_merged(
    chat_src: str,
    intelligence_src: str,
    dest: str,
) -> dict[str, int]:
    """Merge chat SFT + expanded intelligence seeds for fine-tuning."""
    import hashlib
    import json

    candidates: list[dict] = []
    for path in (chat_src, intelligence_src):
        if os.path.isfile(path):
            candidates.extend(load_jsonl_lines(path))

    seen: set[str] = set()
    kept_rows: list[dict] = []
    for row in candidates:
        text = row_training_text(row)
        if not accept_training_text(text):
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept_rows.append(row)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as out:
        for row in kept_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"input": len(candidates), "kept": len(kept_rows), "skipped": len(candidates) - len(kept_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", default=os.path.expanduser("~/Downloads"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    args = parser.parse_args()

    jobs = [
        ("2400-clean.jsonl", build_2400_clean, os.path.join(args.downloads, "2400.jsonl")),
        ("4563-curated.jsonl", build_4563_curated, os.path.join(args.downloads, "4563-fixed.jsonl")),
        (
            "nextaura-extract-merged.jsonl",
            build_extract_merged,
            os.path.join(args.downloads, "nextaura-extract*.jsonl"),
        ),
        (
            "sft-chat-merged.jsonl",
            build_sft_chat_merged,
            None,
        ),
        (
            INTELLIGENCE_SFT_FILE,
            write_intelligence_sft_expanded_all,
            None,
        ),
        (
            SFT_FULL_MERGED_FILE,
            build_sft_full_merged,
            None,
        ),
    ]

    print("Curated corpus build")
    for name, fn, src in jobs:
        dest = os.path.join(args.out_dir, name)
        if fn is build_sft_chat_merged:
            stats = fn(
                os.path.join(args.downloads, "nextaura-extract*.jsonl"),
                os.path.join(args.out_dir, "4563-curated.jsonl"),
                os.path.join(args.out_dir, "sft-identity-seed.jsonl"),
                dest,
            )
        elif fn is write_intelligence_sft_expanded_all:
            seed_paths = [
                os.path.join(args.out_dir, seed_name)
                for seed_name in INTELLIGENCE_SEED_FILES
                if os.path.isfile(os.path.join(args.out_dir, seed_name))
            ]
            if not seed_paths:
                print(f"SKIP {name}: missing seed files ({', '.join(INTELLIGENCE_SEED_FILES)})")
                continue
            stats = fn(seed_paths, dest)
        elif fn is build_sft_full_merged:
            chat_path = os.path.join(args.out_dir, "sft-chat-merged.jsonl")
            intel_path = os.path.join(args.out_dir, INTELLIGENCE_SFT_FILE)
            if not os.path.isfile(chat_path) or not os.path.isfile(intel_path):
                print(f"SKIP {name}: need sft-chat-merged.jsonl + {INTELLIGENCE_SFT_FILE}")
                continue
            stats = fn(chat_path, intel_path, dest)
        elif fn is build_extract_merged:
            stats = fn(src, dest)
        else:
            if not os.path.isfile(src):
                print(f"SKIP {name}: missing {src}")
                continue
            stats = fn(src, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024) if os.path.isfile(dest) else 0
        print(f"  {name}: kept {stats['kept']}/{stats['input']} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
