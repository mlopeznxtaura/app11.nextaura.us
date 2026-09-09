#!/usr/bin/env python3
"""Convert a JSON array file (possibly malformed) to JSONL for Stage 1 training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx] not in "{[":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        idx = end


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "4563-fixed.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else src.with_suffix(".jsonl"))
    text = src.read_text(encoding="utf-8")
    count = 0
    with dst.open("w", encoding="utf-8") as out:
        for row in iter_json_objects(text):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count:,} rows -> {dst} ({dst.stat().st_size / (1024**2):.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
