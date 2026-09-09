#!/usr/bin/env python3
"""Cache world-model HF datasets one-by-one; skip gated/missing; continue on failure."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vjepa_local_cache import (
    cache_dir,
    cache_path,
    cached_datasets,
    download_dataset_from_hf,
    load_manifest,
    save_manifest,
    upload_cache_to_cos,
)
from world_model_mix import VJEPA_MIX_WORLD

# Gated or non-HF — skip to keep the pipeline simple.
SKIP_DATASETS = frozenset({"facebook/jepa-wms"})


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — HF download may fail", file=sys.stderr)

    targets = [ds for ds in VJEPA_MIX_WORLD if ds not in SKIP_DATASETS]
    root = cache_dir()
    os.makedirs(root, exist_ok=True)
    print(f"Cache dir: {root}")
    print(f"Targets: {len(targets)} datasets (skipping {sorted(SKIP_DATASETS)})")

    manifest = load_manifest(root)
    if "datasets" not in manifest:
        manifest["datasets"] = {}

    ready_before = cached_datasets(targets, root)
    print(f"Already cached: {len(ready_before)}/{len(targets)} — {ready_before}")

    results: dict[str, dict] = {}
    for ds_id in targets:
        path = cache_path(ds_id, root)
        if os.path.isfile(path) and os.path.getsize(path) >= 64:
            print(f"[SKIP] {ds_id} — already cached ({os.path.getsize(path)} bytes)")
            results[ds_id] = {"dataset_id": ds_id, "source": "local", "skipped": True}
            continue

        part = path + ".part"
        if os.path.isfile(part) and os.path.getsize(part) == 0:
            os.remove(part)

        print(f"[CACHE] {ds_id} …", flush=True)
        try:
            info = download_dataset_from_hf(ds_id, token=token, root=root)
            if info.get("rows", 0) <= 0 or info.get("bytes", 0) < 64:
                raise RuntimeError(f"no usable rows cached (rows={info.get('rows')}, bytes={info.get('bytes')})")
            manifest["datasets"][ds_id] = info
            save_manifest(manifest, root)
            results[ds_id] = info
            print(f"[OK]   {ds_id} — {info['rows']} rows, {info['bytes']} bytes")
        except Exception as exc:
            results[ds_id] = {"dataset_id": ds_id, "error": str(exc)}
            print(f"[FAIL] {ds_id} — {exc}", file=sys.stderr)
            if os.path.isfile(part):
                try:
                    os.remove(part)
                except OSError:
                    pass

    ready_after = cached_datasets(targets, root)
    print(f"\nDone: {len(ready_after)}/{len(targets)} cached — {ready_after}")

    try:
        keys = upload_cache_to_cos(root)
        if keys:
            print(f"COS upload: {len(keys)} objects")
    except Exception as exc:
        print(f"COS upload skipped: {exc}", file=sys.stderr)

    print(json.dumps({"ready": ready_after, "results": results}, indent=2, default=str))
    if len(ready_after) <= len(ready_before):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
