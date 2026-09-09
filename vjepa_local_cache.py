"""Local NVMe + IBM COS cache for V-JEPA datasets (MCF v2 data plane)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterator

VJEPA_CACHE_DIR = os.environ.get(
    "VJEPA_CACHE_DIR",
    os.path.join(os.environ.get("DATA_DIR", "data"), "vjepa-cache"),
)
COS_DATASET_PREFIX = os.environ.get("COS_DATASET_PREFIX", "app9/datasets/vjepa/").rstrip("/") + "/"
MANIFEST_NAME = "manifest.json"
MAX_CACHE_ROWS = int(os.environ.get("VJEPA_CACHE_MAX_ROWS", "12000"))


def cache_dir() -> str:
    return os.path.abspath(VJEPA_CACHE_DIR)


def dataset_slug(dataset_id: str) -> str:
    return dataset_id.replace("/", "__").replace(":", "_")


def cache_path(dataset_id: str, root: str | None = None) -> str:
    return os.path.join(root or cache_dir(), f"{dataset_slug(dataset_id)}.jsonl")


def manifest_path(root: str | None = None) -> str:
    return os.path.join(root or cache_dir(), MANIFEST_NAME)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cos_client():
    api_key = os.environ.get("IBM_CLOUD_API_KEY", "").strip()
    if not api_key:
        return None
    import ibm_boto3
    from ibm_botocore.client import Config

    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=api_key,
        ibm_service_instance_id=os.environ.get(
            "COS_CRN",
            "crn:v1:bluemix:public:cloud-object-storage:global:a/ee54102c4e17411fa08552596d94e53d:49c90492-ab55-4cba-90f4-589623751191::",
        ),
        config=Config(signature_version="oauth"),
        endpoint_url=os.environ.get(
            "COS_ENDPOINT",
            "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        ),
    )


def _cos_bucket() -> str:
    return os.environ.get("COS_BUCKET", "nextaura-app9-stage1")


def _cos_key(name: str) -> str:
    return f"{COS_DATASET_PREFIX}{name}"


def load_manifest(root: str | None = None) -> dict[str, Any]:
    path = manifest_path(root)
    if not os.path.isfile(path):
        return {"datasets": {}, "updated_at": None}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest: dict[str, Any], root: str | None = None) -> None:
    root = root or cache_dir()
    os.makedirs(root, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path(root), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def cache_ready(dataset_ids: list[str], root: str | None = None) -> bool:
    root = root or cache_dir()
    for ds_id in dataset_ids:
        path = cache_path(ds_id, root)
        if not os.path.isfile(path) or os.path.getsize(path) < 64:
            return False
    return True


def cached_datasets(dataset_ids: list[str], root: str | None = None) -> list[str]:
    root = root or cache_dir()
    ready: list[str] = []
    for ds_id in dataset_ids:
        path = cache_path(ds_id, root)
        if os.path.isfile(path) and os.path.getsize(path) >= 64:
            ready.append(ds_id)
    return ready


def _row_iter(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def open_hybrid_vjepa_interleaved(
    dataset_ids: list[str],
    *,
    token: str | None = None,
    root: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Round-robin: local JSONL when cached, else live HF stream per dataset."""
    import logging

    from hf_stream_loader import open_hf_train_stream

    log = logging.getLogger(__name__)
    root = root or cache_dir()
    local = set(cached_datasets(dataset_ids, root))

    class _LazySlot:
        __slots__ = ("ds_id", "local_path", "is_local", "it", "dead")

        def __init__(self, ds_id: str) -> None:
            self.ds_id = ds_id
            self.is_local = ds_id in local
            self.local_path = cache_path(ds_id, root) if self.is_local else ""
            self.it: Iterator[dict[str, Any]] | None = None
            self.dead = False

        def _ensure(self) -> bool:
            if self.dead:
                return False
            if self.it is not None:
                return True
            try:
                if self.is_local:
                    self.it = _row_iter(self.local_path)
                    return True
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(open_hf_train_stream, self.ds_id, kind="vjepa", token=token)
                    try:
                        stream = fut.result(timeout=25)
                    except concurrent.futures.TimeoutError:
                        log.warning("hybrid HF open timeout %s (>25s) — skipping slot", self.ds_id)
                        self.dead = True
                        return False
                self.it = iter(stream)
                return True
            except Exception as exc:
                log.warning("hybrid skip HF %s: %s", self.ds_id, exc)
                self.dead = True
                return False

        def next_row(self) -> dict[str, Any] | None:
            if not self._ensure():
                return None
            assert self.it is not None
            if self.is_local:
                try:
                    row = next(self.it)
                    return row if isinstance(row, dict) else None
                except StopIteration:
                    self.dead = True
                    return None
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(next, self.it)
                try:
                    row = fut.result(timeout=20)
                    return row if isinstance(row, dict) else None
                except concurrent.futures.TimeoutError:
                    log.warning("hybrid HF timeout %s (>20s) — skipping slot", self.ds_id)
                    self.dead = True
                    return None
                except StopIteration:
                    self.dead = True
                    return None
                except Exception as exc:
                    log.warning("hybrid HF error %s: %s", self.ds_id, exc)
                    self.dead = True
                    return None

    slots = [_LazySlot(ds_id) for ds_id in dataset_ids]
    active = list(range(len(slots)))
    while active:
        exhausted: list[int] = []
        for i in active:
            row = slots[i].next_row()
            if row is None:
                exhausted.append(i)
                continue
            yield row
        for i in exhausted:
            active.remove(i)


def _slim_vjepa_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip multi-MB latent tensors; keep only fields used for training text + scalars."""
    from multimodal_data import (
        VJEPA_LATENT_KEYS,
        VJEPA_TEXT_KEYS,
        _VJEPA_SKIP_KEYS,
        _flatten_vector,
        vjepa_training_text,
    )

    if not isinstance(row, dict):
        return {}
    out: dict[str, Any] = {}
    for key in VJEPA_TEXT_KEYS:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            text = str(val).strip()
            if text:
                out[key] = text if isinstance(val, str) else val
    for key in VJEPA_LATENT_KEYS:
        if key not in row:
            continue
        vec = _flatten_vector(row[key], 12)
        if vec:
            out["latent"] = vec
            break
    meta = row.get("metadata") or row.get("meta")
    if isinstance(meta, dict):
        slim_meta: dict[str, Any] = {}
        for mk, mv in meta.items():
            if isinstance(mv, (str, int, float, bool)) and len(str(mv)) < 800:
                slim_meta[str(mk)] = mv
        if slim_meta:
            out["metadata"] = slim_meta
    elif isinstance(meta, str) and meta.strip():
        out["metadata"] = meta.strip()[:2000]
    for key, value in row.items():
        if key in out or key.startswith("_") or key in _VJEPA_SKIP_KEYS or key in VJEPA_LATENT_KEYS:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str) and 0 < len(value) < 256:
            out[key] = value.strip()
        if len(out) >= 36:
            break
    if row.get("gct_scalar_mean"):
        out["gct_scalar_mean"] = row["gct_scalar_mean"]
    text = vjepa_training_text(out, min_chars=16)
    if not text:
        return {}
    return out


def download_dataset_from_hf(
    dataset_id: str,
    *,
    token: str | None = None,
    root: str | None = None,
    max_scan_rows: int | None = None,
) -> dict[str, Any]:
    from hf_stream_loader import open_hf_train_stream

    root = root or cache_dir()
    os.makedirs(root, exist_ok=True)
    path = cache_path(dataset_id, root)
    tmp = path + ".part"
    rows = 0
    scanned = 0
    scan_limit = max_scan_rows or int(os.environ.get("VJEPA_CACHE_MAX_SCAN", "80000"))
    with open(tmp, "w", encoding="utf-8") as out:
        for row in open_hf_train_stream(dataset_id, kind="vjepa", token=token):
            scanned += 1
            if not isinstance(row, dict):
                if scanned >= scan_limit:
                    break
                continue
            slim = _slim_vjepa_row(row)
            if not slim:
                if scanned >= scan_limit:
                    break
                continue
            out.write(json.dumps(slim, default=_json_default, ensure_ascii=False) + "\n")
            rows += 1
            if rows >= MAX_CACHE_ROWS:
                break
            if scanned >= scan_limit:
                break
    size = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    if rows <= 0 or size < 64:
        if os.path.isfile(tmp):
            os.remove(tmp)
        return {
            "dataset_id": dataset_id,
            "rows": 0,
            "bytes": 0,
            "scanned": scanned,
            "file": os.path.basename(path),
        }
    os.replace(tmp, path)
    size = os.path.getsize(path)
    return {
        "dataset_id": dataset_id,
        "rows": rows,
        "bytes": size,
        "scanned": scanned,
        "file": os.path.basename(path),
    }


def upload_cache_to_cos(root: str | None = None) -> list[str]:
    cos = _cos_client()
    if not cos:
        return []
    root = root or cache_dir()
    uploaded: list[str] = []
    bucket = _cos_bucket()
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl") and name != MANIFEST_NAME:
            continue
        local = os.path.join(root, name)
        if not os.path.isfile(local):
            continue
        key = _cos_key(name)
        with open(local, "rb") as handle:
            cos.put_object(Bucket=bucket, Key=key, Body=handle, ContentType="application/json")
        uploaded.append(key)
    manifest = load_manifest(root)
    if manifest.get("datasets"):
        key = _cos_key(MANIFEST_NAME)
        cos.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        uploaded.append(key)
    return uploaded


def sync_cache_from_cos(dataset_ids: list[str], root: str | None = None) -> bool:
    cos = _cos_client()
    if not cos:
        return False
    root = root or cache_dir()
    os.makedirs(root, exist_ok=True)
    bucket = _cos_bucket()
    ok = True
    names = [MANIFEST_NAME] + [os.path.basename(cache_path(ds, root)) for ds in dataset_ids]
    for name in names:
        key = _cos_key(name)
        local = os.path.join(root, name)
        try:
            obj = cos.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
            tmp = local + ".part"
            with open(tmp, "wb") as handle:
                handle.write(body)
            os.replace(tmp, local)
        except Exception:
            ok = False
    return ok and cache_ready(dataset_ids, root)


def ensure_vjepa_cache(
    dataset_ids: list[str],
    *,
    hf_token: str | None = None,
    cos_sync: bool = True,
    root: str | None = None,
) -> dict[str, Any]:
    """Ensure local cache exists: local disk → IBM COS → HuggingFace."""
    root = root or cache_dir()
    if cache_ready(dataset_ids, root):
        return {"ok": True, "source": "local", "manifest": load_manifest(root)}

    if cos_sync and sync_cache_from_cos(dataset_ids, root):
        return {"ok": True, "source": "cos", "manifest": load_manifest(root)}

    manifest: dict[str, Any] = {"datasets": {}}
    for ds_id in dataset_ids:
        if os.path.isfile(cache_path(ds_id, root)) and os.path.getsize(cache_path(ds_id, root)) >= 64:
            manifest["datasets"][ds_id] = {"dataset_id": ds_id, "source": "local", "skipped": True}
            continue
        try:
            info = download_dataset_from_hf(ds_id, token=hf_token, root=root)
            if info.get("rows", 0) <= 0:
                info = {**info, "error": f"no usable rows (scanned {info.get('scanned', '?')})"}
        except Exception as exc:
            info = {"dataset_id": ds_id, "error": str(exc)}
        manifest["datasets"][ds_id] = info
    save_manifest(manifest, root)

    cos_keys: list[str] = []
    if cos_sync:
        try:
            cos_keys = upload_cache_to_cos(root)
        except Exception:
            cos_keys = []

    return {"ok": True, "source": "huggingface", "manifest": manifest, "cos_keys": cos_keys}


def open_local_vjepa_interleaved(dataset_ids: list[str], root: str | None = None) -> Iterator[dict[str, Any]]:
    """Round-robin rows from cached JSONL files (same pattern as HF interleave)."""
    root = root or cache_dir()
    if not cache_ready(dataset_ids, root):
        raise RuntimeError(f"V-JEPA local cache incomplete under {root}")

    iters = [_row_iter(cache_path(ds_id, root)) for ds_id in dataset_ids]
    active = list(range(len(iters)))
    while active:
        exhausted: list[int] = []
        for i in active:
            try:
                yield next(iters[i])
            except StopIteration:
                exhausted.append(i)
        for i in exhausted:
            active.remove(i)


def local_vjepa_stream_factory(dataset_ids: list[str], root: str | None = None):
    def _factory():
        return open_local_vjepa_interleaved(dataset_ids, root)

    return _factory
