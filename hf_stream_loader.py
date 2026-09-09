"""HuggingFace streaming loaders for multimodal + V-JEPA datasets (configs, fallbacks)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

log = logging.getLogger(__name__)

# Zebra-CoT smoke — text-heavy, parquet-backed (not "default").
ZEBRA_SMOKE_CONFIG = "Scientific Reasoning - Geometry"

MULTIMODAL_STREAM_FALLBACKS: dict[str, list[dict[str, Any]]] = {
    "DAMO-NLP-SG/multimodal_textbook": [
        {"path": "DAMO-NLP-SG/multimodal_textbook", "split": "train"},
        {
            "path": "parquet",
            "data_files": "hf://datasets/DAMO-NLP-SG/multimodal_textbook/**/*.parquet",
            "split": "train",
        },
    ],
    "multimodal-reasoning-lab/Zebra-CoT": [
        {"path": "multimodal-reasoning-lab/Zebra-CoT", "name": ZEBRA_SMOKE_CONFIG, "split": "train"},
        {"path": "multimodal-reasoning-lab/Zebra-CoT", "name": "2D Visual Reasoning - Visual Search", "split": "train"},
    ],
    "SamBP069/olives-multimodal-dataset": [
        {"path": "SamBP069/olives-multimodal-dataset", "split": "train"},
        {"path": "SamBP069/olives-multimodal-dataset", "name": "default", "split": "train"},
    ],
    "WenxingZhu/multimodal-embedding-100M": [
        {
            "path": "parquet",
            "data_files": "hf://datasets/WenxingZhu/multimodal-embedding-100M/**/*.parquet",
            "split": "train",
        },
    ],
}

VJEPA_STREAM_FALLBACKS: dict[str, list[dict[str, Any]]] = {
    "rookierufus/epic-kitchens-vjepa": [
        {"path": "rookierufus/epic-kitchens-vjepa", "split": "train"},
        {"path": "phi-9/epic-kitchens-vjepa", "split": "train"},
    ],
    "facebook/jepa-wms": [
        {"path": "facebook/jepa-wms", "name": "metaworld", "split": "train"},
        {"path": "facebook/jepa-wms", "name": "pusht", "split": "train"},
        {"path": "facebook/jepa-wms", "name": "pointmaze", "split": "train"},
        {"path": "facebook/jepa-wms", "name": "wall", "split": "train"},
        {"path": "facebook/jepa-wms", "name": "robocasa", "split": "train"},
        {"path": "facebook/jepa-wms", "split": "train"},
    ],
    "agibot-world/AgiBotWorld2026": [
        {"path": "agibot-world/AgiBotWorld2026", "split": "train"},
        {
            "path": "parquet",
            "data_files": "hf://datasets/agibot-world/AgiBotWorld2026/**/*.parquet",
            "split": "train",
        },
    ],
    "Silicon23/mujoco-kinematics-probing": [
        {"path": "Silicon23/mujoco-kinematics-probing", "split": "train"},
        {"path": "Silicon23/mujoco-kinematics-probing", "name": "default", "split": "train"},
    ],
    "Swastikr/PhysSim-VLM-Dataset": [
        {"path": "Swastikr/PhysSim-VLM-Dataset", "split": "train"},
    ],
    "Swastikr/PhysSim-VLM-SFT-R2-Data": [
        {"path": "Swastikr/PhysSim-VLM-SFT-R2-Data", "split": "train"},
    ],
    "jialei02/libero_merged_no_noops_20hz": [
        {
            "path": "parquet",
            "data_files": "hf://datasets/jialei02/libero_merged_no_noops_20hz/**/*.parquet",
            "split": "train",
        },
        {"path": "jialei02/libero_merged_no_noops_20hz", "split": "train"},
    ],
    "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings": [
        {
            "path": "parquet",
            "data_files": "hf://datasets/quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings/**/*.parquet",
            "split": "train",
        },
        {"path": "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings", "split": "train"},
    ],
    "HuberyLL/nms_hitl_world_model": [
        {"path": "HuberyLL/nms_hitl_world_model", "split": "train"},
    ],
}

# Datasets that are embedding-only / no extractable text — hide from UI unless probed OK.
EMBEDDING_ONLY_HINTS = (
    "embedding",
    "latent",
    "vjepa2-features",
    "vjepa2-latents",
    "vjepa-latents",
    "tokenized-shards",
)


@dataclass
class HfStreamAttempt:
    label: str
    kwargs: dict[str, Any]


def _dataset_config_names(dataset_id: str, token: str | None) -> list[str | None]:
    try:
        from datasets import get_dataset_config_names

        names = list(get_dataset_config_names(dataset_id, token=token))
        return names if names else [None]
    except Exception as exc:
        log.debug("config names for %s: %s", dataset_id, exc)
        return [None]


def _attempts_for(dataset_id: str, kind: str) -> list[HfStreamAttempt]:
    fallbacks = MULTIMODAL_STREAM_FALLBACKS if kind == "multimodal" else VJEPA_STREAM_FALLBACKS
    specs: list[dict[str, Any]] = []
    if dataset_id in fallbacks:
        specs.extend(fallbacks[dataset_id])
    else:
        specs.append({"path": dataset_id, "split": "train"})
        for name in _dataset_config_names(dataset_id, None):
            if name:
                specs.append({"path": dataset_id, "name": name, "split": "train"})
    seen: set[str] = set()
    out: list[HfStreamAttempt] = []
    for i, s in enumerate(specs):
        key = repr(sorted(s.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(HfStreamAttempt(label=str(i), kwargs=dict(s)))
    return out


def _load_dataset_stream(spec: dict[str, Any], *, token: str | None, streaming: bool) -> Any:
    from datasets import load_dataset

    kwargs = dict(spec)
    path = kwargs.pop("path")
    name = kwargs.pop("name", None)
    split = kwargs.pop("split", "train")
    extra = {k: v for k, v in kwargs.items() if v is not None}
    load_kwargs: dict[str, Any] = {
        "streaming": streaming,
        "trust_remote_code": True,
        **extra,
    }
    if token:
        load_kwargs["token"] = token
    if name is not None:
        return load_dataset(path, name=name, split=split, **load_kwargs)
    return load_dataset(path, split=split, **load_kwargs)


def _iter_stream(ds: Any) -> Iterator[dict[str, Any]]:
    if hasattr(ds, "__iter__"):
        yield from ds
        return
    if isinstance(ds, dict):
        for split in ("train", "validation", "test"):
            if split in ds:
                yield from ds[split]
                return
        first = next(iter(ds.values()))
        yield from first


def open_hf_train_stream(
    dataset_id: str,
    *,
    kind: str = "multimodal",
    token: str | None = None,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    """Open a streaming row iterator; tries dataset-specific fallbacks on failure."""
    errors: list[str] = []
    for attempt in _attempts_for(dataset_id, kind):
        try:
            ds = _load_dataset_stream(attempt.kwargs, token=token, streaming=streaming)
            return _iter_stream(ds)
        except Exception as exc:
            errors.append(f"{attempt.kwargs}: {exc}")
    raise RuntimeError(f"HF stream failed for {dataset_id} ({kind}): " + " | ".join(errors[:3]))


def probe_hf_train_stream(
    dataset_id: str,
    *,
    kind: str = "multimodal",
    token: str | None = None,
    max_rows: int = 200,
) -> dict[str, Any]:
    """Scan up to max_rows for the first row that yields training text."""
    from multimodal_data import multimodal_training_text, vjepa_training_text

    text_fn = multimodal_training_text if kind == "multimodal" else vjepa_training_text
    try:
        stream = open_hf_train_stream(dataset_id, kind=kind, token=token)
    except Exception as exc:
        return {
            "ok": False,
            "dataset_id": dataset_id,
            "kind": kind,
            "rows_scanned": 0,
            "error": str(exc),
        }
    seen = 0
    for row in stream:
        seen += 1
        if not isinstance(row, dict):
            continue
        text = text_fn(row, min_chars=24)
        if text and len(text) >= 24:
            return {
                "ok": True,
                "dataset_id": dataset_id,
                "kind": kind,
                "rows_scanned": seen,
                "keys": list(row.keys())[:20],
                "text_preview": text[:200],
            }
        if seen >= max_rows:
            break
    return {
        "ok": False,
        "dataset_id": dataset_id,
        "kind": kind,
        "rows_scanned": seen,
        "error": f"no usable text in first {seen} rows",
    }


def probe_all_registered(
    multimodal_ids: list[str],
    vjepa_ids: list[str],
    *,
    token: str | None = None,
    max_rows: int = 80,
) -> dict[str, dict[str, Any]]:
    """Probe every registered dataset; returns {dataset_id: probe_result}."""
    out: dict[str, dict[str, Any]] = {}
    for ds_id in multimodal_ids:
        out[ds_id] = probe_hf_train_stream(ds_id, kind="multimodal", token=token, max_rows=max_rows)
    for ds_id in vjepa_ids:
        out[ds_id] = probe_hf_train_stream(ds_id, kind="vjepa", token=token, max_rows=max_rows)
    return out


def likely_embedding_only(dataset_id: str) -> bool:
    low = dataset_id.lower()
    return any(h in low for h in EMBEDDING_ONLY_HINTS)


def open_hf_train_stream_interleaved(
    dataset_ids: list[str],
    *,
    kind: str = "vjepa",
    token: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Round-robin rows across multiple HF streams (equal weight per dataset)."""
    if not dataset_ids:
        raise ValueError("dataset_ids must not be empty")
    if len(dataset_ids) == 1:
        yield from open_hf_train_stream(dataset_ids[0], kind=kind, token=token)
        return

    class _LazyHfSlot:
        __slots__ = ("ds_id", "it", "dead")

        def __init__(self, ds_id: str) -> None:
            self.ds_id = ds_id
            self.it: Iterator[dict[str, Any]] | None = None
            self.dead = False

        def _ensure(self) -> bool:
            if self.dead:
                return False
            if self.it is not None:
                return True
            try:
                self.it = iter(open_hf_train_stream(self.ds_id, kind=kind, token=token))
                return True
            except Exception as exc:
                log.warning("HF interleave skip %s: %s", self.ds_id, exc)
                self.dead = True
                return False

        def next_row(self) -> dict[str, Any] | None:
            if not self._ensure():
                return None
            assert self.it is not None
            try:
                row = next(self.it)
                return row if isinstance(row, dict) else None
            except StopIteration:
                self.dead = True
                return None

    slots = [_LazyHfSlot(ds) for ds in dataset_ids]
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
