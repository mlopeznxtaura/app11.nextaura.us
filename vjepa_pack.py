"""V-JEPA latent row packing for MCF v2 throughput (Hierarchical Latent Stack)."""

from __future__ import annotations

from typing import Any, Iterator

from multimodal_data import vjepa_scalar_features, vjepa_training_text


def pack_vjepa_rows(
    row_iter: Iterator[dict[str, Any]],
    *,
    pack_rows: int = 128,
    min_chars: int = 4000,
    min_accept_chars: int = 20,
) -> Iterator[tuple[str, dict[str, Any] | None]]:
    """Concatenate N short V-JEPA metadata rows into one dense training document."""
    buf_texts: list[str] = []
    buf_rows: list[dict[str, Any]] = []
    char_count = 0

    def _flush() -> tuple[str, dict[str, Any] | None] | None:
        nonlocal buf_texts, buf_rows, char_count
        if not buf_texts:
            return None
        combined = "\n\n--- frame ---\n\n".join(buf_texts)
        meta_row = _merge_pack_metadata(buf_rows)
        buf_texts = []
        buf_rows = []
        char_count = 0
        return combined, meta_row

    for row in row_iter:
        if not isinstance(row, dict):
            continue
        text = vjepa_training_text(row, min_chars=min_accept_chars)
        if not text or len(text) < min_accept_chars:
            continue
        buf_texts.append(text)
        buf_rows.append(row)
        char_count += len(text)
        if len(buf_texts) >= pack_rows or char_count >= min_chars:
            out = _flush()
            if out:
                yield out

    tail = _flush()
    if tail:
        yield tail


def _merge_pack_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build GCT-style scalar summary from packed frame block."""
    if not rows:
        return {}
    scalars: list[list[float]] = []
    for row in rows:
        vec = vjepa_scalar_features(row, dim=12)
        if vec:
            scalars.append(vec)
    merged: dict[str, Any] = {
        "pack_frames": len(rows),
        "pack_source": "mcf-v2-hls",
    }
    if scalars:
        dim = len(scalars[0])
        mean = [sum(s[i] for s in scalars) / len(scalars) for i in range(dim)]
        merged["gct_scalar_mean"] = mean
    first, last = rows[0], rows[-1]
    for key in ("map_name", "factory_id", "video_key", "match_id", "tag"):
        if key in first:
            merged[f"pack_{key}_start"] = first.get(key)
        if key in last:
            merged[f"pack_{key}_end"] = last.get(key)
    return merged
