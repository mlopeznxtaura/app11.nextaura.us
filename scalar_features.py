"""12-d scalar feature vectors for L2EVAL-style scalar_input_projection."""

from __future__ import annotations

import json
import math
import re
from typing import Any

SCALAR_DIM = 12


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def text_scalar_features(text: str, dim: int = SCALAR_DIM) -> list[float]:
    """Derive a fixed-size scalar vector from raw text (geometry / binary / language proxies)."""
    if not text:
        return [0.0] * dim
    n = len(text)
    lines = text.count("\n") + 1
    words = len(text.split())
    digits = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)
    lower = sum(c.islower() for c in text)
    space = text.count(" ")
    punct = sum(not c.isalnum() and not c.isspace() for c in text)
    braces = text.count("{") + text.count("}") + text.count("[") + text.count("]")
    codeish = bool(re.search(r"[{}();=<>]|def |class |function ", text))
    jsonish = False
    try:
        json.loads(text.strip()[:2000])
        jsonish = True
    except Exception:
        jsonish = text.strip().startswith("{") or text.strip().startswith("[")

    feats = [
        _clamp(math.log1p(n) / 10.0),
        _clamp(lines / 50.0),
        _clamp(words / 500.0),
        _clamp(digits / max(n, 1) * 5.0),
        _clamp(upper / max(n, 1) * 5.0),
        _clamp(lower / max(n, 1) * 2.0),
        _clamp(space / max(n, 1) * 3.0),
        _clamp(punct / max(n, 1) * 5.0),
        _clamp(braces / max(n, 1) * 10.0),
        1.0 if codeish else -1.0,
        1.0 if jsonish else -1.0,
        _clamp((n % 256) / 128.0 - 1.0),
    ]
    return feats[:dim] + [0.0] * max(0, dim - len(feats))


def row_scalar_features(row: dict[str, Any], dim: int = SCALAR_DIM) -> list[float] | None:
    raw = row.get("scalar") or row.get("scalars") or row.get("scalar_vector")
    if isinstance(raw, list) and len(raw) >= dim:
        try:
            return [_clamp(float(x)) for x in raw[:dim]]
        except (TypeError, ValueError):
            pass
    if isinstance(raw, dict):
        keys = (
            "geometry",
            "binary",
            "language",
            "g1",
            "g2",
            "g3",
            "b1",
            "b2",
            "b3",
            "l1",
            "l2",
            "l3",
        )
        if all(k in raw for k in keys[:dim]):
            try:
                return [_clamp(float(raw[k])) for k in keys[:dim]]
            except (TypeError, ValueError):
                pass
    return None
