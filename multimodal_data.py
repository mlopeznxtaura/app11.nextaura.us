"""Text and latent extraction for multimodal + V-JEPA HF streams."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from data_jsonl import row_training_text
from text_quality import is_garbage_training_text

_URL_ONLY = re.compile(r"^https?://\S+$", re.I)

MULTIMODAL_TEXT_KEYS = (
    "text",
    "caption",
    "captions",
    "question",
    "Question",
    "answer",
    "Answer",
    "instruction",
    "query",
    "response",
    "description",
    "title",
    "summary",
    "utterance",
    "dialogue",
    "conversation",
    "ocr_text",
    "alt_text",
    "image_caption",
    "table_text",
    "content",
    "review",
    "Review",
    "product_review",
    "body",
    "Text Reasoning Trace",
    "Final Answer",
    "blip_caption_beam_5",
    "clip_tags_ViT_L_14_simple_specific",
    "LLM_Description_gpt3_downstream_tasks_ViT_L_14",
    "label",
)

_HEURISTIC_KEY_HINTS = (
    "caption",
    "description",
    "tag",
    "attribute",
    "reasoning",
    "review",
    "question",
    "answer",
    "utterance",
    "dialogue",
    "ocr",
    "alt_text",
    "comment",
    "title",
)


def _heuristic_text_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in row:
        if key in MULTIMODAL_TEXT_KEYS or key in ("image", "images") or key.startswith("_"):
            continue
        kl = key.lower()
        if any(hint in kl for hint in _HEURISTIC_KEY_HINTS):
            keys.append(key)
    return keys

VJEPA_TEXT_KEYS = (
    "text",
    "caption",
    "label",
    "video_id",
    "clip_id",
    "frame_id",
    "source",
    "task",
    "action",
    "scene",
    "description",
    "map_name",
    "zone",
    "tag",
    "factory_id",
    "worker_id",
    "video_key",
    "match_id",
    "sample_id",
    "instruction",
    "lang",
    "subtask",
    "language_instruction",
    "task_description",
    "episode_id",
    "env_name",
    "robot_type",
    "prompt",
    "reasoning",
    "reasoning_trace",
    "physics_description",
    "scenario",
    "question",
    "answer",
    "world_state",
    "task_name",
    "skill",
    "objects",
    "input",
    "output",
    "messages",
)

VJEPA_LATENT_KEYS = (
    "latent",
    "latents",
    "embedding",
    "embeddings",
    "features",
    "feature",
    "vjepa_latent",
    "vjepa_latents",
    "state",
    "hidden_state",
    "vector",
    "tokens",
    "m_flow",
)

_VJEPA_SKIP_KEYS = frozenset(
    {
        "__key__",
        "__url__",
        "pt",
        "frame_bytes",
        "latent_bytes",
        "image",
        "images",
        "video",
        "pixel_values",
        "shards",
        "observation.images",
        "observation.image",
        "wrist_image",
        "exterior_image",
    }
)

_LEROBOT_ACTION_KEYS = (
    "action",
    "actions",
    "observation.state",
    "observation.proprio",
    "state",
    "proprio",
    "eef_pos",
    "eef_quat",
    "gripper",
)

_PHYSICS_META_KEYS = (
    "gravity",
    "mass",
    "velocity",
    "acceleration",
    "friction",
    "density",
    "force",
    "position",
    "orientation",
)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _flatten_vector(raw: Any, dim: int = 12) -> list[float] | None:
    values: list[float] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, list):
                nested = _flatten_vector(item, dim)
                if nested:
                    values.extend(nested)
            if len(values) >= dim * 4:
                break
    elif hasattr(raw, "tolist"):
        try:
            return _flatten_vector(raw.tolist(), dim)
        except Exception:
            return None
    if not values:
        return None
    if len(values) < dim:
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / max(len(values), 1))
        stats = [mean, std, min(values), max(values)]
        values = (values + stats * dim)[:dim]
    step = max(1, len(values) // dim)
    sampled = [values[i * step] for i in range(dim)]
    return [_clamp(v / (abs(v) + 1e-6) if abs(v) > 1 else v) for v in sampled[:dim]]


def _stringify_value(value: Any, max_len: int = 4000) -> str:
    if isinstance(value, str) and value.strip():
        text = value.strip()
        return text[:max_len] + ("..." if len(text) > max_len else "")
    if isinstance(value, list):
        parts = [_stringify_value(v, 800) for v in value[:8]]
        parts = [p for p in parts if p]
        if parts:
            return "\n".join(parts)
    if isinstance(value, dict) and value:
        try:
            blob = json.dumps(value, ensure_ascii=False)
            return blob[:max_len] + ("..." if len(blob) > max_len else "")
        except Exception:
            pass
    return ""


def multimodal_training_text(row: dict[str, Any], min_chars: int = 40) -> str:
    # Reject URL-only rows (e.g. lfsm/multimodal_wiki image links).
    raw = row.get("text")
    if isinstance(raw, str) and _URL_ONLY.match(raw.strip()):
        return ""

    parts: list[str] = []

    for key in MULTIMODAL_TEXT_KEYS:
        value = row.get(key)
        text = _stringify_value(value)
        if text and len(text) >= 8:
            label = key.replace("_", " ").title()
            parts.append(f"{label}:\n{text}")

    for key in _heuristic_text_keys(row):
        value = row.get(key)
        text = _stringify_value(value)
        if text and len(text) >= 8:
            label = key.replace("_", " ").title()
            parts.append(f"{label}:\n{text}")

    conv = row.get("conversations") or row.get("conversation")
    if isinstance(conv, list):
        conv_parts: list[str] = []
        for turn in conv:
            if isinstance(turn, dict):
                role = str(turn.get("from") or turn.get("role") or "msg")
                content = turn.get("value") or turn.get("content") or turn.get("text")
                text = _stringify_value(content, 2000)
                if text:
                    conv_parts.append(f"{role}: {text}")
            elif isinstance(turn, str) and turn.strip():
                conv_parts.append(turn.strip())
        if conv_parts:
            parts.append("\n".join(conv_parts))

    image = row.get("image") or row.get("images")
    if image is not None:
        if isinstance(image, str):
            parts.append(f"Image: {image[:200]}")
        elif isinstance(image, list) and image:
            parts.append(f"Images: {len(image)} item(s)")

    if parts:
        text = "\n\n".join(parts)
        if len(text) >= min_chars and not is_garbage_training_text(text, min_chars=min_chars):
            return text

    fallback = row_training_text(row, min_chars=min_chars)
    if fallback and not is_garbage_training_text(fallback, min_chars=min_chars):
        return fallback
    return ""


def _vjepa_world_state_lines(row: dict[str, Any], *, max_fields: int = 24) -> list[str]:
    """Aggregate scalar world-state fields common in V-JEPA latent corpora."""
    lines: list[str] = []
    for key, value in row.items():
        if key.startswith("_") or key in _VJEPA_SKIP_KEYS or key in VJEPA_TEXT_KEYS:
            continue
        if key in VJEPA_LATENT_KEYS:
            continue
        if isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, str) and value.strip() and len(value) < 256:
            lines.append(f"{key}: {value.strip()}")
        if len(lines) >= max_fields:
            break
    return lines


def _lerobot_action_lines(row: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in _LEROBOT_ACTION_KEYS:
        if key not in row:
            continue
        vec = _flatten_vector(row[key], 8)
        if vec:
            summary = ", ".join(f"{v:.3f}" for v in vec[:6])
            lines.append(f"{key}: [{summary}]")
    return lines


def _physics_meta_lines(row: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for container_key in ("metadata", "meta", "labels", "physics", "kinematics"):
        container = row.get(container_key)
        if not isinstance(container, dict):
            continue
        for pk in _PHYSICS_META_KEYS:
            if pk not in container:
                continue
            val = container[pk]
            if isinstance(val, (list, tuple)) and val:
                vec = _flatten_vector(val, 6)
                if vec:
                    summary = ", ".join(f"{v:.3f}" for v in vec[:4])
                    lines.append(f"{pk}: [{summary}]")
            elif isinstance(val, (int, float)):
                lines.append(f"{pk}: {val}")
            elif isinstance(val, str) and val.strip():
                lines.append(f"{pk}: {val.strip()[:120]}")
    return lines


def _hitl_input_lines(row: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("key", "mouse", "keys", "buttons", "axes"):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, dict):
            parts = [f"{k}={v}" for k, v in list(val.items())[:8]]
            if parts:
                lines.append(f"{key}: " + ", ".join(parts))
        elif isinstance(val, (list, tuple)) and val:
            lines.append(f"{key}: {_stringify_value(val, 200)}")
        elif isinstance(val, str) and val.strip():
            lines.append(f"{key}: {val.strip()[:120]}")
    return lines


def vjepa_training_text(row: dict[str, Any], min_chars: int = 20) -> str:
    parts: list[str] = []

    for key in VJEPA_TEXT_KEYS:
        value = row.get(key)
        text = _stringify_value(value)
        if text:
            parts.append(f"{key}: {text}")

    for key in VJEPA_LATENT_KEYS:
        if key not in row:
            continue
        raw = row[key]
        vec = _flatten_vector(raw, 8)
        if vec:
            summary = ", ".join(f"{v:.3f}" for v in vec[:6])
            parts.append(f"{key} stats: [{summary}]")
            break

    meta = row.get("metadata") or row.get("meta")
    if isinstance(meta, dict):
        meta_text = _stringify_value(meta, 1500)
        if meta_text:
            parts.append(f"metadata: {meta_text}")

    lerobot_lines = _lerobot_action_lines(row)
    if lerobot_lines:
        parts.append("robotics_state:\n" + "\n".join(lerobot_lines))

    physics_lines = _physics_meta_lines(row)
    if physics_lines:
        parts.append("physics:\n" + "\n".join(physics_lines))

    hitl_lines = _hitl_input_lines(row)
    if hitl_lines:
        parts.append("hitl_input:\n" + "\n".join(hitl_lines))

    world_lines = _vjepa_world_state_lines(row)
    if world_lines:
        parts.append("world_state:\n" + "\n".join(world_lines))

    if parts:
        text = "\n\n".join(parts)
        if len(text) >= min_chars:
            return text

    fallback = row_training_text(row, min_chars=min_chars)
    if fallback:
        return fallback

    for key, value in row.items():
        if key.startswith("_") or key in _VJEPA_SKIP_KEYS:
            continue
        if isinstance(value, (int, float, str)) and str(value).strip():
            return f"{key}: {value}"
    return ""


def vjepa_scalar_features(row: dict[str, Any], dim: int = 12) -> list[float] | None:
    gct = row.get("gct_scalar_mean")
    if isinstance(gct, (list, tuple)) and len(gct) >= dim:
        return [float(x) for x in gct[:dim]]
    for key in VJEPA_LATENT_KEYS:
        if key in row:
            vec = _flatten_vector(row[key], dim)
            if vec:
                return vec
    return None
