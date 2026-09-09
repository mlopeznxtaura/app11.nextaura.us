"""Local JSONL + Fable trace + preference text extraction for Stage 1 training."""

from __future__ import annotations

import json
import re
from typing import Any

from text_quality import is_garbage_training_text, is_sapo_metadata_notes, strip_sapo_comment_lines


TEXT_KEYS = ("text", "content", "trace", "prompt", "completion")
SAPO_KEYS = ("observation", "reasoning", "unified_principle", "generalization_notes")
FABLE_KEYS = ("context", "cot", "output", "completion")
PREFERENCE_KEYS = ("chosen", "answer", "response", "completion")
MATH_KEYS = ("statement", "title", "background", "research_summary")
LOCAL_CURRICULUM_KEYS = (
    "action",
    "phase",
    "entry_id",
    "description",
    "analysis",
    "why_this_matters",
    "final_statement",
)
FABLE_MERGED_URL = (
    "https://huggingface.co/datasets/Glint-Research/Fable-5-traces/resolve/main/fable5_cot_merged.jsonl"
)
_SAPO_ROW_MARKERS = ("unified_principle", "sapo_records")
_URL_HEAVY = re.compile(r"https?://", re.I)


def _is_sapo_row(row: dict[str, Any]) -> bool:
    return isinstance(row.get("observation"), str) and isinstance(row.get("unified_principle"), str)


def sapo_training_text(row: dict[str, Any], min_chars: int = 80) -> str:
    """Reformatted SAPO rows: observation + sapo_records scenario, no tag soup."""
    if not _is_sapo_row(row):
        return ""

    parts: list[str] = []
    observation = strip_sapo_comment_lines(str(row.get("observation") or ""))
    if observation:
        parts.append(observation)

    recs = row.get("sapo_records")
    if isinstance(recs, list) and recs and isinstance(recs[0], dict):
        rec = recs[0]
        for label, key in (
            ("Scenario", "scenario"),
            ("Action", "action_taken"),
            ("Outcome", "physical_outcome"),
            ("Principle", "encoded_principle"),
        ):
            value = rec.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(f"{label}: {value.strip()}")
        overfit = rec.get("overfit_rejection")
        if isinstance(overfit, str) and len(overfit.strip()) > 40:
            parts.append(f"Boundary: {overfit.strip()}")

    principle = row.get("unified_principle")
    if isinstance(principle, str) and principle.strip():
        parts.append(f"Unified principle: {principle.strip()}")

    notes = row.get("generalization_notes")
    if isinstance(notes, str) and notes.strip() and not is_sapo_metadata_notes(notes):
        parts.append(notes.strip())

    if not parts:
        return ""
    text = "\n\n".join(parts)
    return text if accept_training_text(text, min_chars=min_chars) else ""


def principle_sft_training_text(row: dict[str, Any], min_chars: int = 80) -> str:
    """principle / user / assistant rows from nextaura-extract files."""
    user = row.get("user")
    assistant = row.get("assistant")
    if not (isinstance(user, str) and isinstance(assistant, str) and user.strip() and assistant.strip()):
        return ""

    parts: list[str] = []
    principle = row.get("principle")
    if isinstance(principle, str) and principle.strip():
        parts.append(f"Principle: {principle.strip()}")
    parts.append(f"User: {user.strip()}")
    parts.append(f"Assistant: {assistant.strip()}")
    text = "\n\n".join(parts)
    return text if accept_training_text(text, min_chars=min_chars) else ""


def should_skip_local_row(row: dict[str, Any]) -> bool:
    """Drop low-signal 4563 web-scrape rows before extraction."""
    if _is_sapo_row(row):
        return False
    if row.get("principle") and row.get("user") and row.get("assistant"):
        return False
    if row.get("step") is not None and row.get("action"):
        return False
    if row.get("messages") or row.get("conversations"):
        return False

    tot = row.get("train_of_thought")
    if isinstance(tot, str) and len(tot.strip()) >= 120:
        return False

    text = row.get("text")
    url = row.get("url")
    if isinstance(text, str) and url and not tot:
        stripped = text.strip()
        if stripped.lower().startswith("http"):
            return True
        if stripped.count("http") > 2 or len(_URL_HEAVY.findall(stripped)) > 2:
            return True
        if len(stripped) < 160:
            return True
    return False


def preference_training_text(row: dict[str, Any], min_chars: int = 80) -> str:
    for key in PREFERENCE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and len(value.strip()) >= min_chars:
            return value.strip()
    return row_training_text(row, min_chars=min_chars)


def nemotron_math_training_text(row: dict[str, Any], min_chars: int = 80) -> str:
    parts: list[str] = []
    problem = row.get("problem")
    if isinstance(problem, str) and problem.strip():
        parts.append(f"Problem:\n{problem.strip()}")

    source = row.get("source")
    if isinstance(source, str) and source.strip():
        parts.append(f"Source: {source.strip()}")

    subset = row.get("subset")
    if isinstance(subset, str) and subset.strip():
        parts.append(f"Subset: {subset.strip()}")

    messages = row.get("messages")
    if isinstance(messages, list):
        msg_parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("type") or "msg").strip()
            content = msg.get("content") or msg.get("text") or msg.get("value")
            if isinstance(content, str) and content.strip():
                msg_parts.append(f"{role}: {content.strip()}")
        if msg_parts:
            parts.append("\n\n".join(msg_parts))

    expected = row.get("expected_answer")
    if isinstance(expected, str) and expected.strip():
        parts.append(f"Expected answer:\n{expected.strip()}")

    if parts:
        text = "\n\n".join(parts)
        if len(text) >= min_chars:
            return text

    return row_training_text(row, min_chars=min_chars)


def nemotron_swe_training_text(row: dict[str, Any], min_chars: int = 80, max_chars: int = 12000) -> str:
    """Agentic SWE trajectories — nvidia/Nemotron-SFT-SWE-v3.5 (messages + tools)."""
    parts: list[str] = []
    tools = row.get("tools")
    if isinstance(tools, list) and tools:
        names: list[str] = []
        for tool in tools[:16]:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        if names:
            parts.append("Tools: " + ", ".join(sorted(set(names))[:10]))

    messages = row.get("messages")
    if isinstance(messages, list):
        msg_parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("type") or "msg").strip()
            content = msg.get("content") or msg.get("text") or msg.get("value")
            if isinstance(content, str) and content.strip():
                text = content.strip()
                if len(text) > 2400:
                    text = text[:2400] + "…"
                msg_parts.append(f"{role}:\n{text}")
        if msg_parts:
            parts.append("\n\n".join(msg_parts))

    if parts:
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        if len(text) >= min_chars:
            return text

    return row_training_text(row, min_chars=min_chars)


def unsolved_math_training_text(row: dict[str, Any], min_chars: int = 80) -> str:
    parts: list[str] = []
    title = row.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"# {title.strip()}")

    category = row.get("category")
    if isinstance(category, dict):
        name = category.get("display_name") or category.get("name")
        if isinstance(name, str) and name.strip():
            parts.append(f"Category: {name.strip()}")
    elif isinstance(category, str) and category.strip():
        parts.append(f"Category: {category.strip()}")

    difficulty = row.get("difficulty")
    if isinstance(difficulty, dict):
        dname = difficulty.get("name") or difficulty.get("level")
        if dname is not None:
            parts.append(f"Difficulty: {dname}")

    status = row.get("status")
    if isinstance(status, str) and status.strip():
        parts.append(f"Status: {status.strip()}")

    statement = row.get("statement")
    if isinstance(statement, str) and statement.strip():
        parts.append(f"Problem:\n{statement.strip()}")

    background = row.get("background")
    if isinstance(background, str) and background.strip():
        bg = background.strip()
        if len(bg) > 12_000:
            bg = bg[:12_000] + "..."
        parts.append(f"Background:\n{bg}")

    research = row.get("research_summary")
    if isinstance(research, str) and research.strip():
        parts.append(f"Research:\n{research.strip()}")

    if parts:
        text = "\n\n".join(parts)
        if len(text) >= min_chars:
            return text

    for key in MATH_KEYS:
        value = row.get(key)
        if isinstance(value, str) and len(value.strip()) >= min_chars:
            return value.strip()
    return row_training_text(row, min_chars=min_chars)


def _format_messages_sft(messages: list[Any], min_chars: int = 80) -> str:
    """Chat-template SFT: User / Assistant turns."""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or msg.get("type") or "msg").strip().lower()
        content = msg.get("content") or msg.get("text") or msg.get("value")
        if not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()
        if role in ("user", "human"):
            parts.append(f"User: {content}")
        elif role in ("assistant", "gpt", "model", "bot"):
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
        else:
            parts.append(f"{role.title()}: {content}")
    if not parts:
        return ""
    text = "\n\n".join(parts)
    return text if len(text) >= min_chars else ""


def row_training_text(row: dict[str, Any], min_chars: int = 80) -> str:
    if should_skip_local_row(row):
        return ""

    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        sft = _format_messages_sft(messages, min_chars=min_chars)
        if sft and not is_garbage_training_text(sft, min_chars=min_chars):
            return sft

    sft = principle_sft_training_text(row, min_chars=min_chars)
    if sft:
        return sft

    sapo = sapo_training_text(row, min_chars=min_chars)
    if sapo:
        return sapo

    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and len(value.strip()) >= min_chars:
            text = value.strip()
            if not is_garbage_training_text(text, min_chars=min_chars):
                return text

    fable_parts: list[str] = []
    for key in FABLE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            fable_parts.append(value.strip())
        elif isinstance(value, dict) and value:
            fable_parts.append(json.dumps(value, ensure_ascii=False)[:12_000])
    if fable_parts:
        text = "\n\n".join(fable_parts)
        if accept_training_text(text, min_chars=min_chars):
            return text

    curriculum_parts: list[str] = []
    step = row.get("step")
    if step is not None and str(step).strip():
        curriculum_parts.append(f"Step {step}")
    for key in LOCAL_CURRICULUM_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            label = key.replace("_", " ").title()
            curriculum_parts.append(f"{label}: {value.strip()}")
    if curriculum_parts:
        text = "\n\n".join(curriculum_parts)
        if len(text) >= min_chars and not is_garbage_training_text(text, min_chars=min_chars):
            return text

    tot = row.get("train_of_thought")
    if isinstance(tot, str) and len(tot.strip()) >= min_chars:
        text = tot.strip()
        if not is_garbage_training_text(text, min_chars=min_chars):
            return text

    return ""


def accept_training_text(text: str, min_chars: int = 80) -> bool:
    return bool(text) and len(text) >= min_chars and not is_garbage_training_text(text, min_chars=min_chars)


def load_jsonl_lines(path: str, *, skip_invalid: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows
