"""Expand intelligence seed corpus (axiom/theorem/essay rows) into chat SFT JSONL."""

from __future__ import annotations

from typing import Any

from data_jsonl import accept_training_text, load_jsonl_lines, row_training_text

INTELLIGENCE_SEED_FILE = "intelligence-seed-corpus.jsonl"
INTELLIGENCE_SEED_ML_FILE = "intelligence-seed-corpus-ml.jsonl"
INTELLIGENCE_SEED_FILES = (INTELLIGENCE_SEED_FILE, INTELLIGENCE_SEED_ML_FILE)
INTELLIGENCE_SFT_FILE = "intelligence-sft-expanded.jsonl"
SFT_FULL_MERGED_FILE = "sft-full-merged.jsonl"

_ASSISTANT_MAX = 6000


def _first_sentence(text: str) -> str:
    parts = text.split(". ")
    head = (parts[0] if parts else text).strip()
    if head and not head.endswith("."):
        head += "."
    return head


def intelligence_seed_to_sft_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one seed row into multiple User/Assistant SFT examples."""
    text = str(row.get("text") or "").strip()
    if len(text) < 80:
        return []

    seed_id = row.get("id", "?")
    tier = row.get("tier", "?")
    domain = str(row.get("domain") or "general").replace("_", " ")
    token_class = str(row.get("token_class") or "concept").replace("_", " ")
    source = f"intelligence-seed://{seed_id}"
    principle = f"Rigorous {domain} reasoning ({token_class}, tier {tier})"
    assistant = text[:_ASSISTANT_MAX]
    first = _first_sentence(text)

    domain_label = domain.replace("_", " ")
    prompts: list[str] = [
        f"Explain this {token_class} about {domain_label} (tier {tier}).",
        f"What are the key ideas in {domain_label} entry {seed_id}?",
        f"Teach me the {token_class} for {domain_label}.",
    ]
    if len(first) > 40:
        prompts.append(f"Elaborate: {first}")
    if token_class in ("theorem", "definition", "axiom"):
        prompts.append(f"State and explain the {token_class} in {domain_label}.")
    if token_class in ("technical_note", "concept"):
        prompts.extend(
            [
                f"How does this apply to training a 50M transformer?",
                f"Summarize the practical takeaway for {domain_label}.",
                f"What should I know about {first[:80]}?",
            ]
        )
    if token_class == "example" and domain == "code":
        prompts.append("Walk through this code and explain the reasoning.")
    if token_class == "essay":
        prompts.append(f"Discuss: {first}")

    out: list[dict[str, Any]] = []
    seen_users: set[str] = set()
    for user in prompts:
        user = user.strip()
        if not user or user in seen_users:
            continue
        seen_users.add(user)
        out.append(
            {
                "principle": principle,
                "user": user,
                "assistant": assistant,
                "source": source,
                "seed_id": seed_id,
                "tier": tier,
                "domain": row.get("domain"),
                "token_class": token_class,
            }
        )
    return out


def expand_intelligence_seeds(seed_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl_lines(seed_path):
        rows.extend(intelligence_seed_to_sft_rows(row))
    return rows


def expand_intelligence_seed_files(seed_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in seed_paths:
        rows.extend(expand_intelligence_seeds(path))
    return rows


def write_intelligence_sft_expanded(seed_path: str, dest: str) -> dict[str, int]:
    import hashlib
    import json
    import os

    candidates = expand_intelligence_seeds(seed_path)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in candidates:
        text = row_training_text(row)
        if not accept_training_text(text):
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"input": len(candidates), "kept": len(kept), "skipped": len(candidates) - len(kept)}


def write_intelligence_sft_expanded_all(seed_paths: list[str], dest: str) -> dict[str, int]:
    import hashlib
    import json
    import os

    candidates = expand_intelligence_seed_files(seed_paths)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in candidates:
        text = row_training_text(row)
        if not accept_training_text(text):
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"input": len(candidates), "kept": len(kept), "skipped": len(candidates) - len(kept)}
