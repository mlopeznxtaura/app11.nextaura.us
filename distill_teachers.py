"""Response-level distillation: teacher HF models → JSONL corpus for student SFT."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from data_jsonl import accept_training_text
from reward_eval import load_benchmark_prompts
from sft_topics import SFT_TOPICS, TOPIC_LABELS, expand_topic_seeds, topic_jsonl_filename

ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = ROOT / "data" / "benchmark-prompts.jsonl"

DISTILL_TOPICS: tuple[str, ...] = SFT_TOPICS

TEACHER_REGISTRY: dict[str, dict[str, Any]] = {
    "math": {
        "repo": "nkthebass/tinybrainbot-100m-v3-math",
        "label": "tinybrainbot-100m-v3-math",
        "kind": "tinybrainbot_chat",
        "max_new_tokens": 120,
    },
    "coding": {
        "repo": "SupraLabs/Supra-50M-Instruct",
        "label": "supra-50m-instruct",
        "kind": "hf_causal",
        "max_new_tokens": 160,
    },
    "physics": {
        "repo": "nkthebass/tinybrainbot-100m-v3-math",
        "label": "tinybrainbot-100m-v3-math-interim",
        "kind": "tinybrainbot_chat",
        "max_new_tokens": 140,
        "note": "Interim teacher until moe100m-physics adapter lands",
    },
}

_BENCHMARK_CATEGORIES: dict[str, tuple[str, ...]] = {
    "coding": ("code", "debug", "format"),
    "math": ("math", "direct", "reasoning"),
    "physics": ("physics", "ml", "reasoning"),
}


def distill_jsonl_filename(topic: str) -> str:
    if topic not in DISTILL_TOPICS:
        raise ValueError(f"Unknown distill topic: {topic}")
    return f"distill-{topic}.jsonl"


def merged_distill_filename() -> str:
    return "distill-merged.jsonl"


def distill_catalog(dest_dir: str) -> list[dict[str, Any]]:
    """On-disk distill JSONL status for UI and agent contract."""
    catalog: list[dict[str, Any]] = []
    for topic in DISTILL_TOPICS:
        fname = distill_jsonl_filename(topic)
        path = os.path.join(dest_dir, fname)
        teacher = TEACHER_REGISTRY.get(topic, {})
        entry: dict[str, Any] = {
            "id": topic,
            "label": TOPIC_LABELS.get(topic, topic),
            "file": fname,
            "data_source": f"distill-{topic}",
            "teacher": teacher.get("label") or teacher.get("repo"),
            "teacher_repo": teacher.get("repo"),
            "exists": os.path.isfile(path),
            "rows": 0,
        }
        if entry["exists"]:
            with open(path, encoding="utf-8") as handle:
                entry["rows"] = sum(1 for line in handle if line.strip())
        catalog.append(entry)
    merged_name = merged_distill_filename()
    merged_path = os.path.join(dest_dir, merged_name)
    merged_entry: dict[str, Any] = {
        "id": "merged",
        "label": "Distill merged (coding + physics + math)",
        "file": merged_name,
        "data_source": "distill-merged",
        "exists": os.path.isfile(merged_path),
        "rows": 0,
    }
    if merged_entry["exists"]:
        with open(merged_path, encoding="utf-8") as handle:
            merged_entry["rows"] = sum(1 for line in handle if line.strip())
    catalog.append(merged_entry)
    return catalog


def _tinybrainbot_prompt(user: str) -> str:
    return f"<|user|>\n{user.strip()}\n<|end|>\n<|assistant|>\n"


def _load_hf_causal(repo_id: str, hf_token: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    try:
        tok = AutoTokenizer.from_pretrained(repo_id, token=hf_token, trust_remote_code=True)
    except (ValueError, OSError, ImportError, RuntimeError):
        tok = PreTrainedTokenizerFast.from_pretrained(repo_id, token=hf_token)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except Exception:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            repo_id,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    model.to(device)
    model.eval()
    return model, tok


def _generate_hf_causal(
    model,
    tok,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt").to(device)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(ids[0], skip_special_tokens=True)
    answer = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
    return answer


def _generate_tinybrainbot(model, tok, user: str, *, device: str, max_new_tokens: int) -> str:
    prompt = _tinybrainbot_prompt(user)
    return _generate_hf_causal(model, tok, prompt, device=device, max_new_tokens=max_new_tokens)


def _topic_user_prompts(topic: str, *, benchmark_limit: int = 12) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    cats = _BENCHMARK_CATEGORIES.get(topic, ())
    for row in load_benchmark_prompts(path=BENCHMARK_PATH, limit=0):
        cat = str(row.get("category") or "")
        if cats and cat not in cats:
            continue
        text = str(row.get("prompt") or row.get("question") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        prompts.append({"id": row.get("id"), "user": text, "source": f"benchmark:{row.get('id')}"})
        if benchmark_limit > 0 and len(prompts) >= benchmark_limit:
            break
    for row in expand_topic_seeds(topic):
        user = str(row.get("user") or "").strip()
        if not user or user in seen:
            continue
        seen.add(user)
        prompts.append(
            {
                "id": row.get("seed_id"),
                "user": user,
                "source": row.get("source") or f"seed:{topic}",
            }
        )
    return prompts


def _make_distill_row(
    topic: str,
    *,
    user: str,
    assistant: str,
    teacher: str,
    source: str,
    prompt_id: str | int | None = None,
) -> dict[str, Any]:
    principle = f"Distill · {TOPIC_LABELS.get(topic, topic)} · {teacher}"
    return {
        "principle": principle,
        "user": user.strip(),
        "assistant": assistant.strip(),
        "topic": topic,
        "teacher": teacher,
        "source": source,
        "distill": True,
        "prompt_id": prompt_id,
    }


make_distill_row = _make_distill_row


def generate_topic_distill(
    topic: str,
    dest_dir: str,
    *,
    hf_token: str | None = None,
    device: str | None = None,
    benchmark_limit: int = 12,
    max_rows: int = 80,
) -> dict[str, Any]:
    if topic not in DISTILL_TOPICS:
        raise ValueError(f"Unknown topic: {topic}")
    teacher_cfg = TEACHER_REGISTRY[topic]
    prompts = _topic_user_prompts(topic, benchmark_limit=benchmark_limit)
    if not prompts:
        return {"topic": topic, "kept": 0, "error": "no prompts"}

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    repo = str(teacher_cfg["repo"])
    max_new = int(teacher_cfg.get("max_new_tokens") or 120)
    model, tok = _load_hf_causal(repo, hf_token, device)
    kind = teacher_cfg.get("kind") or "hf_causal"
    generate_fn: Callable[..., str]
    if kind == "tinybrainbot_chat":
        generate_fn = lambda u: _generate_tinybrainbot(model, tok, u, device=device, max_new_tokens=max_new)
    else:
        generate_fn = lambda u: _generate_hf_causal(model, tok, u, device=device, max_new_tokens=max_new)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in prompts[:max_rows]:
        user = str(item["user"])
        try:
            answer = generate_fn(user)
        except Exception as exc:
            errors.append(f"{user[:40]}: {exc}")
            continue
        if not accept_training_text(answer, min_chars=16):
            continue
        rows.append(
            _make_distill_row(
                topic,
                user=user,
                assistant=answer,
                teacher=str(teacher_cfg.get("label") or repo),
                source=str(item.get("source") or "distill"),
                prompt_id=item.get("id"),
            )
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, distill_jsonl_filename(topic))
    with open(dest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "topic": topic,
        "teacher": teacher_cfg.get("label") or repo,
        "prompts": len(prompts),
        "kept": len(rows),
        "path": dest,
        "errors": errors[:5],
    }


def generate_distill_corpus(
    topics: list[str] | None = None,
    dest_dir: str = "data",
    *,
    hf_token: str | None = None,
    device: str | None = None,
    benchmark_limit: int = 12,
    write_merged: bool = True,
) -> dict[str, Any]:
    chosen = [t for t in (topics or list(DISTILL_TOPICS)) if t in DISTILL_TOPICS]
    results: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for topic in chosen:
        result = generate_topic_distill(
            topic,
            dest_dir,
            hf_token=hf_token,
            device=device,
            benchmark_limit=benchmark_limit,
        )
        results.append(result)
        path = result.get("path")
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))

    merged_path = None
    if write_merged and merged:
        merged_path = os.path.join(dest_dir, merged_distill_filename())
        with open(merged_path, "w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "topics": results,
        "merged": merged_path,
        "merged_rows": len(merged),
        "teachers": {t: TEACHER_REGISTRY[t].get("label") for t in chosen},
    }


LIVE_DISTILL_LOG = "distill-live.jsonl"


def live_distill_log_path(dest_dir: str) -> str:
    return os.path.join(dest_dir, LIVE_DISTILL_LOG)


def append_live_distill_row(dest_dir: str, row: dict[str, Any]) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    path = live_distill_log_path(dest_dir)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_distill_live_prompts(
    topics: list[str] | None = None,
    *,
    benchmark_limit: int = 0,
):
    """Cycle topic prompts forever — benchmark_limit 0 = all benchmark rows per topic."""
    import random

    chosen = [t for t in (topics or list(DISTILL_TOPICS)) if t in DISTILL_TOPICS]
    if not chosen:
        raise ValueError("No valid distill topics")
    pools = {t: _topic_user_prompts(t, benchmark_limit=benchmark_limit) for t in chosen}
    for t, prompts in pools.items():
        if not prompts:
            raise ValueError(f"No prompts for distill topic: {t}")
    epoch = 0
    while True:
        epoch += 1
        order = list(chosen)
        random.shuffle(order)
        for topic in order:
            items = list(pools[topic])
            random.shuffle(items)
            for item in items:
                yield epoch, topic, item


class TopicTeacherSession:
    """Load one HF teacher at a time; unload before returning GPU to student."""

    def __init__(self, *, hf_token: str | None, prefer_device: str | None = None) -> None:
        self.hf_token = hf_token
        self.prefer_device = prefer_device
        self._topic: str | None = None
        self._model = None
        self._tok = None
        self._generate_fn: Callable[[str], str] | None = None
        self._label = ""

    def close(self) -> None:
        import torch

        self._topic = None
        self._generate_fn = None
        self._label = ""
        if self._model is not None:
            del self._model
            self._model = None
        self._tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_topic(self, topic: str) -> None:
        if topic not in DISTILL_TOPICS:
            raise ValueError(f"Unknown topic: {topic}")
        if self._topic == topic and self._generate_fn is not None:
            return
        self.close()
        import torch

        device = self.prefer_device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = TEACHER_REGISTRY[topic]
        repo = str(cfg["repo"])
        max_new = int(cfg.get("max_new_tokens") or 120)
        kind = cfg.get("kind") or "hf_causal"
        try:
            model, tok = _load_hf_causal(repo, self.hf_token, device)
        except Exception:
            if device == "cuda":
                model, tok = _load_hf_causal(repo, self.hf_token, "cpu")
                device = "cpu"
            else:
                raise
        if kind == "tinybrainbot_chat":
            generate_fn: Callable[[str], str] = lambda u: _generate_tinybrainbot(
                model, tok, u, device=device, max_new_tokens=max_new
            )
        else:
            generate_fn = lambda u: _generate_hf_causal(
                model, tok, u, device=device, max_new_tokens=max_new
            )
        self._topic = topic
        self._model = model
        self._tok = tok
        self._generate_fn = generate_fn
        self._label = str(cfg.get("label") or repo)

    def generate(self, user: str) -> str:
        if not self._generate_fn:
            raise RuntimeError("Teacher not loaded — call ensure_topic first")
        return self._generate_fn(user)

    @property
    def label(self) -> str:
        return self._label
