"""Training-text quality gates — reject URLs, path spam, and low-signal blobs."""

from __future__ import annotations

import re

_URL = re.compile(r"https?://|www\.\S+", re.I)
_WIKI_ASSET = re.compile(r"upload\.wikimedia|wikipedia/commons|\.svg\b|\.png\b|\.jpg\b", re.I)
_PATH_SPAM = re.compile(
    r"\{name\}/users/|assert_all\(\)|\.index\.html`|/api/v\d+/|\.g\.\d+\.com/",
    re.I,
)
_CODE_ONLY = re.compile(r"^[\s\w\.\{\}\(\)\[\];,:\"'\\/`#\-=<>|&*!@$%^+\d]+$")
_SAPO_META = re.compile(r"cursor_gold=|routing_score=|weights_grounded=|weights_sapo=", re.I)
_SAPO_TAGS = re.compile(r"\[(?:mode|grounded|stage\d+|preclass)[^\]]*\]", re.I)


def is_garbage_training_text(text: str, *, min_chars: int = 80) -> bool:
    """Return True when text should not be used for LM training."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return True

  # URL-only or URL-dominated rows (multimodal_wiki failure mode)
    without_urls = _URL.sub(" ", t)
    if len(without_urls.strip()) < min_chars * 0.35:
        return True
    if _WIKI_ASSET.search(t) and len(without_urls.strip()) < min_chars * 0.5:
        return True

    if _PATH_SPAM.search(t):
        return True

  # Extreme character repetition
    words = t.split()
    if len(words) >= 12:
        from collections import Counter

        top, count = Counter(words).most_common(1)[0]
        if count / len(words) > 0.45 and len(top) > 3:
            return True

  # Mostly non-alphabetic code paths with almost no prose
    alpha = sum(1 for c in t if c.isalpha())
    if alpha < len(t) * 0.18 and _CODE_ONLY.match(t[: min(len(t), 400)]):
        return True

    if _SAPO_META.search(t) and len(t) < 220:
        return True

    return False


def is_sapo_metadata_notes(text: str) -> bool:
    """True when generalization_notes is routing metadata, not training prose."""
    t = (text or "").strip()
    return bool(t) and bool(_SAPO_META.search(t))


def strip_sapo_comment_lines(text: str) -> str:
    """Turn // comment blocks into plain prose lines."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            stripped = stripped[2:].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def truncate_chat_answer(text: str) -> str:
    """Stop generation at the next role boundary (small-model inference guard)."""
    out = (text or "").strip()
    if not out:
        return out
    stops = ("\nUser:", "\nHuman:", "\nAssistant:", "\nSystem:", "\n\nUser", "\n\nHuman")
    cut = len(out)
    for stop in stops:
        idx = out.find(stop)
        if idx > 0:
            cut = min(cut, idx)
    return out[:cut].strip()


def wrap_chat_prompt(user: str, assistant_prefix: str = "") -> str:
    """Standard chat prefix for inference on SFT-trained weights."""
    u = (user or "").strip()
    if not u:
        return "User:\nAssistant:"
    if u.lower().startswith("user:") or u.lower().startswith("system:"):
        return u if "assistant:" in u.lower() else f"{u}\nAssistant:"
    return f"User: {u}\nAssistant:{assistant_prefix}"
