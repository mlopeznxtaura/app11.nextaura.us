"""Configurable tokenizers: tiktoken encodings or HuggingFace AutoTokenizer."""

from __future__ import annotations

from typing import Any, Protocol

TIKTOKEN_IDS = frozenset({"gpt2", "cl100k_base", "p50k_base", "r50k_base"})

TOKENIZER_GUIDE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "id": "floor-10m",
        "footprint_label": "~10M floor",
        "params_lo": 8_000_000,
        "params_hi": 14_000_000,
        "vocab_size": 8192,
        "tokenizer_id": "SupraLabs/StorySupra-10M",
        "tokenizer_label": "StorySupra 8K",
        "arch_hint": "8L×256d · ffn 1024",
        "rule": "embed ≤25% params · benchmark vs StorySupra-10M",
    },
    {
        "id": "gpt2-min",
        "footprint_label": "~17M (gpt2 min)",
        "params_lo": 14_000_000,
        "params_hi": 22_000_000,
        "vocab_size": 50257,
        "tokenizer_id": "gpt2",
        "tokenizer_label": "gpt2 (tiktoken)",
        "arch_hint": "8L×192d · vocab auto",
        "rule": "embed-heavy (~75%) · smoke only, not a true 10M",
    },
    {
        "id": "prod-50m",
        "footprint_label": "~50M default",
        "params_lo": 40_000_000,
        "params_hi": 70_000_000,
        "vocab_size": 50257,
        "tokenizer_id": "gpt2",
        "tokenizer_label": "gpt2 (tiktoken)",
        "arch_hint": "8L×512d · mixed streams",
        "rule": "default pretrain · embed ~25% at 512d",
    },
    {
        "id": "gpt2-124m",
        "footprint_label": "GPT-2 124M",
        "params_lo": 100_000_000,
        "params_hi": 150_000_000,
        "vocab_size": 50257,
        "tokenizer_id": "gpt2",
        "tokenizer_label": "gpt2 (tiktoken)",
        "arch_hint": "12L×768d",
        "rule": "embed ~31% · standard English pretrain",
    },
    {
        "id": "finetune",
        "footprint_label": "Fine-tune / SFT",
        "params_lo": 0,
        "params_hi": 10_000_000_000,
        "vocab_size": 0,
        "tokenizer_id": "",
        "tokenizer_label": "match checkpoint",
        "arch_hint": "load .pt first",
        "rule": "tokenizer + vocab MUST match pretrain ckpt — never swap mid-run",
    },
)


def embedding_fraction(vocab_size: int, n_embd: int, model_params: int) -> float:
    """Share of params in token + position embeddings (wte + wpe proxy)."""
    if model_params <= 0:
        return 0.0
    embed = int(vocab_size) * int(n_embd)
    return embed / model_params


def recommend_tokenizer(
    *,
    model_params: int,
    n_embd: int,
    vocab_size: int = 50257,
    mode: str = "pretrain",
    checkpoint_vocab: int | None = None,
    checkpoint_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Pick tokenizer + vocab from arch footprint and training mode."""
    params = max(1, int(model_params))
    dim = max(1, int(n_embd))
    vocab = int(vocab_size)
    finetune = mode == "finetune"

    if finetune and (checkpoint_vocab or checkpoint_tokenizer):
        ck_vocab = int(checkpoint_vocab or vocab)
        ck_tok = (checkpoint_tokenizer or "gpt2").strip()
        return {
            "tokenizer_id": ck_tok,
            "vocab_size": ck_vocab,
            "tokenizer_label": ck_tok,
            "reason": "finetune: locked to loaded checkpoint tokenizer",
            "embed_pct": round(embedding_fraction(ck_vocab, dim, params) * 100, 1),
            "warning": "",
            "row_id": "finetune",
        }

    embed_pct = embedding_fraction(vocab, dim, params)
    if embed_pct > 0.30 and params < 30_000_000:
        row = TOKENIZER_GUIDE_ROWS[0]
        return {
            "tokenizer_id": row["tokenizer_id"],
            "vocab_size": row["vocab_size"],
            "tokenizer_label": row["tokenizer_label"],
            "reason": f"embed {embed_pct:.0%} of params on sub-30M — use 8K vocab",
            "embed_pct": round(embedding_fraction(row["vocab_size"], dim, params) * 100, 1),
            "warning": "gpt2 vocab wastes params on small models",
            "row_id": row["id"],
        }

    if params <= 14_000_000 and embed_pct > 0.25:
        row = TOKENIZER_GUIDE_ROWS[0]
        return {
            "tokenizer_id": row["tokenizer_id"],
            "vocab_size": row["vocab_size"],
            "tokenizer_label": row["tokenizer_label"],
            "reason": f"true ~10M footprint · embed {embed_pct:.0%} with gpt2",
            "embed_pct": round(embedding_fraction(row["vocab_size"], dim, params) * 100, 1),
            "warning": "gpt2 vocab wastes params on small models",
            "row_id": row["id"],
        }

    for row in TOKENIZER_GUIDE_ROWS:
        if row["id"] == "finetune":
            continue
        if row["params_lo"] <= params <= row["params_hi"]:
            rec_vocab = int(row["vocab_size"])
            return {
                "tokenizer_id": row["tokenizer_id"],
                "vocab_size": rec_vocab,
                "tokenizer_label": row["tokenizer_label"],
                "reason": row["rule"],
                "embed_pct": round(embedding_fraction(rec_vocab, dim, params) * 100, 1),
                "warning": "",
                "row_id": row["id"],
            }

    row = TOKENIZER_GUIDE_ROWS[2]
    return {
        "tokenizer_id": row["tokenizer_id"],
        "vocab_size": row["vocab_size"],
        "tokenizer_label": row["tokenizer_label"],
        "reason": "default gpt2 for mid-size pretrain",
        "embed_pct": round(embedding_fraction(row["vocab_size"], dim, params) * 100, 1),
        "warning": "",
        "row_id": row["id"],
    }


def tokenizer_guidance_bundle(
    model_params: int,
    *,
    n_embd: int = 512,
    vocab_size: int = 50257,
    mode: str = "pretrain",
    checkpoint_vocab: int | None = None,
    checkpoint_tokenizer: str | None = None,
) -> dict[str, Any]:
    rec = recommend_tokenizer(
        model_params=model_params,
        n_embd=n_embd,
        vocab_size=vocab_size,
        mode=mode,
        checkpoint_vocab=checkpoint_vocab,
        checkpoint_tokenizer=checkpoint_tokenizer,
    )
    rows = []
    for row in TOKENIZER_GUIDE_ROWS:
        if row["id"] == "finetune" and mode != "finetune":
            continue
        vocab = int(row["vocab_size"]) if row["vocab_size"] else rec["vocab_size"]
        rows.append(
            {
                **row,
                "display_vocab": row["vocab_size"] or "match ckpt",
                "embed_pct": round(embedding_fraction(vocab, n_embd, model_params) * 100, 1)
                if row["vocab_size"]
                else None,
                "active": row["id"] == rec.get("row_id"),
            }
        )
    return {
        "law": (
            "vocab_size must equal tokenizer vocab · embed% = vocab×dim / params · "
            "if embed >30% on sub-20M → 8K vocab · finetune → match checkpoint"
        ),
        "rows": rows,
        "recommendation": rec,
    }


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class TiktokenWrapper:
    def __init__(self, encoding) -> None:
        self._enc = encoding

    def encode(self, text: str) -> list[int]:
        # Training corpora may mention special token strings literally (e.g. <|endoftext|>).
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class HFTokenizerWrapper:
    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


def _load_hf_tokenizer(repo_id: str):
    """Load HF tokenizer; fall back to tokenizer.json when hub class is too new."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        msg = str(exc)
        if "TokenizersBackend" in msg or "does not exist" in msg:
            return PreTrainedTokenizerFast.from_pretrained(repo_id)
        raise


def load_tokenizer(tokenizer_id: str) -> TextTokenizer:
    tid = (tokenizer_id or "gpt2").strip()
    if tid in TIKTOKEN_IDS:
        import tiktoken

        return TiktokenWrapper(tiktoken.get_encoding(tid))
    return HFTokenizerWrapper(_load_hf_tokenizer(tid))
