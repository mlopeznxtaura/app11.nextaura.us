"""Import external checkpoints into app7 .pt manifest format."""

from __future__ import annotations

import io
import os
from typing import Any

import torch

from hf_model_loader import (
    hf_param_count,
    hf_weights_slug,
    load_hf_causal_lm,
)
from model import GPT, ModelConfig

_GPT2_KEY_MAP = {
    "transformer.wte.weight": "transformer.wte.weight",
    "transformer.wpe.weight": "transformer.wpe.weight",
    "transformer.ln_f.weight": "transformer.ln_f.weight",
    "transformer.ln_f.bias": "transformer.ln_f.bias",
    "lm_head.weight": "lm_head.weight",
}


_CONV1D_SUFFIXES = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def _map_block_key(hf_key: str) -> str | None:
    if hf_key.startswith("transformer.h."):
        mapped = hf_key.replace(".ln_1.", ".ln1.").replace(".ln_2.", ".ln2.")
        # HF mask bias buffers — not used by our attention
        if mapped.endswith(".attn.bias"):
            return None
        return mapped
    return _GPT2_KEY_MAP.get(hf_key)


def _maybe_transpose_conv1d(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """HF GPT-2 Conv1D weights are (in, out); nn.Linear expects (out, in)."""
    if tensor.dim() == 2 and any(key.endswith(suffix) for suffix in _CONV1D_SUFFIXES):
        return tensor.t().contiguous()
    return tensor


def hf_gpt2_state_dict(hf_state: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Map HuggingFace GPT-2 state dict keys onto our GPT module."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state.items():
        mapped = _map_block_key(key)
        if mapped is None:
            continue
        t = _maybe_transpose_conv1d(mapped, tensor)
        if mapped == "transformer.wpe.weight" and t.size(0) > cfg.block_size:
            t = t[: cfg.block_size].contiguous()
        out[mapped] = t
    # tied embeddings
    if "transformer.wte.weight" in out and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["transformer.wte.weight"]
    return out


def import_hf_gpt2(
    repo_id: str = "gpt2",
    *,
    token: str | None = None,
    n_layer: int | None = None,
    n_head: int | None = None,
    n_embd: int | None = None,
    block_size: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(repo_id, token=token)
    hcfg = hf.config
    cfg = ModelConfig(
        n_layer=int(n_layer or hcfg.n_layer),
        n_head=int(n_head or hcfg.n_head),
        n_embd=int(n_embd or hcfg.n_embd),
        block_size=int(block_size or hcfg.n_positions),
        vocab_size=int(hcfg.vocab_size),
        dropout=float(hcfg.resid_pdrop),
    )
    model = GPT(cfg)
    mapped = hf_gpt2_state_dict(hf.state_dict(), cfg)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    buf = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {},
            "model_config": cfg.to_dict(),
            "step": 1,
            "last_loss": 0.0,
            "total_tokens": 0,
            "learning_rate": 3e-4,
            "base_learning_rate": 3e-4,
            "batch_size": 1024,
            "imported_from": repo_id,
        },
        buf,
    )
    meta = {
        "repo_id": repo_id,
        "model_config": cfg.to_dict(),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "param_count": model.param_count(),
    }
    return buf.getvalue(), meta


def import_hf_causal(
    repo_id: str,
    *,
    weights_root: str,
    token: str | None = None,
    block_size: int | None = None,
    device: str = "cpu",
) -> tuple[bytes, dict[str, Any]]:
    """Load any HF causal LM and write a small manifest + on-disk weights."""
    slug = hf_weights_slug(repo_id)
    weights_subdir = f"hf-{slug}"
    weights_dir = os.path.join(weights_root, weights_subdir)
    os.makedirs(weights_dir, exist_ok=True)

    model, cfg = load_hf_causal_lm(
        repo_id,
        token=token,
        device=device,
        block_size=block_size,
    )
    model.save_pretrained(weights_dir, safe_serialization=True)
    try:
        from transformers import AutoTokenizer

        src = repo_id if not os.path.isdir(repo_id) else os.path.abspath(repo_id)
        tok = AutoTokenizer.from_pretrained(src, token=token, trust_remote_code=True)
        tok.save_pretrained(weights_dir)
    except Exception:
        pass

    meta = {
        "repo_id": repo_id,
        "model_backend": "hf",
        "model_config": cfg.to_dict(),
        "hf_weights_dir": weights_subdir,
        "param_count": hf_param_count(model),
        "architecture": f"{cfg.n_layer}L×{cfg.n_embd}d",
        "block_size": cfg.block_size,
        "vocab_size": cfg.vocab_size,
    }
    buf = io.BytesIO()
    torch.save(
        {
            "model_backend": "hf",
            "hf_repo_id": repo_id,
            "hf_weights_dir": weights_subdir,
            "model_config": cfg.to_dict(),
            "step": 1,
            "run_start_step": 0,
            "last_loss": 0.0,
            "total_tokens": 0,
            "learning_rate": 1e-5,
            "base_learning_rate": 1e-5,
            "batch_size": 512,
            "micro_batch_size": 1,
            "grad_accum_steps": 512,
            "mixed_precision": "bf16" if device == "cuda" else "fp32",
            "tokenizer_id": repo_id,
            "imported_from": repo_id,
            "session_meta": {"model_tag": slug.split("-")[0][:32]},
        },
        buf,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return buf.getvalue(), meta
