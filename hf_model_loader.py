"""Load / save HuggingFace causal LMs for native finetuning (Qwen, Llama, Mistral, …)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from model import ModelConfig


def _resolve_repo_src(repo_id: str) -> tuple[str, bool]:
    local = Path(repo_id)
    if local.is_dir():
        return str(local.resolve()), True
    return repo_id.strip(), False


def hf_config_to_model_config(hf_cfg: Any, block_size: int | None = None) -> ModelConfig:
    n_embd = int(
        getattr(hf_cfg, "hidden_size", None)
        or getattr(hf_cfg, "n_embd", None)
        or 768
    )
    n_layer = int(
        getattr(hf_cfg, "num_hidden_layers", None)
        or getattr(hf_cfg, "n_layer", None)
        or 12
    )
    n_head = int(
        getattr(hf_cfg, "num_attention_heads", None)
        or getattr(hf_cfg, "num_heads", None)
        or getattr(hf_cfg, "n_head", None)
        or max(1, n_embd // 64)
    )
    vocab_size = int(getattr(hf_cfg, "vocab_size", 50257))
    max_pos = int(
        getattr(hf_cfg, "max_position_embeddings", None)
        or getattr(hf_cfg, "max_seq_len", None)
        or getattr(hf_cfg, "seq_length", None)
        or 2048
    )
    bs = int(block_size or max_pos)
    bs = min(bs, max_pos)
    cap = int(os.environ.get("HF_TRAIN_BLOCK_CAP", "2048"))
    if cap > 0:
        bs = min(bs, cap)
    ffn = int(getattr(hf_cfg, "intermediate_size", None) or 4 * n_embd)
    return ModelConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=bs,
        vocab_size=vocab_size,
        ffn_dim=ffn,
        scalar_input_projection=False,
        gradient_checkpointing=True,
    )


def _pick_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_hf_causal_lm(
    repo_id: str,
    *,
    token: str | None = None,
    device: str = "cpu",
    torch_dtype: torch.dtype | None = None,
    block_size: int | None = None,
):
    from transformers import AutoModelForCausalLM

    src, local_only = _resolve_repo_src(repo_id)
    dtype = torch_dtype or _pick_dtype(device)
    kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    device_map = os.environ.get("HF_DEVICE_MAP", "").strip()
    if device_map:
        kwargs["device_map"] = device_map
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
    if not device_map:
        model.to(device)
    cfg = hf_config_to_model_config(model.config, block_size)
    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    model.train()
    return model, cfg


def load_hf_from_weights_dir(
    weights_dir: str,
    *,
    token: str | None = None,
    device: str = "cpu",
    torch_dtype: torch.dtype | None = None,
    block_size: int | None = None,
):
    from transformers import AutoModelForCausalLM

    path = str(Path(weights_dir).resolve())
    dtype = torch_dtype or _pick_dtype(device)
    kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    device_map = os.environ.get("HF_DEVICE_MAP", "").strip()
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not device_map:
        model.to(device)
    cfg = hf_config_to_model_config(model.config, block_size)
    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    model.train()
    return model, cfg


def hf_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_hf_weights_dir(checkpoint_root: str, weights_ref: str) -> str:
    ref = (weights_ref or "").strip()
    if not ref:
        raise ValueError("hf_weights_dir missing from checkpoint manifest")
    if os.path.isabs(ref):
        return ref
    return str(Path(checkpoint_root).joinpath(ref).resolve())


def hf_weights_slug(repo_id: str) -> str:
    slug = repo_id.replace("\\", "/").strip("/").replace("/", "-")
    slug = "".join(c if c.isalnum() or c in "-_." else "-" for c in slug)
    return slug[:120] or "hf-model"
