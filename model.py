"""Configurable GPT decoder for stage-1 pretrain (+ L2EVAL-style extensions)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 8192  # StorySupra 8K — 10M floor default
BLOCK_SIZE = 256
N_LAYER = 8
N_HEAD = 8
N_EMBD = 256
DROPOUT = 0.1
SCALAR_DIM = 12
GPT2_VOCAB_SIZE = 50257
DEFAULT_FFN_DIM_10M = 1024


@dataclass
class ModelConfig:
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    block_size: int = BLOCK_SIZE
    vocab_size: int = VOCAB_SIZE
    dropout: float = DROPOUT
    ffn_dim: int | None = None
    use_flash_attention: bool = False
    scalar_input_projection: bool = False
    scalar_dim: int = SCALAR_DIM
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.n_layer = int(self.n_layer)
        self.n_head = int(self.n_head)
        self.n_embd = int(self.n_embd)
        self.block_size = int(self.block_size)
        self.vocab_size = int(self.vocab_size)
        self.scalar_dim = int(self.scalar_dim)
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.n_embd
        else:
            self.ffn_dim = int(self.ffn_dim)
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.scalar_dim < 1:
            raise ValueError("scalar_dim must be >= 1")

    @classmethod
    def default(cls) -> ModelConfig:
        return cls(ffn_dim=DEFAULT_FFN_DIM_10M)

    @classmethod
    def footprint_10m(cls) -> ModelConfig:
        """8L×256d · StorySupra 8K — ladder floor (~12.6M params)."""
        return cls(
            n_layer=8,
            n_head=8,
            n_embd=256,
            block_size=256,
            vocab_size=8192,
            ffn_dim=DEFAULT_FFN_DIM_10M,
        )

    @classmethod
    def footprint_25m(cls) -> ModelConfig:
        """10L×336d · StorySupra 8K — distill student target (~25.4M params)."""
        return cls(
            n_layer=10,
            n_head=8,
            n_embd=336,
            block_size=256,
            vocab_size=8192,
            ffn_dim=1344,
        )

    @classmethod
    def footprint_50m(cls) -> ModelConfig:
        """8L×512d · gpt2 vocab — scale-up target (~68M params)."""
        return cls(
            n_layer=8,
            n_head=8,
            n_embd=512,
            block_size=256,
            vocab_size=GPT2_VOCAB_SIZE,
            ffn_dim=2048,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> ModelConfig:
        if not data:
            return cls.default()
        return cls(
            n_layer=int(data.get("n_layer") or N_LAYER),
            n_head=int(data.get("n_head") or N_HEAD),
            n_embd=int(data.get("n_embd") or N_EMBD),
            block_size=int(data.get("block_size") or BLOCK_SIZE),
            vocab_size=int(data.get("vocab_size") or VOCAB_SIZE),
            dropout=float(data.get("dropout") or DROPOUT),
            ffn_dim=int(data["ffn_dim"]) if data.get("ffn_dim") is not None else None,
            use_flash_attention=bool(data.get("use_flash_attention")),
            scalar_input_projection=bool(data.get("scalar_input_projection")),
            scalar_dim=int(data.get("scalar_dim") or SCALAR_DIM),
            gradient_checkpointing=bool(data.get("gradient_checkpointing")),
        )


def estimate_params(cfg: ModelConfig) -> int:
    ffn = cfg.ffn_dim or (4 * cfg.n_embd)
    embed = cfg.vocab_size * cfg.n_embd + cfg.block_size * cfg.n_embd
    per_block = 12 * cfg.n_embd * cfg.n_embd + 2 * cfg.n_embd * ffn
    scalar = cfg.scalar_dim * cfg.n_embd if cfg.scalar_input_projection else 0
    return embed + cfg.n_layer * per_block + 2 * cfg.n_embd + scalar


def flash_attention_available() -> bool:
    return hasattr(F, "scaled_dot_product_attention")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.cfg.n_embd, dim=2)
        head_dim = c // self.cfg.n_head
        k = k.view(b, t, self.cfg.n_head, head_dim).transpose(1, 2)
        q = q.view(b, t, self.cfg.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_head, head_dim).transpose(1, 2)

        use_flash = (
            self.cfg.use_flash_attention
            and x.is_cuda
            and flash_attention_available()
        )
        if use_flash:
            dropout_p = self.cfg.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        ffn = cfg.ffn_dim or (4 * cfg.n_embd)
        self.c_fc = nn.Linear(cfg.n_embd, ffn)
        self.c_proj = nn.Linear(ffn, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig.default()
        c = self.cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(c.vocab_size, c.n_embd),
                wpe=nn.Embedding(c.block_size, c.n_embd),
                drop=nn.Dropout(c.dropout),
                h=nn.ModuleList(Block(c) for _ in range(c.n_layer)),
                ln_f=nn.LayerNorm(c.n_embd),
            )
        )
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.scalar_proj = (
            nn.Linear(c.scalar_dim, c.n_embd, bias=False)
            if c.scalar_input_projection
            else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        scalars: torch.Tensor | None = None,
    ):
        b, t = idx.size()
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} > block size {self.cfg.block_size}")
        pos = torch.arange(0, t, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        if self.scalar_proj is not None and scalars is not None:
            if scalars.dim() == 2:
                x = x + self.scalar_proj(scalars).unsqueeze(1)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        penalty = max(float(repetition_penalty or 1.0), 1.0)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond, scalars=scalars)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if penalty > 1.0:
                for token_id in set(idx[0].tolist()):
                    score = logits[0, token_id]
                    logits[0, token_id] = score / penalty if score < 0 else score * penalty
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(device: str = "cpu", cfg: ModelConfig | None = None) -> GPT:
    model = GPT(cfg)
    model.to(device)
    return model
