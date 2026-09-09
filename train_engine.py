"""Online CPU trainer: token buffer, micro-batches, checkpoint I/O."""

from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch

from model import ModelConfig, build_model
from scalar_features import SCALAR_DIM, text_scalar_features
from tokenizer_utils import HFTokenizerWrapper, TIKTOKEN_IDS, load_tokenizer, _load_hf_tokenizer
from hf_model_loader import (
    hf_param_count,
    load_hf_from_weights_dir,
    resolve_hf_weights_dir,
)

BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))
DEFAULT_LR = float(os.environ.get("TRAIN_LR", "3e-4"))
BASE_LR = DEFAULT_LR
DEFAULT_WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CUDA_MICRO_BATCH = int(os.environ.get("TRAIN_MICRO_BATCH", "256"))
HF_TRAIN_BLOCK_CAP = int(os.environ.get("HF_TRAIN_BLOCK_CAP", "2048"))


def should_save_model(step: int, every_n: int = 0) -> bool:
    if every_n <= 0 or step <= 0:
        return False
    return step % every_n == 0


def _filter_legacy_state_dict(state: dict) -> dict:
    """Drop non-persistent causal-mask buffers saved by older checkpoints."""
    return {k: v for k, v in state.items() if not k.endswith(".attn.bias")}


@dataclass
class StepResult:
    step: int
    loss: float
    lr: float
    tok_s: float


class TrainEngine:
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int | None = None,
        learning_rate: float | None = None,
        model_config: ModelConfig | None = None,
        weight_decay: float | None = None,
        warmup_steps: int = 0,
        mixed_precision: str = "fp16",
        micro_batch_size: int | None = None,
        grad_accum_steps: int | None = None,
        save_every_n_steps: int = 0,
        lr_schedule: str = "warmup",
        total_train_steps: int = 0,
        tokenizer_id: str = "gpt2",
    ) -> None:
        self.device = device
        self.model_config = model_config or ModelConfig.default()
        self.block_size = self.model_config.block_size
        default_bs = 1024 if device == "cuda" else (32 if device == "mps" else 2)
        self.batch_size = int(batch_size or os.environ.get("TRAIN_BATCH_SIZE") or default_bs)
        self.base_learning_rate = float(learning_rate or os.environ.get("TRAIN_LR") or DEFAULT_LR)
        self.learning_rate = self.base_learning_rate
        self.weight_decay = float(weight_decay if weight_decay is not None else DEFAULT_WEIGHT_DECAY)
        self.warmup_steps = max(0, int(warmup_steps))
        self.mixed_precision = (mixed_precision or "fp16").lower()
        self.save_every_n_steps = max(0, int(save_every_n_steps))
        self.lr_schedule = (lr_schedule or "warmup").lower()
        self.total_train_steps = max(0, int(total_train_steps))
        self._pending_scalar: list[float] | None = None
        if micro_batch_size is not None and micro_batch_size > 0:
            self.micro_batch_size = min(int(micro_batch_size), self.batch_size)
        else:
            self.micro_batch_size = self._resolve_micro_batch(self.batch_size)
        if grad_accum_steps is not None and grad_accum_steps > 0:
            self.grad_accum_steps = int(grad_accum_steps)
        else:
            self.grad_accum_steps = max(
                1, (self.batch_size + self.micro_batch_size - 1) // self.micro_batch_size
            )
        self.model = build_model(device, self.model_config)
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            # GradScaler is for fp16 only — bf16 does not need loss scaling and breaks unscale/update pairing.
            self._scaler = (
                torch.amp.GradScaler("cuda") if self.mixed_precision == "fp16" else None
            )
        else:
            self._scaler = None
        self.enc = None
        self.hf_tokenizer = None
        self.tokenizer_id = "gpt2"
        self.model_backend = "gpt"
        self.hf_repo_id = ""
        self.hf_weights_ref = ""
        self.checkpoint_root = os.environ.get("APP7_CHECKPOINT_DIR", "checkpoints")
        self.session_meta: dict = {}
        self.set_tokenizer(tokenizer_id)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        self.tokens: list[int] = []
        self.step = 0
        self.run_start_step = 0
        self.last_loss = 0.0
        self.total_tokens = 0

    @property
    def param_count(self) -> int:
        if self.model_backend == "hf":
            return hf_param_count(self.model)
        return self.model.param_count()

    @property
    def is_hf_backend(self) -> bool:
        return self.model_backend == "hf"

    @property
    def dropout(self) -> float:
        return self.model_config.dropout

    def _amp_dtype(self) -> torch.dtype:
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        if self.mixed_precision == "fp32":
            return torch.float32
        return torch.float16

    def _apply_warmup_lr(self) -> None:
        base = self.base_learning_rate
        run_step = max(0, self.step - self.run_start_step)
        if self.lr_schedule == "cosine" and self.total_train_steps > self.warmup_steps:
            if self.warmup_steps > 0 and run_step < self.warmup_steps:
                scale = (run_step + 1) / self.warmup_steps
            else:
                denom = max(1, self.total_train_steps - self.warmup_steps)
                progress = min(1.0, (run_step - self.warmup_steps) / denom)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            self.learning_rate = base * scale
        elif self.warmup_steps > 0 and run_step < self.warmup_steps:
            self.learning_rate = base * ((run_step + 1) / self.warmup_steps)
        else:
            self.learning_rate = base
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _resolve_micro_batch(self, batch_size: int) -> int:
        if self.device != "cuda":
            return batch_size
        return min(batch_size, CUDA_MICRO_BATCH)

    def batch_label(self) -> str:
        if self.micro_batch_size < self.batch_size:
            return f"{self.batch_size} (micro={self.micro_batch_size}×{self.grad_accum_steps})"
        return str(self.batch_size)

    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        payload: dict = {
            "step": self.step,
            "run_start_step": self.run_start_step,
            "last_loss": self.last_loss,
            "total_tokens": self.total_tokens,
            "learning_rate": self.learning_rate,
            "base_learning_rate": self.base_learning_rate,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "mixed_precision": self.mixed_precision,
            "save_every_n_steps": self.save_every_n_steps,
            "lr_schedule": self.lr_schedule,
            "total_train_steps": self.total_train_steps,
            "tokens_tail": self.tokens[-self.block_size * self.batch_size * 2 :],
            "tokenizer_id": self.tokenizer_id,
            "session_meta": dict(self.session_meta),
            "model_config": self.model_config.to_dict(),
            "model_backend": self.model_backend,
        }
        if self.model_backend == "hf":
            weights_dir = resolve_hf_weights_dir(self.checkpoint_root, self.hf_weights_ref)
            os.makedirs(weights_dir, exist_ok=True)
            self.model.save_pretrained(weights_dir, safe_serialization=True)
            opt_path = os.path.join(weights_dir, "train_optimizer.pt")
            torch.save(self.optimizer.state_dict(), opt_path)
            payload.update(
                {
                    "hf_repo_id": self.hf_repo_id,
                    "hf_weights_dir": self.hf_weights_ref,
                    "optimizer_path": "train_optimizer.pt",
                }
            )
        else:
            payload["model"] = self.model.state_dict()
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, buf)
        return buf.getvalue()

    def _apply_checkpoint_meta(self, data: dict) -> None:
        self.step = int(data.get("step") or 0)
        self.run_start_step = int(data.get("run_start_step") or 0)
        self.last_loss = float(data.get("last_loss") or 0.0)
        self.total_tokens = int(data.get("total_tokens") or 0)
        self.learning_rate = float(data.get("learning_rate") or self.learning_rate)
        self.base_learning_rate = float(
            data.get("base_learning_rate") or data.get("learning_rate") or self.learning_rate
        )
        self.batch_size = int(data.get("batch_size") or self.batch_size)
        self.micro_batch_size = int(data.get("micro_batch_size") or self.micro_batch_size)
        self.grad_accum_steps = int(data.get("grad_accum_steps") or self.grad_accum_steps)
        self.weight_decay = float(data.get("weight_decay") or self.weight_decay)
        self.warmup_steps = int(data.get("warmup_steps") or self.warmup_steps)
        self.mixed_precision = str(data.get("mixed_precision") or self.mixed_precision)
        self.save_every_n_steps = int(data.get("save_every_n_steps") or self.save_every_n_steps)
        self.lr_schedule = str(data.get("lr_schedule") or self.lr_schedule)
        self.total_train_steps = int(data.get("total_train_steps") or self.total_train_steps)
        self.tokens = list(data.get("tokens_tail") or [])
        self.session_meta = dict(data.get("session_meta") or {})
        if "micro_batch_size" in data:
            self.micro_batch_size = int(data.get("micro_batch_size") or self.micro_batch_size)
        else:
            self.micro_batch_size = self._resolve_micro_batch(self.batch_size)
        if "grad_accum_steps" in data:
            self.grad_accum_steps = int(data.get("grad_accum_steps") or self.grad_accum_steps)
        else:
            self.grad_accum_steps = max(
                1, (self.batch_size + self.micro_batch_size - 1) // self.micro_batch_size
            )
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _load_hf_checkpoint(self, data: dict) -> None:
        saved_cfg = ModelConfig.from_dict(data.get("model_config") or {})
        saved_cfg.scalar_input_projection = False
        cap = int(os.environ.get("HF_TRAIN_BLOCK_CAP", "2048"))
        if cap > 0 and saved_cfg.block_size > cap:
            saved_cfg.block_size = cap
        self.model_backend = "hf"
        self.model_config = saved_cfg
        self.block_size = saved_cfg.block_size
        self.hf_repo_id = str(data.get("hf_repo_id") or data.get("imported_from") or "")
        self.hf_weights_ref = str(data.get("hf_weights_dir") or "")
        weights_dir = resolve_hf_weights_dir(self.checkpoint_root, self.hf_weights_ref)
        self.model, _ = load_hf_from_weights_dir(
            weights_dir,
            device=self.device,
            block_size=saved_cfg.block_size,
        )
        if self.device == "cuda":
            self._scaler = (
                torch.amp.GradScaler("cuda") if self.mixed_precision == "fp16" else None
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        opt_rel = data.get("optimizer_path")
        if opt_rel:
            opt_path = os.path.join(weights_dir, str(opt_rel))
            if os.path.isfile(opt_path):
                try:
                    self.optimizer.load_state_dict(
                        torch.load(opt_path, map_location=self.device, weights_only=False)
                    )
                except Exception:
                    pass
        elif data.get("optimizer"):
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except Exception:
                pass
        self._apply_checkpoint_meta(data)
        tid = data.get("tokenizer_id") or self.hf_repo_id or "gpt2"
        self.set_tokenizer(str(tid))

    def load_state_bytes(self, payload: bytes, *, checkpoint_root: str | None = None) -> None:
        if checkpoint_root:
            self.checkpoint_root = checkpoint_root
        data = torch.load(io.BytesIO(payload), map_location=self.device, weights_only=False)
        if isinstance(data, dict) and data.get("model_backend") == "hf":
            self._load_hf_checkpoint(data)
            return
        if isinstance(data, dict) and "model" in data:
            self.model_backend = "gpt"
            saved_cfg = ModelConfig.from_dict(data.get("model_config"))
            if saved_cfg.to_dict() != self.model_config.to_dict():
                self.model_config = saved_cfg
                self.block_size = saved_cfg.block_size
                self.model = build_model(self.device, saved_cfg)
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                )
            self.model.load_state_dict(_filter_legacy_state_dict(data["model"]), strict=False)
            if "optimizer" in data:
                try:
                    self.optimizer.load_state_dict(data["optimizer"])
                except Exception:
                    pass
            self._apply_checkpoint_meta(data)
            tid = data.get("tokenizer_id")
            if not tid:
                tid = (
                    "SupraLabs/StorySupra-10M"
                    if saved_cfg.vocab_size <= 8192
                    else "gpt2"
                )
            self.set_tokenizer(str(tid))
        else:
            state = data if isinstance(data, dict) else data
            self.model_backend = "gpt"
            self.model.load_state_dict(_filter_legacy_state_dict(state), strict=False)
            self.step = 0
            self.run_start_step = 0
            self.last_loss = 0.0
            self.total_tokens = 0
            self.tokens = []
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

    def set_tokenizer(self, tokenizer_id: str) -> None:
        tid = (tokenizer_id or "gpt2").strip()
        if self.enc is not None and tid == self.tokenizer_id:
            return
        self.tokenizer_id = tid
        if tid in TIKTOKEN_IDS:
            self.hf_tokenizer = None
        else:
            self.hf_tokenizer = _load_hf_tokenizer(tid)
            self.enc = HFTokenizerWrapper(self.hf_tokenizer)
            return
        self.enc = load_tokenizer(tid)

    def feed_text(
        self,
        text: str,
        scalars: list[float] | None = None,
    ) -> list[StepResult]:
        if self.model_config.scalar_input_projection:
            self._pending_scalar = scalars or text_scalar_features(
                text, dim=self.model_config.scalar_dim
            )
        else:
            self._pending_scalar = None
        ids = self.enc.encode(text)
        if not ids:
            return []
        self.tokens.extend(ids)
        out: list[StepResult] = []
        need = self.block_size * self.batch_size + 1
        while len(self.tokens) >= need:
            out.append(self._train_step())
        return out

    def _optimizer_step(self) -> None:
        """Apply clipped optimizer step; always finalize GradScaler state when used."""
        if self._scaler is not None:
            try:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self._scaler.step(self.optimizer)
            finally:
                self._scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()

    def _train_step(self) -> StepResult:
        if self.model_backend == "hf":
            return self._train_step_hf()
        need = self.block_size * self.batch_size + 1
        chunk = self.tokens[:need]
        self.tokens = self.tokens[need - 1 :]
        x_rows = []
        y_rows = []
        for i in range(self.batch_size):
            start = i * self.block_size
            seq = chunk[start : start + self.block_size + 1]
            x_rows.append(seq[:-1])
            y_rows.append(seq[1:])
        x = torch.tensor(x_rows, dtype=torch.long, device=self.device)
        y = torch.tensor(y_rows, dtype=torch.long, device=self.device)
        scalar_t = None
        if self.model_config.scalar_input_projection and self._pending_scalar is not None:
            dim = self.model_config.scalar_dim
            vec = (self._pending_scalar + [0.0] * dim)[:dim]
            scalar_t = torch.tensor([vec] * self.batch_size, dtype=torch.float32, device=self.device)
        self._apply_warmup_lr()
        t0 = time.time()
        self.model.train()
        amp_dtype = self._amp_dtype()
        if self.device == "cuda":
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            micro_batches = 0
            for start in range(0, self.batch_size, self.micro_batch_size):
                end = min(start + self.micro_batch_size, self.batch_size)
                xb = x[start:end]
                yb = y[start:end]
                if amp_dtype == torch.float32:
                    _, loss = self.model(xb, yb, scalars=scalar_t[start:end] if scalar_t is not None else None)
                    scaled = loss / self.grad_accum_steps
                    scaled.backward()
                else:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, loss = self.model(
                            xb,
                            yb,
                            scalars=scalar_t[start:end] if scalar_t is not None else None,
                        )
                    scaled = loss / self.grad_accum_steps
                    if self._scaler is not None:
                        self._scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                loss_sum += float(loss.item())
                micro_batches += 1
            self._optimizer_step()
            self.last_loss = loss_sum / max(micro_batches, 1)
        else:
            _, loss = self.model(x, y, scalars=scalar_t)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            self.last_loss = float(loss.item())
        self.step += 1
        self.total_tokens += self.block_size * self.batch_size
        dt = max(time.time() - t0, 1e-6)
        tok_s = (self.block_size * self.batch_size) / dt
        lr = float(self.optimizer.param_groups[0]["lr"])
        return StepResult(step=self.step, loss=self.last_loss, lr=lr, tok_s=tok_s)

    def _train_step_hf(self) -> StepResult:
        need = self.block_size * self.batch_size + 1
        chunk = self.tokens[:need]
        self.tokens = self.tokens[need - 1 :]
        x_rows = []
        y_rows = []
        for i in range(self.batch_size):
            start = i * self.block_size
            seq = chunk[start : start + self.block_size + 1]
            x_rows.append(seq[:-1])
            y_rows.append(seq[1:])
        x = torch.tensor(x_rows, dtype=torch.long, device=self.device)
        y = torch.tensor(y_rows, dtype=torch.long, device=self.device)
        self._apply_warmup_lr()
        t0 = time.time()
        self.model.train()
        amp_dtype = self._amp_dtype()
        if self.device == "cuda":
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            micro_batches = 0
            for start in range(0, self.batch_size, self.micro_batch_size):
                end = min(start + self.micro_batch_size, self.batch_size)
                xb = x[start:end]
                yb = y[start:end]
                if amp_dtype == torch.float32:
                    out = self.model(input_ids=xb, labels=yb)
                    loss = out.loss
                    scaled = loss / self.grad_accum_steps
                    scaled.backward()
                else:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = self.model(input_ids=xb, labels=yb)
                        loss = out.loss
                    scaled = loss / self.grad_accum_steps
                    if self._scaler is not None:
                        self._scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                loss_sum += float(loss.item())
                micro_batches += 1
            self._optimizer_step()
            self.last_loss = loss_sum / max(micro_batches, 1)
        else:
            out = self.model(input_ids=x, labels=y)
            loss = out.loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            self.last_loss = float(loss.item())
        self.step += 1
        self.total_tokens += self.block_size * self.batch_size
        dt = max(time.time() - t0, 1e-6)
        tok_s = (self.block_size * self.batch_size) / dt
        lr = float(self.optimizer.param_groups[0]["lr"])
        return StepResult(step=self.step, loss=self.last_loss, lr=lr, tok_s=tok_s)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
    ) -> str:
        self.model.eval()
        if self.model_backend == "hf" and self.hf_tokenizer is not None:
            inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs.pop("token_type_ids", None)
            gen_kwargs: dict = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 0.01),
                "pad_token_id": self.hf_tokenizer.eos_token_id or self.hf_tokenizer.pad_token_id,
            }
            if repetition_penalty and repetition_penalty > 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            ids = self.model.generate(**inputs, **gen_kwargs)
            text = self.hf_tokenizer.decode(ids[0], skip_special_tokens=True)
            return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
        ids = self.enc.encode(prompt)
        if not ids:
            ids = [0]
        idx = torch.tensor([ids[-self.block_size :]], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        new_ids = out[0].tolist()[len(ids) :]
        return self.enc.decode(new_ids).strip()
