#!/usr/bin/env python3
"""Muse infer sidecar — 4-bit generate on leftover VRAM. Does not touch sft_qlora.py."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/datasets")

import torch
from fastapi import FastAPI
from pydantic import BaseModel

BASE = os.environ.get("MUSE_BASE", "/data/models/Muse-Glimmer-30B")
V1 = Path(os.environ.get("MUSE_SFT_V1", "/data/muse/sft-v1/final"))
V2 = Path(os.environ.get("MUSE_OUT", "/data/muse/sft-v2"))


def pick_adapter() -> tuple[str, str]:
    ckpts = []
    if V2.is_dir():
        for p in V2.glob("checkpoint-*"):
            n = p.name.split("-")[-1]
            if n.isdigit() and (p / "adapter_model.safetensors").is_file():
                ckpts.append((int(n), p))
        final = V2 / "final"
        if (final / "adapter_model.safetensors").is_file():
            return str(final), "sft-v2/final"
    if ckpts:
        step, path = max(ckpts)
        return str(path), f"sft-v2/checkpoint-{step}"
    if (V1 / "adapter_model.safetensors").is_file():
        return str(V1), "sft-v1/final"
    return "", "none"


ADAPTER, ADAPTER_LABEL = pick_adapter()
READY = False
LOAD_ERROR = ""
MODEL = None
TOKENIZER = None
LOADED_AT = 0.0

app = FastAPI()


class AskIn(BaseModel):
    question: str
    max_new_tokens: int = 128
    temperature: float = 0.35


@app.get("/health")
def health():
    return {
        "ok": READY,
        "ready": READY,
        "error": LOAD_ERROR or None,
        "adapter": ADAPTER_LABEL,
        "adapter_path": ADAPTER,
        "device": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0,
    }


@app.post("/ask")
def ask(body: AskIn):
    if not READY or MODEL is None or TOKENIZER is None:
        return {"error": LOAD_ERROR or "Muse infer sidecar is still loading."}
    q = (body.question or "").strip()
    if not q:
        return {"error": "Type a question first."}
    t0 = time.time()
    msgs = [{"role": "user", "content": q}]
    try:
        prompt = TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
    tok = getattr(TOKENIZER, "tokenizer", TOKENIZER)
    try:
        inputs = tok(prompt, return_tensors="pt")
    except Exception:
        inputs = TOKENIZER(text=prompt, return_tensors="pt")
    device = next(MODEL.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    max_new = max(16, min(int(body.max_new_tokens or 128), 256))
    temp = float(body.temperature if body.temperature is not None else 0.35)
    try:
        with torch.inference_mode():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=max(temp, 1e-5),
                do_sample=temp > 0.05,
                pad_token_id=getattr(tok, "eos_token_id", None) or TOKENIZER.eos_token_id,
            )
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        text = decoded
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:800]}
    return {
        "answer": text.strip(),
        "adapter": ADAPTER_LABEL,
        "adapter_path": ADAPTER,
        "latency_ms": int((time.time() - t0) * 1000),
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')} (sidecar, train untouched)",
    }


def _load():
    global READY, LOAD_ERROR, MODEL, TOKENIZER, LOADED_AT
    if not ADAPTER:
        LOAD_ERROR = "No QLoRA adapter on disk (v2 saves at step 200; v1 missing)."
        return
    from unsloth import FastLanguageModel
    from peft import PeftModel

    try:
        MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
            model_name=BASE,
            max_seq_length=2048,
            load_in_4bit=True,
            device_map={"": 0},
        )
        MODEL = PeftModel.from_pretrained(MODEL, ADAPTER)
        FastLanguageModel.for_inference(MODEL)
        READY = True
        LOADED_AT = time.time()
        print("MUSE_INFER_READY", ADAPTER_LABEL, flush=True)
    except Exception as exc:
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"[:500]
        print("MUSE_INFER_FAIL", LOAD_ERROR, flush=True)


if __name__ == "__main__":
    import uvicorn

    _load()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MUSE_INFER_PORT", "8766")), log_level="info")
