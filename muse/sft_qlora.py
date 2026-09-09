#!/usr/bin/env python3
"""Phase 1 QLoRA SFT — Muse-Glimmer-30B on app9 (2×L40S).

Follows the GUI finetune playbook (nemotron_swe_finetune + scaling table):
  lr 1e-5, 1–3 epochs, abort if usable rows < 200, stop if loss collapses.

v1 failed by packing 8,733 almost-empty chat renders into 9 sequences and
running 3,000 epochs (loss 4.3 → 2e-8). This run formats tools+messages,
drops short rows, packing=False, 2 epochs max, early-stop on collapse.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

BASE = os.environ.get("MUSE_BASE", "/data/models/Muse-Glimmer-30B")
OUT = os.environ.get("MUSE_OUT", "/data/muse/sft-v2")
LOG = pathlib.Path(os.environ.get("MUSE_RUN_LOG", "/data/muse/run_log.jsonl"))
MIN_CHARS = 256
MIN_ROWS = 200
COLLAPSE_LOSS = 0.05
COLLAPSE_AFTER = 40

MIX = [
    ("nvidia/Nemotron-SFT-SWE-v3.5", 0.6, None, "train"),
    ("nvidia/OpenCodeReasoning", 0.4, "split_0", "split_0"),
]


def log_event(ev: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ev.setdefault("t", time.time())
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev) + "\n")


def _as_messages(msgs_raw):
    if not msgs_raw:
        return []
    if isinstance(msgs_raw, dict):
        keys = list(msgs_raw.keys())
        if not keys:
            return []
        n = len(msgs_raw[keys[0]])
        msgs_raw = [{k: msgs_raw[k][i] for k in keys} for i in range(n)]
    out = []
    for m in msgs_raw:
        if isinstance(m, dict):
            m = dict(m)
            tcs = m.get("tool_calls") or []
            fixed = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        fn["arguments"] = {"raw": args}
                tc["function"] = fn
                fixed.append(tc)
            if fixed:
                m["tool_calls"] = fixed
            out.append(m)
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            out.append({"role": str(m[0]), "content": str(m[1])})
    return out


def _render_manual(msgs, tools=None) -> str:
    parts = []
    if tools:
        try:
            parts.append("<|system|>\nAvailable tools:\n" + json.dumps(tools)[:8000])
        except Exception:
            pass
    for m in msgs:
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        content = (m.get("content") or "") if isinstance(m, dict) else ""
        parts.append(f"<|{role}|>\n{content}")
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            try:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                parts.append(f"<tool_call>{json.dumps(fn)}</tool_call>")
            except Exception:
                pass
    return "\n".join(parts)


def format_one(row, tokenizer) -> str:
    msgs = _as_messages(row.get("messages"))
    tools = row.get("tools")
    if msgs:
        try:
            kwargs = {"tokenize": False, "add_generation_prompt": False}
            if tools:
                try:
                    return tokenizer.apply_chat_template(msgs, tools=tools, **kwargs)
                except TypeError:
                    pass
            return tokenizer.apply_chat_template(msgs, **kwargs)
        except Exception:
            return _render_manual(msgs, tools)
    inp = row.get("input") or row.get("problem") or ""
    out = row.get("output") or row.get("solution") or ""
    if inp or out:
        return f"### Problem\n{inp}\n\n### Solution\n{out}"
    return ""


def load_mix(take_n: int):
    from datasets import load_dataset, interleave_datasets

    streams, weights, loaded = [], [], []
    for repo, weight, cfg_name, split in MIX:
        try:
            if cfg_name:
                ds = load_dataset(repo, cfg_name, split=split, streaming=True)
            else:
                ds = load_dataset(repo, split=split, streaming=True)
            streams.append(ds)
            weights.append(weight)
            loaded.append(repo)
            log_event({"event": "dataset_ok", "repo": repo})
        except Exception as exc:
            log_event({"event": "dataset_skip", "repo": repo, "err": str(exc)[:240]})
    if not streams:
        raise SystemExit("no datasets available")
    total_w = sum(weights)
    interleaved = interleave_datasets(
        streams,
        probabilities=[w / total_w for w in weights],
        stopping_strategy="all_exhausted",
    )
    log_event({"event": "materialize_start", "rows": take_n, "repos": loaded})
    rows_buf = []
    for row in interleaved:
        rows_buf.append(row)
        if len(rows_buf) >= take_n:
            break
    log_event({"event": "materialize_done", "rows": len(rows_buf)})
    return rows_buf, loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset, Features, Value
    from transformers import TrainerCallback

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=args.seq,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.r,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    take_n = 200 if args.smoke else args.rows
    rows_buf, loaded = load_mix(take_n)

    texts, dropped, lens = [], 0, []
    for row in rows_buf:
        text = format_one(row, tokenizer)
        if text and len(text) >= MIN_CHARS:
            # seq=8192 cannot use 150k-char traces; also Arrow string offsets
            # overflow at ~2GB (20k × 149k chars). Truncate to context.
            clipped = text[: args.seq * 6]
            texts.append(clipped)
            lens.append(len(text))
        else:
            dropped += 1
    mean_len = int(sum(lens) / len(lens)) if lens else 0
    log_event({
        "event": "format_stats",
        "kept": len(texts),
        "dropped": dropped,
        "mean_chars": mean_len,
        "min_chars": MIN_CHARS,
    })
    print(f"FORMAT kept={len(texts)} dropped={dropped} mean_chars={mean_len}", flush=True)
    if len(texts) < (40 if args.smoke else MIN_ROWS):
        log_event({"event": "abort_too_few_rows", "kept": len(texts)})
        raise SystemExit(f"too few usable rows ({len(texts)}) — abort, same failure mode as sft-v1")

    train_ds = Dataset.from_dict(
        {"text": texts},
        features=Features({"text": Value("large_string")}),
    )
    steps_per_epoch = max(1, math.ceil(len(texts) / 16))
    max_steps = 10 if args.smoke else min(args.steps, steps_per_epoch * max(1, args.epochs))
    warmup = min(50, max(5, max_steps // 10))
    save_every = max(50, max_steps // 4) if not args.smoke else 10
    log_event({
        "event": "schedule",
        "n_examples": len(texts),
        "epochs_cap": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "lr": args.lr,
        "warmup": warmup,
        "packing": False,
        "out": OUT,
    })
    print(
        f"SCHEDULE examples={len(texts)} steps={max_steps} "
        f"(≤{args.epochs} epochs × {steps_per_epoch} steps/epoch) lr={args.lr}",
        flush=True,
    )

    class CollapseStop(TrainerCallback):
        def on_log(self, args_t, state, control, logs=None, **kwargs):
            logs = logs or {}
            loss = logs.get("loss")
            if loss is None or state.global_step < COLLAPSE_AFTER:
                return
            if float(loss) < COLLAPSE_LOSS:
                log_event({
                    "event": "early_stop_collapse",
                    "step": state.global_step,
                    "loss": float(loss),
                    "threshold": COLLAPSE_LOSS,
                })
                print(
                    f"EARLY_STOP collapse loss={loss} at step {state.global_step} "
                    f"(GUI rule: loss<<1 on tiny replay = memorization)",
                    flush=True,
                )
                control.should_training_stop = True

    cfg = SFTConfig(
        output_dir=OUT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        max_steps=max_steps,
        bf16=True,
        logging_steps=5,
        save_steps=save_every,
        save_total_limit=5,
        max_length=args.seq,
        packing=False,
        dataset_text_field="text",
        dataset_num_proc=1,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[CollapseStop()],
    )
    t0 = time.time()
    trainer.train()
    dur = time.time() - t0
    model.save_pretrained(OUT + "/final")
    tokenizer.save_pretrained(OUT + "/final")
    log_event({
        "event": "sft_complete",
        "version": "sft-v2",
        "steps": max_steps,
        "n_examples": len(texts),
        "duration_s": round(dur, 1),
        "repos": loaded,
        "lr": args.lr,
    })
    print("SFT_DONE", OUT + "/final")


if __name__ == "__main__":
    main()
