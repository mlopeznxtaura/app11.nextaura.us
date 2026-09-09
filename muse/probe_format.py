#!/usr/bin/env python3
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
import importlib.util

spec = importlib.util.spec_from_file_location("sft", "/data/muse/sft_qlora.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
tok = AutoTokenizer.from_pretrained("/data/models/Muse-Glimmer-30B", trust_remote_code=True)
ds1 = load_dataset("nvidia/Nemotron-SFT-SWE-v3.5", split="train", streaming=True)
ds2 = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
it = interleave_datasets([ds1, ds2], probabilities=[0.6, 0.4], stopping_strategy="all_exhausted")
kept = drop = 0
lens = []
for i, row in enumerate(it):
    if i >= 40:
        break
    t = mod.format_one(row, tok)
    if t and len(t) >= 256:
        kept += 1
        lens.append(len(t))
        if kept == 1:
            print("SAMPLE", t[:180].replace("\n", " | "))
    else:
        drop += 1
        print("DROP", i, "len", len(t or ""), "keys", list(row.keys())[:8])
print("SMOKE kept", kept, "drop", drop, "mean", int(sum(lens) / len(lens)) if lens else 0, "min", min(lens) if lens else 0)
