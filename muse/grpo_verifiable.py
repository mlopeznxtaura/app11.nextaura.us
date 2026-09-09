#!/usr/bin/env python3
"""Phase 2: GRPO with verifiable rewards on the Phase-1 SFT policy.

Recipe follows https://huggingface.co/blog/grpo-with-trl-ifstruct, extended with
execution-grounded coding rewards. Rewards are programmatic (no reward model):
  1. format_reward   — IFStruct-style: output parses as required structure (JSON schema / unified diff / tool-call)
  2. execution_reward — generated code/patch runs and passes sandboxed tests
  3. hygiene_reward  — penalties: repetition, truncation, refusal markers
GRPO group-relative advantages (8 samples/prompt), small KL to SFT policy.
"""
from __future__ import annotations
import json, os, re, subprocess, tempfile, time, pathlib

SFT = os.environ.get("MUSE_SFT", "/data/muse/sft-v1/final")
OUT = os.environ.get("MUSE_GRPO_OUT", "/data/muse/grpo-v1")
LOG = pathlib.Path("/data/muse/run_log.jsonl")

def log_event(ev: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ev["t"] = time.time()
    with LOG.open("a") as f:
        f.write(json.dumps(ev) + "\n")

# ---------- verifiable rewards ----------

def format_reward(completions: list[str], task: list[dict] | None = None, **kw) -> list[float]:
    """1.0 if completion parses as the required structure, else partial credit."""
    out = []
    for comp in completions:
        text = comp if isinstance(comp, str) else comp[0].get("content", "")
        score = 0.0
        # unified diff / patch
        if re.search(r"^diff --git|^--- a/|^\+\+\+ b/", text, re.M):
            score = max(score, 1.0 if re.search(r"^@@ ", text, re.M) else 0.5)
        # JSON block
        m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
        if m:
            try:
                json.loads(m.group(1)); score = max(score, 1.0)
            except Exception:
                score = max(score, 0.3)
        # tool call
        if re.search(r"<tool_call>.*?</tool_call>", text, re.S):
            score = max(score, 0.8)
        out.append(score)
    return out

def execution_reward(completions: list[str], tests: list[str] | None = None, **kw) -> list[float]:
    """Run extracted python against provided test snippet in a sandbox subprocess."""
    out = []
    tests = tests or [""] * len(completions)
    for comp, test in zip(completions, tests):
        text = comp if isinstance(comp, str) else comp[0].get("content", "")
        m = re.search(r"```python\s*(.*?)```", text, re.S)
        if not m or not test:
            out.append(0.0); continue
        code = m.group(1) + "\n\n" + test
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code); path = f.name
            r = subprocess.run(["python3", path], capture_output=True, timeout=20)
            out.append(1.0 if r.returncode == 0 else 0.0)
        except Exception:
            out.append(0.0)
        finally:
            try: os.unlink(path)
            except Exception: pass
    return out

def hygiene_reward(completions: list[str], **kw) -> list[float]:
    out = []
    for comp in completions:
        text = comp if isinstance(comp, str) else comp[0].get("content", "")
        s = 0.0
        words = text.split()
        if words and len(set(words)) / len(words) < 0.2: s -= 0.5   # repetition collapse
        if len(text) > 50: s += 0.1
        if re.search(r"I (cannot|can't|won't)", text): s -= 0.3     # refusal
        out.append(s)
    return out

def main():
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset

    prompts = load_dataset("nvidia/OpenCodeReasoning", split="train", streaming=True).take(2000)

    cfg = GRPOConfig(
        output_dir=OUT,
        learning_rate=1e-6,
        beta=0.01,                       # KL to SFT policy — anti reward-hacking
        num_generations=8,
        max_prompt_length=4096,
        max_completion_length=1024,
        temperature=1.0,
        top_p=0.95,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_steps=200,
        save_total_limit=5,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=SFT,
        args=cfg,
        reward_funcs=[format_reward, execution_reward, hygiene_reward],
        train_dataset=prompts,
    )
    trainer.train()
    trainer.save_model(OUT + "/final")
    log_event({"event": "grpo_complete", "out": OUT + "/final"})
    print("GRPO_DONE", OUT + "/final")

if __name__ == "__main__":
    main()
