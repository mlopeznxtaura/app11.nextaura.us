# Muse-Glimmer-30B → best coding agent at 30B

**Host:** app9.nextaura.us (2×L40S 96GB, 500GB data volume at /data)
**Base:** `meta-models/Muse-Glimmer-30B` (bf16 safetensors, Apache-2.0, 52L×6656d, GQA 32/2, 131k ctx, vision encoder)
**Toolchain:** Unsloth (day-zero muse-glimmer kernels) + TRL + vLLM, transformers ≥ 5.1.0

## Why this can win at 30B
Muse-Glimmer is already distilled from Muse Spark for *agentic* work (tool use, multi-step
reasoning, failure recovery). It is under-trained on *verifiable coding* specifically.
The gap at 30B is not knowledge — it's (a) repo-level SWE behavior, (b) structured-output
reliability, (c) execution-grounded correctness. All three are trainable post-hoc.

## Phase 0 — Environment (app9)
- `requirements-muse.txt`: unsloth, trl>=0.19, vllm, transformers>=5.1.0, datasets, bitsandbytes, peft, accelerate, flash-attn
- Download bf16 base (~60GB) to `/data/models/Muse-Glimmer-30B`
- Smoke: 10-step QLoRA on 100 rows, verify loss finite + VRAM < 40GB

## Phase 1 — SFT (QLoRA, GPU0+GPU1 via DDP or sequential)
Datasets (streaming, interleaved, weights):
1. `nvidia/Nemotron-SFT-SWE-v3.5` (0.35) — proven in this lab (app7 lineage, loss 1.16 @ 59k tok/s on 237M)
2. `SWE-Gym` / `SWE-smith` style repo-patch traces (0.25)
3. `nvidia/OpenCodeReasoning` (0.25) — CoT coding
4. Agentic tool-use traces (app7's distill-coding.jsonl + sol-traces format) (0.15)

Config: r=32 alpha=32 (harder agentic tasks), target q,k,v,o,gate,up,down; lr 2e-4 cosine;
seq 8192 packing; bf16 LoRA on 4-bit base (QLoRA); grad ckpt ON; 1–2 epochs; save every 500 steps.
**Gate G1:** RM panel (OpenAssistant Deberta, 20 prompts) must not regress vs base; HumanEval/MBPP
execution eval must improve ≥ +5pp before Phase 2.

## Phase 2 — GRPO with verifiable rewards (the HF ifstruct recipe, extended)
Per https://huggingface.co/blog/grpo-with-trl-ifstruct + this lab's RM gating:
- **Format reward (IFStruct-style):** JSON/diff/schema compliance checked by parser — verifiable, no model needed
- **Execution reward:** generated patches applied to sandboxed repos; reward = tests passing (SWE-gym style)
- **Penalty terms:** repetition, truncation, invalid tool-call syntax
- vLLM colocate rollouts (temp 1.0, top_p 0.95, 8–16 samples/prompt), GRPO group-relative advantages
- KL β small (0.01) against Phase-1 SFT policy — prevents the reward-hacking collapse the app7 lessons flagged
**Gate G2:** IFStruct compliance ≥ 95%, execution reward climbing, RM panel no regression.

## Phase 3 — Eval + export
- Benchmarks: HumanEval+, MBPP+, SWE-bench-verified-lite (50-task subset), IFStruct
- Export merged bf16 → GGUF Q4_K_M for Ollama (`ollama create muse-glimmer-coder`)
- Register in platform index; app9 GUI leaderboard shows all gates

## Integration with app9 platform
- Every phase logs to run_history (seed lessons format) + auto RM panel on save
- Checkpoints → IBM COS `app9/` prefix, auto-prune keeps latest 5 (MAX_CHECKPOINTS_KEEP)
- Speed telemetry: tok/s, s/step, MB/hr in status line (inherited from app8)
