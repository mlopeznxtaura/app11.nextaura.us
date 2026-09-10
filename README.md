# app11.nextaura.us — Open Source Training Lab

**Public, CPU-first fork** of the NextAura multimodal fine-tuning GUI. Explore the full training playbook, Chinchilla scaling tables, MCF/V-JEPA recipes, SFT/distill generators, reward-model eval, and Stage 2 inference — **without a GPU or any secrets**.

Live demo: **https://app11.nextaura.us**

Private GPU labs (`app7`–`app9`) keep production throughput; **app11 is the teachable, forkable reference**.

---

## Quick start

### Local (CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export TRAIN_DEVICE=cpu PUBLIC_URL=http://localhost:8080
uvicorn server:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080

### Docker

```bash
docker build -t app11-nextaura-us .
docker run -p 8080:8080 -e TRAIN_DEVICE=cpu app11-nextaura-us
```

### Optional secrets

Copy `.env.example` → `.env`. **Nothing is required** for browsing the GUI or running tiny CPU smokes. Set `HF_TOKEN` only when you want Hugging Face streaming datasets, HF checkpoint import, or live teacher distillation.

---

## Architecture

```
Browser  →  Cloudflare Worker (nextaura-app11-us)  →  IBM Code Engine (CPU)
                app11.nextaura.us                         uvicorn + FastAPI
```

| Layer | Role |
|-------|------|
| `public/index.html` + `public/*.js` | Workflow-gated training GUI |
| `server.py` | FastAPI: `/api/start`, `/api/status`, checkpoints, infer, eval |
| `train_engine.py` + `model.py` | GPT trainer (CPU/CUDA/MPS via `TRAIN_DEVICE`) |
| `hf_datasets.py` / `hf_stream_loader.py` | Hugging Face stream registry |
| `vjepa_pack.py` / `scalar_features.py` | MCF v2 latent packing + 12D scalar projection |
| `distill_teachers.py` / `sft_topics.py` | Offline corpus generators |
| `reward_eval.py` | OpenAssistant Deberta reward panel |

---

## GUI map — every section and why it exists

The UI is a **single-page lab console** split into theory, playbook (config), data tools, training controls, eval, and inference. Sections unlock via a **3-step workflow** (`run-workflow.js`) so you cannot start a run before locking architecture and stating a goal.

### Header bar

| Element | What it does | Why |
|---------|--------------|-----|
| **Ready badge** | Polls `/api/health` — online / error | Confirms server + torch device before you touch controls |
| **Pretrain / Fine-tune badge** | Mirrors training mode toggle | Chinchilla byte budgets differ 10× between pretrain and SFT |
| **Infer model select** | Switch Stage 2 loaded checkpoint | Lets you compare checkpoints without retraining |
| **Sync → UpCloud** | Upload local `.pt` to optional S3 archive | Only active when `UPCLOUD_S3_*` env is set |
| **Stage 2 ↓** | Scroll to inference panel | Stage 1 (train) and Stage 2 (chat) are intentionally separated |
| **Host badge** | Live session: model, params, device | Prevents “mystery checkpoint” confusion during long polls |

### Theory panel (collapsible)

Explains the **2017 Transformer** (attention, multi-head, FFN, positional encoding) and the **NextAura recipe** (causal mask, geometry/binary/world-model ideas, MoE notes).

**Why:** New contributors see *what* the trainer implements before touching hyperparameters. Preset buttons apply documented baselines instead of random sliders.

### Run Playbook — recipe presets

One-click configs in `run-playbook.js`:

| Preset | Purpose |
|--------|---------|
| Transformer baseline SFT | Small GPT SFT sanity check |
| HF Qwen SFT | Import path for HuggingFace causal LMs |
| NextAura novel recipe | Internal architecture experiments |
| Muse QLoRA | Documents 30B QLoRA path (GPU sidecar — not auto-started on app11) |
| Nemotron SWE / 51M SFT | Coding SFT smokes |
| Multimodal / V-JEPA smokes | Caption+text and video-latent streams |
| Distill / text mixed | Teacher-generated JSONL paths |
| **MCF v2 HLS** | Metamorphic Constraint Field v2 — latent packing, GCT, dual-target loss |

**Why:** Training labs die from “wrong combo” (e.g. FineWeb byte stop on V-JEPA rows). Presets encode lessons from production failures.

### MCF v2 block

Documents **Metamorphic Constraint Field** v2:

- Latent packing (128 frames/block)
- Global Context Token for physics constraints
- Dual-target loss (latent + scalar MSE)
- ~360M profile, **1.5B token stop** (~5.25 GB) — avoids the old 24 GB trap on tiny V-JEPA rows

**Why:** MCF v1 on V-JEPA with FineWeb-style batches took ~23 min/step. v2 HLS is the corrected recipe.

### Step 1 — configuration panels

#### Data stream (`#sectionData`)

- **Data source** — fineweb, jsonl, multimodal, vjepa, mixed, distill-*, fable/sol/kimi traces, nemotron, etc.
- **HF dataset** — picker when source needs a dataset id
- **Target GB** — byte stop for streaming pretrain

**Why:** One trainer serves text pretrain, multimodal captions, and V-JEPA latent metadata. Source drives tokenizer + scalar projection requirements.

#### Architecture (`#sectionArch`)

Layers, embedding dim, heads, sequence length, FFN, vocab → live param estimate via `/api/model/estimate`.

**Why:** Chinchilla optimal tokens scale with params. The GUI shows footprint *before* you commit GPU hours.

#### Optimizer (`#sectionOpt`)

LR, batch, dropout, weight decay, micro-batch, grad accumulation, warmup.

**Why:** Footprint ladder (10M / 25M / 50M) sets sane defaults; advanced users override here.

#### Data mix (`#sectionCorpus`)

- Local JSONL mix ratio
- Curated corpora list
- HF stream chips: Fable, Sol, Kimi, Nemotron, preference, FineWeb

**Why:** Production runs interleave curated local data with streaming HF shards. Chips toggle streams without editing JSON.

#### Advanced (`#sectionAdv`)

LR schedule, total steps, mixed precision, save interval, flash attention, **scalar 12D projection**, gradient checkpointing, tokenizer id.

**Why:** V-JEPA rows need `scalar_input_projection=true`. Flash attention auto-disables on CPU.

#### Footprint quick buttons

10M / 25M / 50M / Multimodal / V-JEPA / Text mixed + **Lock step 1**.

**Why:** Locks config into workflow step 2 (goal). Prevents accidental edits mid-run.

### Run Goal (`#sectionRunGoal`) — workflow step 2

- Goal title, agent name, intention textarea
- Stop mode: **bytes** / **steps** / **Chinchilla optimal**
- Live byte + step progress bars

**Why:** Forces you to state *why* you're running before Start. Chinchilla mode computes ~20 tokens/param automatically.

### Chinchilla cheat sheet

Three tables (`#chinchillaScalePanel`):

1. Scale multipliers (pretrain vs finetune)
2. Footprint ladder (10M → 360M MCF)
3. Tokenizer guide

Plus **MM/V-JEPA recipe grid** (`recipe-cheatsheet.js`) — one-click multimodal and V-JEPA dataset picks.

**Why:** Hoffmann/Chinchilla law is easy to get wrong in bytes. Tables match `/api/scaling/chinchilla`.

### Stage 1 — data tools

| Row | Function |
|-----|----------|
| **FineWeb** | HF FineWeb config picker (hidden unless fineweb source) |
| **SFT generator** | Build coding/physics/math JSONL via `/api/sft/generate` |
| **Distill generator** | Teacher distillation via `/api/distill/generate` |
| **JSONL upload** | Upload local training files to `data/` |

**Why:** Stage 1 is *data prep* before the train loop. Generators keep corpora reproducible without manual curl.

### Run history

Table of past runs + lessons from `run_history.py`.

**Why:** Engineering memory — what config produced which checkpoint.

### Live recipe panel

Shows **active** vs **pending** config from last lock + edits.

**Why:** Diff view before you hit Start.

### Train controls

Start / Resume / Pause / Stop / Download live weights, stats line, progress bar.

**Why:** Mirrors `train_engine` state machine. Preflight (`run-guard.js`) calls `/api/preflight` before start.

### Checkpoint panel

- Upload `.pt`
- List saved checkpoints
- HF import (`#hfRepoId`)
- Reset session

**Why:** Stage 2 infer and RM eval need a loaded checkpoint. HF import path uses `transformers` (optional `HF_TOKEN` for gated models).

### Terminal mirror + GPU VRAM bar

Scrollable log (`#mirror`) and VRAM display (`#gpuVramLive`).

**Why:** On app11 CPU, VRAM shows “N/A” — on GPU forks it prevents OOM surprises. Log is the same output you'd see in tmux on a GPU box.

### Eval panel — reward model leaderboard

- `/api/eval/leaderboard` — RM scores across checkpoints
- Compare selected checkpoints
- TimeMoE baseline button

**Why:** Preference quality without human A/B. Uses OpenAssistant Deberta RM (downloads on first use).

### Stage 2 — inference (`infer.js`)

- **Context grid** — shows training config that produced loaded weights
- **Prompt chips** — quick test prompts
- Sampling: max tokens, temperature, repetition penalty
- **Ask / Score / Clear** — chat + RM score single reply
- `#inferChat` transcript

**Why:** Validates that a checkpoint *behaves* before you ship it. Muse sidecar path documented but not required on CPU demo.

### Changelog footer

Loads `public/changelog.json` for release notes.

---

## API reference (common routes)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Boot config, datasets, capabilities |
| GET | `/api/status` | Live training snapshot |
| POST | `/api/start` | Start training |
| POST | `/api/preflight` | Validate config without running |
| POST | `/api/stop` | Stop training |
| GET | `/api/checkpoints` | List checkpoints |
| POST | `/api/infer/ask` | Stage 2 generation |
| GET | `/agent.json` | Machine-readable agent contract |

Full contract: `GET /agent.json` or `public/agent.txt`.

---

## Training concepts (short glossary)

| Term | Meaning |
|------|---------|
| **FineWeb** | HuggingFaceFW/fineweb text pretrain stream |
| **Chinchilla** | ~20 tokens per parameter optimal training budget |
| **V-JEPA** | Video latent features → short token sequences; needs 12D scalar projection |
| **MCF** | Metamorphic Constraint Field — dual-stream physics-aware training |
| **Distill** | Generate JSONL from teacher LMs (coding/math/physics) |
| **SFT** | Supervised fine-tuning on instruction JSONL |
| **RM eval** | OpenAssistant reward model scores for checkpoint comparison |

See `docs/MCF-VJEPA-HANDOFF-NEXT-AGENT.md` for MCF v1 failure postmortem and v2 fixes.

---

## Deploy to IBM Code Engine

```powershell
.\scripts\deploy-app11-ce.ps1
```

Updates Cloudflare worker `nextaura-app11-us` → CE URL. Requires `ibmcloud` CLI + wrangler login.

---

## Differences from app9 (private GPU lab)

| | app9 | app11 |
|---|------|-------|
| GPU | 2× L40S required for real runs | CPU default; set `TRAIN_DEVICE=cuda` locally |
| Secrets | IBM COS, HF token on server | None required for demo |
| Muse QLoRA | Sidecar on :8766 | Documented only |
| Browser agents | Internal automation | Removed from OSS tree |
| Checkpoints | COS sync | Local `checkpoints/` only unless you opt in |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Fork, experiment, PR. Keep secrets out of the tree. For production GPU throughput, use the private `app9.nextaura.us` lab; use **app11** to learn the GUI and reproduce smokes on CPU.
