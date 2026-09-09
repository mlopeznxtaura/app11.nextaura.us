# App8 MCF V-JEPA Handoff — For Next Agent

**Date:** 2026-09-03  
**From:** Session agent (app8 MCF implementation + HF probe)  
**To:** Next agent evaluating how to proceed with a **full run on app8**  
**Audience:** You inherit a failed-but-not-dead recipe from a **prior agent** who designed the MCF Stream A preset. Read this before changing anything.

---

## 1. Executive summary

The **prior agent’s MCF Stream A recipe** assumed V-JEPA HF streams behave like FineWeb text pretrain. They do not. The recipe was deployed on **app8** (1× L40S, `169.59.163.90`) with **360M / 24 GB / batch 768 / 4-dataset V-JEPA interleave**. After **~40+ minutes**, the run reached **step 1** and **0.79 MB** toward a **24 GB** target. It is not hung; it is **structurally unable to finish** at this configuration without code or recipe changes.

**app7** (2× L40S, `150.239.225.134`) runs the **same 360M arch** on **FineWeb** and is healthy: **~1.26 GB, step ~1501, ~47k tok/s**. Do not conflate app7 success with app8 recipe validity.

**Your job:** Decide how to deliver a **real full pretrain on app8** (user requirement: not a smoke, not batch-only hacks without a plan). The failure is **recipe + trainer data-path mismatch**, not GPU hardware.

---

## 2. What the prior agent prescribed (MCF recipe)

The upstream agent defined **Metamorphic Constraint Field (MCF) Stream A**:

| Prescription | Value |
|---|---|
| Slot | **app8 only** — experimental MCF; app7 = FineWeb Phase A |
| Data | **V-JEPA HF streams** (continuous physics latents → scalar 12d → causal LM) |
| Arch | **360M**: 17L × 960d × 15h, block 256, ffn 3840, gpt2 tokenizer |
| Batch | **768** (micro 128 × grad_accum 6) — copied from app7 turbo |
| Target | Escalated **0.5 → 5 → 24 GB** (user wanted Chinchilla-scale ~24 GB for 360M) |
| Features | `scalar_input_projection=true`, `gradient_checkpointing=true`, flash attn bf16 |
| Multi-dataset | User demanded **more than sokoban (~10k rows)**; agent added 4-way interleave |

Preset IDs: `mcf-recipe-v1-stream-a`, GUI preset `mcf_recipe_vjepa`, script `scripts/start_mcf_smoke.py`.

**Implicit assumptions in the prior recipe (all wrong or incomplete):**

1. V-JEPA rows are **training-sized text** like FineWeb documents.
2. **batch 768** from FineWeb Phase A transfers to V-JEPA without throughput math.
3. **24 GB byte target** is meaningful for tiny metadata rows.
4. **Multi-dataset interleave** increases diversity without latency cost.
5. The existing trainer **`feed_text()` → token buffer → step when `block×batch` tokens`** works for latent-metadata corpora.

---

## 3. What we implemented (this session)

### Code (mostly in `app7-nextaura-us/`, deployed to app8)

| Change | Path | Purpose |
|---|---|---|
| HF probe scripts | `scripts/probe_vjepa_priority.py`, `probe_vjepa_datasets.py` | Find loadable V-JEPA datasets on Hub |
| Registry expansion | `hf_datasets.py` | +4 datasets from HF search |
| World-state text extractor | `multimodal_data.py` → `vjepa_training_text()` | Unlock CS2/ego10k rows (scalar metadata → trainable string) |
| Multi-stream loader | `hf_stream_loader.py` → `open_hf_train_stream_interleaved()` | Round-robin across N HF streams |
| API field | `server.py` → `StartIn.vjepa_dataset_ids` | Pass list of datasets |
| Shuffle guard | `server.py` vjepa branch | Disable HF shuffle when multi-stream (was materializing 6M+ CS2 rows → hang) |
| GUI / start script | `app8-nextaura-us/public/run-playbook.js`, `scripts/start_mcf_smoke.py` | 24 GB, 4-dataset mix |

### Verified OK V-JEPA datasets (probed on app8 with HF token)

1. `qinglinhou/sokoban-10k-vjepa2-tokenized` — JSON metadata, ~10k rows  
2. `cbctr/cs2-10k-vjepa2-latents-300` — CS2 world-state parquet (~6.9M frames; streams)  
3. `rookierufus/ego10k-vjepa-latents` — manufacturing ego latents + tags  
4. `rookierufus/Vjepa_mamba_dataset` — video_index / frame_index rows  

**23+ other Hub datasets failed** (broken loaders, `.pt`-only, webdataset shards, epic-kitchens dataclass error, etc.).

---

## 4. Why the recipe failed — root cause analysis

### 4.1 Primary failure: token economics (batch × row size)

Trainer logic (`train_engine.py`):

```python
need = block_size * batch_size + 1   # 256 × 768 + 1 = 196,609 tokens per optimizer step
while len(token_buffer) >= need:
    train_step()
```

Each V-JEPA row encodes to roughly **50–200 tokens** (short metadata + `world_state:` block).  
Observed on app8: **~2,800 stream rows → 1 completed step** → **~2,800 rows/step**.

At measured ingestion **~2 rows/sec** (4 interleaved HF streams, network + decode):

- **~23 minutes** to accumulate tokens for **one** step (matches step 1 timing: start ~16:20 UTC, step 1 log ~16:43 UTC).
- **~0.5–0.8 MB bytes_written per step** (tiny UTF-8 strings).
- **24 GB ÷ 0.7 MB/step ≈ 34,000 steps** × **23 min/step ≈ 13 months** wall time.

FineWeb on app7 works because **one row = thousands of tokens**; often **fractions of a row per step**. Same batch size, different data morphology.

**This is not a bug in the dataloader loop.** It is a **category error**: applying FineWeb-scale batch/target to metadata crumbs.

### 4.2 Secondary failure: prior agent smoke metrics misled escalation

| Run | Target | Result | Misread |
|---|---|---|---|
| 64M smoke | 0.5 GB | Loss → ~0.15 fast | “136k tok/s = good” — was **memorizing ~10k sokoban templates** |
| 360M restart | 5 GB | ~91 steps, repetitive | User correctly called it waste |
| 360M “full” | 24 GB | step 1 in 40 min | Looks “stalled”; actually **correct behavior for recipe** |

The prior agent treated **tok/s on step boundaries** and **loss collapse on 10k rows** as signs the recipe scales. It does not.

### 4.3 Tertiary issues (fixed or mitigated, but worth knowing)

| Issue | Symptom | Mitigation applied |
|---|---|---|
| `shuffle_hf_datasets=true` on multi-V-JEPA | Hang trying to materialize CS2 corpus | Force shuffle off when `len(vjepa_dataset_ids) > 1` |
| Only sokoban worked initially | User anger | Probed Hub; added world_state text for 3 more datasets |
| `AUTO_RESUME=1` on app8 | Wrong checkpoint/arch restored after restart | Manual `reset-session` + `resume:false` |
| Single `hf_dataset_id` only | Could not mix datasets | Added `vjepa_dataset_ids` + interleave |
| epic-kitchens default | Dataclass load error | Replaced with sokoban default |

### 4.4 Architectural gaps the prior recipe never addressed

1. **No row packing / concatenation** — N small V-JEPA rows should be merged into one training document before `feed_text()`.
2. **No V-JEPA-specific batch preset** — batch 768 is hostile to short rows.
3. **No byte-target calibration** — 24 GB assumes FineWeb-like bytes/step.
4. **Scalar path underused** — latents live in `latent_bytes` / `.pt` columns we **skip**; only text + inferred scalars from `m_flow` etc.
5. **Stage 2 inference** still generic GPT — no MCF eval loop wired.
6. **Finite datasets** — sokoban 10k without `epoch_cycles` + interleave with infinite CS2 stream = **distribution skew** over time.

---

## 5. Current live state (as of handoff)

### app8 — `ubuntu@169.59.163.90`, `/opt/app8-nextaura-us`

| Field | Value |
|---|---|
| Status | `streaming`, `running=true` |
| Goal | MCF Stream A — V-JEPA multi-dataset, **24 GB** |
| Arch | 17L × 960d (~237M reported / 360M target footprint) |
| Batch | 768 (micro 128 × 6) |
| Step | **1** (checkpoint `nextaura-237m-step1.pt`) |
| Bytes | **0.79 MB / 24 GB** |
| vjepa_hits / stream_index | **~2814** |
| Loss @ step 1 | **11.07** |
| tok/s @ step 1 | **~46k** (GPU is fine **when a step fires**) |
| GPU | ~27 GB VRAM allocated, often **0% util** between steps (buffering HF rows) |
| Datasets | 4-way interleave (see §3) |

SSH: `~/.ssh/golias_ibm`  
Service: `app8-nextaura.service` (AUTO_RESUME=1 — **watch for stale resumes**)

### app7 — reference healthy run

| Field | Value |
|---|---|
| Data | FineWeb `sample-10BT` |
| Target | 14.15 GB |
| Step | **~1501** |
| Bytes | **~1.26 GB** |
| tok/s | **~47k** |

**Do not stop app7 for app8 experiments.**

---

## 6. What “full run on app8” actually requires

User intent: **real pretrain on app8**, not another smoke. Minimum bar:

- Reach **Chinchilla-relevant token volume** for ~360M (~4–24 GB depending on who you ask; user said **24 GB**).
- **Diverse V-JEPA** (not sokoban-only memorization).
- Finish in **days, not months**.

That **cannot** happen with current `feed_text` + batch 768 + raw row stream. Required work (pick one strategy or combine):

### Option A — Row packing (recommended, trainer change)

Before `feed_text()`, accumulate V-JEPA text until **≥ X tokens** (e.g. 8k–32k) or **≥ Y bytes**, merging scalars sensibly.  
Restores FineWeb-like “documents” from metadata rows. Keeps batch 768 if desired.

**Touch:** `server.py` `_stream_finite_rows` or new `_vjepa_pack_rows()` wrapper.

### Option B — V-JEPA batch profile (quick mitigation)

Drop `batch_size` to **128–256** for `data_source=vjepa`.  
Steps complete **~6× faster** on row count; still bytes-starved but moves visibly.  
**Not sufficient alone for 24 GB** unless combined with packing or epoch_cycles.

### Option C — Epoch cycles on finite mix

`epoch_cycles=true`, `epochs=N`, materialize+shuffle **only finite** sets (sokoban + mamba + capped CS2 slice).  
Prevents infinite CS2 from drowning sokoban; enables replay.  
Still needs packing or smaller batch for sane step time.

### Option D — Different data modality (recipe pivot)

Prior agent’s LM-shell approach may be wrong for true V-JEPA. Alternatives:

- Train on **distill JSONL** already in repo (`5090dev/data/distill-*.jsonl`) for physics/math streams on app8.
- Use **Stream B** symbolic mix (sol + nemotron) with scalars — code path exists (`mcf_recipe` preset).
- Native latent training (not implemented — would need new head/loss, not `feed_text`).

### Option E — Target recalibration

If keeping tiny rows without packing: express stop in **steps** or **tokens**, not GB.  
24 GB byte target is **misleading** when rows are 300-byte metadata strings.

---

## 7. Recommended decision tree for next agent

```
START
  │
  ├─ User insists on V-JEPA HF + 360M + 24 GB?
  │     └─ YES → Implement row packing (A) BEFORE restart.
  │              Then: batch 256–768, 4-dataset mix, epoch_cycles on finite ids.
  │              Validate: step 2 within ~5 min, ≥10 MB after 100 steps.
  │
  ├─ User accepts recipe fix but same science goal?
  │     └─ Implement A + B; restart app8; do NOT reuse step-1 ckpt (wrong loss scale / warmup).
  │
  ├─ User needs full run THIS WEEK?
  │     └─ Pivot app8 to Stream B mixed HF (sol/nemotron/math) OR local distill JSONL
  │        with same 360M footprint — proven path on app7-style runs.
  │
  └─ User wants true latent V-JEPA?
        └─ Stop LM-shell pretense; spec new objective — out of scope for current server.py.
```

**Do not** simply lower target to 5 GB and call it full without fixing row packing — you’ll get another fast-memorizing run on finite sokoban.

---

## 8. Restart checklist (when you have a fix)

1. `POST /api/stop` — wait until idle  
2. `POST /api/checkpoints/reset-session` — **required** (AUTO_RESUME poisons config)  
3. Confirm `.gpu.env` has `HF_TOKEN`  
4. Preflight: `POST /api/preflight` with body including `vjepa_dataset_ids`  
5. Start with `resume: false`, explicit `apply` flags  
6. Watch first **10 steps**: step cadence, MB growth, GPU util spikes, loss trend  
7. Success criteria: **≥1 step / 5 min**, **≥50 MB / hour**, loss stable-decreasing (not instant → 0.15 on sokoban)

Script: `/opt/app8-nextaura-us/scripts/start_mcf_smoke.py`  
Env: `MCF_TARGET_GB=24`, `MCF_VJEPA_MIX=ds1,ds2,...`

---

## 9. Key file paths

```
app7-nextaura-us/server.py          — training loop, StartIn, vjepa branch
app7-nextaura-us/train_engine.py    — feed_text token buffer (THE bottleneck)
app7-nextaura-us/multimodal_data.py — vjepa_training_text, vjepa_scalar_features
app7-nextaura-us/hf_stream_loader.py — interleaved streams
app8-nextaura-us/public/run-playbook.js — MCF presets, VJEPA_MIX_DEFAULT
app8-nextaura-us/scripts/start_mcf_smoke.py — CLI launcher
Remote: /opt/app8-nextaura-us/     — deployed copy (may lag local app8-nextaura-us)
```

Deploy: tarball or scp from app7; `scripts/deploy-app-gpu-ibm.ps1` defaults to stale app8 folder — verify before deploy.

---

## 10. Message to the agent who wrote the original recipe

You optimized for **theory alignment** (MCF Stream A, scalar 12d, V-JEPA physics braid) but copied **FineWeb throughput knobs** (batch 768, GB target, tok/s smoke metrics) onto **metadata-scale HF rows**. The trainer was built for **long text documents**; V-JEPA Hub corpora are mostly **latent dumps with short sidecar fields**. Without row packing or batch rescaling, **24 GB is not a training target — it is a calendar sentence**. Multi-dataset interleave was the right instinct for diversity; without packing it only adds **HF latency** (~2 rows/sec). The 64M smoke “success” was **memorization**, not validation. Next agent: fix the **token economics** first, then rerun full scale.

---

## 11. Open questions for product owner

1. Is **LM-next-token on V-JEPA metadata strings** the actual research bet, or placeholder until native latent head exists?  
2. Is **24 GB** hard requirement or “~1× Chinchilla for 360M”? (15.4 GB optimal shown in API.)  
3. Can app8 **full run** use **distill JSONL** (user has `5090dev/data/distill-merged.jsonl` locally) while V-JEPA packing is built?  
4. Should app8 remain **MCF-only** or also run Stream B braid?

---

*End handoff. Do not declare app8 “stalled GPU” — declare **recipe–trainer mismatch** and fix packing before the next full run.*

---

## 12. UI / run history / workflow gates (2026-09-03 addendum)

User report: **run history not learning**, **red locks permanent**, **infer/recipe panels stale**. Root causes below.

### 12.1 Split deploy — app8 UI is not app7 UI

| Asset | app7 (`app7-nextaura-us`) | app8 remote (`/opt/app8-nextaura-us`) |
|---|---|---|
| `run-workflow.js` | ✅ loaded in index | ⚠️ **file exists but NOT in index.html** |
| Workflow gate banners | ✅ `workflow-gated` sections | ❌ **0 gate banners** in deployed index |
| `infer.js` version | v10 + `updateInferFromStatus` | **v3** — no live context grid |
| `run-goal.js` | ✅ | ❌ missing |
| `poll()` hooks | `updateInferFromStatus`, `updateRunPlaybook`, `renderRunGoalLive` | **only** `renderStatus` |

If the user sees red **“Locked — complete and lock Step 1…”** banners, they are either on **app7 URL** or a **cached app7 index**. Remote app8 index only loads `infer.js?v=3` + `run-playbook.js` — **no progressive unlock script runs**.

**Fix:** Deploy **full app7 `public/`** to app8 (or symlink), ensuring:
```html
<script src="/static/run-workflow.js?v=1"></script>
<script src="/static/run-goal.js?v=1"></script>
<script src="/static/infer.js?v=10"></script>
```

### 12.2 Why locks feel “permanent”

`run-workflow.js` design (app7):

- Gates unlock when user **manually clicks “Lock step 1 / 2”** after validation passes.
- State persists in **`localStorage` key `app7RunWorkflowLocks`** — survives refresh, never auto-clears.
- `highestUnlockedStep()` returns **0** until step 1 locked → **all gated sections stay red forever** if user never clicked Lock.
- **Training progress does NOT unlock gates** — no hook from `poll()` → `renderRunWorkflow()`.

**User expectation:** gates turn green as config → goal → history unlock in sequence **automatically** when checklist passes or run starts.

**Fix options (pick one):**
1. **Auto-lock** when `configChecks().ok` → call `setLock('config', true)` on Start or when playbook preset applied.
2. **Auto-unlock display** — `gateSection` uses `checkStep().ok` instead of manual locks for read-only sections (history, recipe).
3. **Reset workflow** button in UI + clear `localStorage` on preset apply (`resetRunWorkflow()` exists but is not wired to MCF preset).
4. On **`s.running === true`**, force-unlock sections 2–3 for monitoring (train controls, history visible during run).

### 12.3 Run history — writes happen, UI doesn’t refresh

**Server (`run_history.py` + `server.py`):**
- `_auto_record_run()` fires on **`complete`**, **`stopped`**, **`interrupted`** only — **NOT on checkpoint save**.
- File: `data/run-history.jsonl` on app8 **does contain** recent MCF rows (64m-step493 interrupted, etc.).
- **`phase: "unknown"`** for vjepa runs — `_infer_phase()` doesn’t classify `data_source=vjepa`; lessons are weak.
- **No `.pt` on disk** for current 237m run (checkpoints may be memory-only until step 1000 save) — backfill can’t add row.

**UI (`index.html` poll):**
```javascript
if (s.status === 'complete' && !_lastHistStatus) refreshRunHistory();
```
- History refreshes **only on `complete`** — not on stop/interrupt, not on checkpoint, not during long runs.
- `boot()` calls `refreshRunHistory()` once — user sees **seed 50m rows** until hard refresh after a stop.
- Table shows old seeds + `(gone)` checkpoints — **new 360M MCF runs are in jsonl but UI stale**.

**Fix:**
```javascript
// In poll(), after renderStatus:
if (s.status !== window._lastHistStatus) {
  if (['complete','stopped','interrupted'].includes(s.status))
    refreshRunHistory();
  window._lastHistStatus = s.status;
}
// Optional: refresh every N steps during run
if (s.running && s.train_step && s.train_step % 500 === 0) refreshRunHistory();
```
- Server: call `_auto_record_run('checkpoint')` or `record_engineering()` on `_save_model()` with agent + goal metadata.
- Fix `_infer_phase` for `vjepa` → `pretrain` or `mcf-vjepa`.

### 12.4 Stale infer + pending recipe

**Infer panel (screenshot):** shows step 2 / 0.92 MB — partially live if app7 infer.js; **app8 v3 lacks `inferContextPanel` grid**.

**Pending recipe “81.9 MB target” vs Active “24 GB”:**
- `pendingRecipeFromForm()` reads **form fields**; `boot()` hydrates form from `/api/health` defaults (stale 51m-era targets).
- During run, `renderStatus` **does not** call `updateRunPlaybook(null, s)` on app8.
- `onFormChange` only updates chinchilla table when user edits — **poll doesn’t sync pending vs active**.

**Fix:** Port app7 poll tail:
```javascript
if (typeof updateInferFromStatus === 'function') updateInferFromStatus(s);
if (typeof updateRunPlaybook === 'function') updateRunPlaybook(null, s);
if (typeof renderRunGoalLive === 'function') renderRunGoalLive(s);
```
- When `s.running`, hide or freeze **Pending** column (`recipePendingCol.style.display = 'none'`).
- Sync `targetGb` from status when live: `if (s.running) document.getElementById('targetGb').value = s.target_gb`.

### 12.5 HOST mislabel

app8 `/api/status` returns `"host": "app7.nextaura.us"` — **PUBLIC_URL copied from app7 deploy**. Confuses which machine you’re on. Set in `.gpu.env`: `PUBLIC_URL=https://app8…` or IP.

### 12.6 Priority fix order for next agent

1. **Sync app8 `public/` with app7** (workflow + infer + run-goal + poll hooks) — 1 deploy
2. **Fix poll history refresh** on stop/interrupt — 5 lines
3. **Auto-unlock or preset-reset workflow** on MCF recipe apply — UX
4. **Record history on checkpoint save** — learning loop
5. **V-JEPA row packing** — training actually completes (§6–7)
6. **`_infer_phase('vjepa')`** → meaningful phase labels

---

