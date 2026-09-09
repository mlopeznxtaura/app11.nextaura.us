# app9-nextaura-us — Merge Notes

Merged 2026-09-04 from:
- **A = app7-nextaura-us** (complete GUI, Nemotron SWE, support modules)
- **B = app8-nextaura-us** (newer training core: speed metrics, V-JEPA world mix, GradScaler fix, checkpoint auto-prune)

## From B (base — entire repo copied)
- `server.py` (training core; `_prune_old_checkpoints`, `MAX_CHECKPOINTS_KEEP`, `_speed_suffix`, resume-defaults fix in `start()` preserved)
- `train_engine.py`, `train_gpu.py`, `model.py`, `run_history.py`, `reward_eval.py`, `text_quality.py`, `scalar_features.py`, `tokenizer_utils.py`, `hf_datasets.py`, `hf_stream_loader.py`, `intelligence_expand.py`, `multimodal_data.py`
- `vjepa_local_cache.py`, `vjepa_pack.py`, `world_model_mix.py`
- `scripts/`, `deploy/`, `docs/`, `proxy/`, `parked/`, `requirements.txt`, `Dockerfile`, `bootstrap_gpu.sh`
- `worker.js` / `wrangler.toml` (rebranded to app9: worker `nextaura-app9-us`, route `app9.nextaura.us`, ORIGIN `http://150-239-227-60.sslip.io`)

## From A (copied wholesale — B lacked them)
- `preflight.py`, `timemoe_baseline.py`, `eval_orchestrator.py`, `object_storage.py`, `sft_topics.py`, `distill_teachers.py`, `hf_model_loader.py`, `platform_index.py` (imported by B's server.py but missing from B's folder)
- `convert_model.py` — A's version taken because B's lacked `import_hf_causal`, which B's own `server.py` imports
- `data_jsonl.py` — A's version taken (strict superset; B's was A's minus the 38-line `nemotron_swe_training_text`)
- `public/` — A's whole GUI (infer.js v12 cheat sheets, eval panel, `eval-panel.js`, `sft-generate.js`, `distill-generate.js`, `recipe-cheatsheet.js`, `mm-vjepa-recipes.txt`)

## Ported (merged by hand)
- **Nemotron SWE feature** from A's `server.py` into B's `server.py`: `nemotron_swe_training_text` import, `NEMOTRON_SWE_DATASET_ID`, `"nemotron-swe"` in `DATA_SOURCES` / dataset map / `needs_hf`, `swe_hits` in StreamState (init, `apply_checkpoint`, `checkpoint_dict`, `snapshot`, both reset sites), `_nemotron_swe_stream()`, `swe` hits label, finetune-mode + suggested-prompts branches, start banner, and the `_stream_and_train` `nemotron-swe` branch.
- **MCF / V-JEPA recipe panel** from B's `public/index.html` into A's `index.html` (rebranded "app9 slot"), plus the MCF presets (`mcf_recipe_v2_hls`, `mcf_recipe`, `mcf_recipe_vjepa`), `FOOTPRINT_MCF(_V2)`, `VJEPA_MIX_*`, `applyMcfCommon()`, `setEpochCycles()` from B's `run-playbook.js` into A's `run-playbook.js` (adapted to A's 2-arg `fillRunGoal`).

## Branding
- `app8.nextaura.us` → `app9.nextaura.us`, `nextaura-app8-us` → `nextaura-app9-us`, `APP_ID` → `app9`, COS bucket default `nextaura-app8-stage1` → `nextaura-app9-stage1`, COS prefix `app8/datasets/vjepa/` → `app9/datasets/vjepa/` in: `server.py`, `agent_contract.py`, `worker.js`, `wrangler.toml`, `vjepa_local_cache.py`, `public/agent.json`, `public/index.html` (host badge).
- References to other apps (app7 lineage, handoff docs) left intact.

## Conflicts resolved
1. `convert_model.py`: B's version missing `import_hf_causal` that B's own server imports → took A's.
2. `data_jsonl.py`: B's missing `nemotron_swe_training_text` → took A's (verified superset).
3. `public/index.html` / `run-playbook.js`: A's GUI is newer overall but lacked B's MCF presets → A's base + MCF section/presets ported from B.
4. `platform_index.py`: imported by B's `server.py` but absent from B's folder → copied from A.
5. `public/changelog.json`, `public/run-goal.js`: A's versions kept per "public/ = A's" rule (B's were newer timestamps but A's GUI is the intended base).

## Not copied (by design)
`.pt` checkpoints, `.gpu.env`, `data/*.jsonl` corpora, `__pycache__`, `node_modules`, `.git`, `.wrangler`, `checkpoints/`.

## Sanity
- `python -m py_compile` passed on `server.py`, `train_engine.py`, and all copied modules.
- Grep checks: `_prune_old_checkpoints` ✓, `MAX_CHECKPOINTS_KEEP` ✓, `_speed_suffix` ✓, `_nemotron_swe_stream` ✓, `import_hf_causal` (import in server.py + def in convert_model.py) ✓.

## TODO before deploy
- `deploy/`, `scripts/`, `worker-fit.js`, `wrangler-fit.toml`, `CHANGELOG.md`, `docs/` still contain app8 paths/service names — update when actually deploying.
