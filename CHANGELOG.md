# Changelog — app8.nextaura.us

## 2026-09-03 — MCF v2 HLS release

### Added
- **MCF v2 Hierarchical Latent Stack (HLS)** preset in GUI theory panel (top preset button).
- V-JEPA **latent row packing** (`vjepa_pack.py`): 128 frames/block for throughput fix.
- **GCT scalar bridge** via `gct_scalar_mean` on packed blocks.
- `scripts/start_mcf_v2.py` deployment script (stop → reset-session → start, resume:false).
- Full GUI sync: run playbook workflow locks, run goal live panel, infer v10 hooks, changelog footer.
- Private GitHub repo `app8.nextaura.us`.

### Changed
- MCF v2 target: **1.5B tokens (~5.25 GB)** instead of 24 GB byte trap for short V-JEPA rows.
- Batch **256** (micro 64 × grad 4) for dense packed input vs legacy batch 768.
- V-JEPA mix for v2: CS2 + ego10k + sokoban (no mamba).
- Run history phase: `vjepa` → `mcf` for 360M runs.
- Host identity: `app8.nextaura.us` (was incorrectly reporting app7).

### Fixed
- Run history refresh on **stop/interrupted** (not only complete).
- Workflow locks: app8 storage key + auto-lock on valid preset apply.
- Stale infer/recipe: poll hooks for `updateInferFromStatus`, `updateRunPlaybook`, `renderRunGoalLive`.

### Known limitations
- Dual-target loss (α·L_latent + β·L_scalar) and ODE Stream A braid are **documented targets**; packing + scalar GCT are implemented in this release.
- Legacy MCF Stream A (24 GB · batch 768) preset retained for comparison.
