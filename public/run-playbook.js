/** Run playbook — stream presets, footprint ladder, read the log below. */

const FOOTPRINT_10M = {
  nLayer: 8,
  nHead: 8,
  nEmbd: 256,
  blockSize: 256,
  ffnDim: 1024,
  vocabSize: 8192,
  tokenizerId: 'SupraLabs/StorySupra-10M',
  targetGb: 0.08,
  totalTrainSteps: 0,
  batchSize: 1024,
  microBatchSize: 256,
  gradAccumSteps: 4,
};

const FOOTPRINT_25M = {
  nLayer: 10,
  nHead: 8,
  nEmbd: 336,
  blockSize: 256,
  ffnDim: 1344,
  vocabSize: 8192,
  tokenizerId: 'SupraLabs/StorySupra-10M',
  targetGb: 0.2,
  totalTrainSteps: 600,
  batchSize: 1024,
  microBatchSize: 256,
  gradAccumSteps: 4,
};

const FOOTPRINT_50M = {
  nLayer: 8,
  nHead: 8,
  nEmbd: 512,
  blockSize: 256,
  ffnDim: 2048,
  vocabSize: 50257,
  tokenizerId: 'gpt2',
  targetGb: 1.1,
  totalTrainSteps: 0,
  batchSize: 1024,
  microBatchSize: 256,
  gradAccumSteps: 4,
};

const DEFAULT_MM_DATASET = 'multimodal-reasoning-lab/Zebra-CoT';
const DEFAULT_VJEPA_DATASET = 'rookierufus/epic-kitchens-vjepa';

/** MCF — Metamorphic Constraint Field footprints (ported from app8). */
const FOOTPRINT_MCF = {
  nLayer: 17,
  nHead: 15,
  nEmbd: 960,
  blockSize: 256,
  ffnDim: 3840,
  vocabSize: 50257,
  tokenizerId: 'gpt2',
  targetGb: 24.0,
  totalTrainSteps: 0,
  batchSize: 768,
  microBatchSize: 128,
  gradAccumSteps: 6,
};

/** MCF v2 — 360M · batch 256 · 1.5B token stop (~5.25 GB). */
const FOOTPRINT_MCF_V2 = {
  nLayer: 17,
  nHead: 15,
  nEmbd: 960,
  blockSize: 256,
  ffnDim: 3840,
  vocabSize: 50257,
  tokenizerId: 'gpt2',
  targetGb: 5.25,
  totalTrainSteps: 0,
  batchSize: 256,
  microBatchSize: 64,
  gradAccumSteps: 4,
};

/** MCF v2 HLS — CS2 + ego + sokoban (no mamba). */
const VJEPA_MIX_HLS = [
  'cbctr/cs2-10k-vjepa2-latents-300',
  'rookierufus/ego10k-vjepa-latents',
  'qinglinhou/sokoban-10k-vjepa2-tokenized',
];

/** MCF v2 world-model mix — HLS + robotics + physics + gameplay (11 streams). */
const VJEPA_MIX_WORLD = [
  ...VJEPA_MIX_HLS,
  'jialei02/libero_merged_no_noops_20hz',
  'quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings',
  'Silicon23/mujoco-kinematics-probing',
  'Swastikr/PhysSim-VLM-Dataset',
  'Swastikr/PhysSim-VLM-SFT-R2-Data',
  'agibot-world/AgiBotWorld2026',
  'HuberyLL/nms_hitl_world_model',
];

/** Legacy multi-dataset V-JEPA mix. */
const VJEPA_MIX_DEFAULT = [
  'qinglinhou/sokoban-10k-vjepa2-tokenized',
  'cbctr/cs2-10k-vjepa2-latents-300',
  'rookierufus/ego10k-vjepa-latents',
  'rookierufus/Vjepa_mamba_dataset',
];

function setArchField(id, value) {
  const el = document.getElementById(id);
  if (!el || value === undefined || value === null) return;
  el.value = String(value);
  if (typeof archUserSet !== 'undefined') archUserSet = true;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function enableParam(key, on) {
  const toggle = document.querySelector('.param-toggle[data-param="' + key + '"]');
  if (!toggle) return;
  toggle.checked = !!on;
  if (typeof toggleParam === 'function') toggleParam(key);
}

function setMixLocalJsonl(on) {
  const el = document.getElementById('mixLocalJsonl');
  if (!el) return;
  el.checked = !!on;
  if (typeof toggleLocalMix === 'function') toggleLocalMix();
  if (typeof onFormChange === 'function') onFormChange();
}

function setHfDataset(id) {
  const sel = document.getElementById('hfDataset');
  if (!sel || !id) return;
  if (!Array.from(sel.options).some((o) => o.value === id)) {
    const o = document.createElement('option');
    o.value = id;
    o.textContent = id;
    sel.appendChild(o);
  }
  sel.value = id;
  if (typeof hfDatasetUserSet !== 'undefined') hfDatasetUserSet = true;
  if (typeof onFormChange === 'function') onFormChange();
  if (typeof markWorkflowStreamPicked === 'function') markWorkflowStreamPicked();
  if (typeof updateHfProbeHint === 'function') updateHfProbeHint();
}

function fillRunGoal(title, detail) {
  const t = document.getElementById('runGoalTitle');
  const d = document.getElementById('runGoalDetail');
  if (t) t.value = title;
  if (d) d.value = detail;
  const steps = document.getElementById('totalTrainSteps');
  if (steps) steps.value = 0;
  const bytes = document.querySelector('input[name="runStopPrimary"][value="bytes"]');
  if (bytes) bytes.checked = true;
  if (typeof syncRunGoalHints === 'function') syncRunGoalHints();
  if (typeof renderRunWorkflow === 'function') renderRunWorkflow();
}

function applyFootprintPreset(preset, label) {
  setArchField('nLayer', preset.nLayer);
  setArchField('nHead', preset.nHead);
  setArchField('nEmbd', preset.nEmbd);
  setArchField('blockSize', preset.blockSize);
  enableParam('ffn_dim', true);
  setArchField('ffnDim', preset.ffnDim);
  enableParam('vocab_size', true);
  setArchField('vocabSize', preset.vocabSize);
  enableParam('tokenizer_id', true);
  const tok = document.getElementById('tokenizerId');
  if (tok) tok.value = preset.tokenizerId;
  setArchField('targetGb', preset.targetGb);
  enableParam('total_train_steps', true);
  setArchField('totalTrainSteps', preset.totalTrainSteps);
  setArchField('batchSize', preset.batchSize);
  if (typeof batchUserSet !== 'undefined') batchUserSet = true;
  enableParam('micro_batch_size', true);
  setArchField('microBatchSize', preset.microBatchSize);
  enableParam('grad_accum_steps', true);
  setArchField('gradAccumSteps', preset.gradAccumSteps);
  if (typeof syncFfnDim === 'function') syncFfnDim();
  if (typeof updateModelEstimate === 'function') updateModelEstimate();
  if (typeof syncRunGoalHints === 'function') syncRunGoalHints();
  if (typeof onFormChange === 'function') onFormChange();
  const note = document.getElementById('playbookFootprintNote');
  if (note) note.textContent = 'Active footprint: ' + label;
  if (typeof markWorkflowFootprintApplied === 'function') markWorkflowFootprintApplied();
}

function applyFootprint10M() {
  applyFootprintPreset(FOOTPRINT_10M, '~10M floor (8L×256d · StorySupra 8K)');
}

function applyFootprint25M() {
  pickDataSource('distill-merged', true);
  if (typeof setScaleMode === 'function') setScaleMode('finetune');
  applyFootprintPreset(FOOTPRINT_25M, '~25M distill student (10L×336d · StorySupra 8K · distill-merged)');
}

function applyFootprint50M() {
  applyFootprintPreset(FOOTPRINT_50M, '~50M scale (8L×512d · gpt2) — only after 10M smokes pass');
}

function scrollToDataSource() {
  const el = document.getElementById('playbookStep1Panel') || document.getElementById('sectionData');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function configureDataMixForSource(ds) {
  const running = !!(typeof lastActiveSnapshot !== "undefined" && lastActiveSnapshot && lastActiveSnapshot.running);
  if (running) return;
  const hfOnly = ds === "multimodal" || ds === "vjepa";
  const jsonlOnly =
    ds === "jsonl" ||
    ds === "local-corpus-mix" ||
    (typeof isSftFinetune === "function" && isSftFinetune(ds)) ||
    (typeof isDistillTopic === "function" && isDistillTopic(ds)) ||
    ds === "distill-merged";
  if (hfOnly) {
    setMixLocalJsonl(false);
    const mc = document.getElementById("mixLocalCorpora");
    if (mc) mc.checked = false;
    enableParam("hf_streams", false);
    ["hfStreamFable", "hfStreamSol", "hfStreamKimi", "hfStreamNemotron", "hfStreamMath", "hfStreamPref", "hfStreamFineweb"].forEach(
      (id) => {
        const el = document.getElementById(id);
        if (el) el.checked = false;
      }
    );
  } else if (ds === "mixed") {
    setMixLocalJsonl(true);
    enableParam("hf_streams", true);
    const defaults = {
      hfStreamFable: true,
      hfStreamSol: true,
      hfStreamKimi: false,
      hfStreamNemotron: false,
      hfStreamMath: false,
      hfStreamPref: false,
      hfStreamFineweb: false,
    };
    Object.keys(defaults).forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.checked = defaults[id];
    });
  } else if (jsonlOnly) {
    setMixLocalJsonl(false);
    enableParam("hf_streams", false);
  }
  if (typeof toggleLocalCorpus === "function") toggleLocalCorpus();
  if (typeof toggleLocalMix === "function") toggleLocalMix();
  if (typeof toggleHfStreams === "function") toggleHfStreams();
}

function scrollToStage1() {
  scrollToDataSource();
}

function pickDataSource(value, skipScroll) {
  const ds = document.getElementById("dataSource");
  if (!ds) return;
  ds.value = value;
  if (typeof dataSourceUserSet !== "undefined") dataSourceUserSet = true;
  window.__streamConfigured = true;
  if (typeof toggleDataSource === "function") toggleDataSource();
  configureDataMixForSource(value);
  if (typeof onFormChange === "function") onFormChange();
  if (typeof markWorkflowStreamPicked === "function") markWorkflowStreamPicked();
  if (!skipScroll) scrollToDataSource();
}

function setHfStreams(flags) {
  enableParam("hf_streams", true);
  const ids = {
    hfStreamFable: "fable",
    hfStreamSol: "sol",
    hfStreamKimi: "kimi",
    hfStreamNemotron: "nemotron",
    hfStreamMath: "math",
    hfStreamPref: "pref",
    hfStreamFineweb: "fineweb",
  };
  Object.keys(ids).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.checked = !!flags[ids[id]];
  });
  if (typeof toggleHfStreams === "function") toggleHfStreams();
  if (typeof onFormChange === "function") onFormChange();
}

function apply51mFinetuneBase() {
  applyFootprint50M();
  setArchField("learningRate", 0.00001);
  setArchField("totalTrainSteps", 0);
  setArchField("batchSize", 256);
  enableParam("learning_rate", true);
  enableParam("target_gb", true);
  enableParam("total_train_steps", true);
  enableParam("lr_schedule", true);
  const lr = document.getElementById("lrSchedule");
  if (lr) lr.value = "constant";
  const ec = document.getElementById("epochCycles");
  const ep = document.getElementById("jsonlEpochs");
  if (ec) {
    ec.checked = false;
    enableParam("epoch_cycles", true);
  }
  if (ep) {
    ep.disabled = true;
    ep.value = "1";
  }
  if (typeof toggleEpochCycles === "function") toggleEpochCycles();
  const note = document.getElementById("playbookFootprintNote");
  if (note) {
    note.textContent =
      "Load nextaura-51m-step3898.pt first — keep 8L×512d gpt2 arch from checkpoint.";
  }
}

function setEpochCycles(on, epochs) {
  enableParam('epoch_cycles', true);
  const ec = document.getElementById('epochCycles');
  if (ec) {
    ec.checked = !!on;
    ec.disabled = false;
  }
  const ep = document.getElementById('jsonlEpochs');
  if (ep) {
    ep.disabled = !on;
    if (epochs) ep.value = String(epochs);
  }
  if (typeof toggleEpochCycles === 'function') toggleEpochCycles();
}

function applyMcfCommon() {
  if (typeof setScaleMode === 'function') setScaleMode('pretrain', true);
  enableParam('scalar_input_projection', true);
  const sp = document.getElementById('useScalarProjection');
  if (sp) sp.checked = true;
  enableParam('gradient_checkpointing', true);
  const gc = document.getElementById('useGradCkpt');
  if (gc) gc.checked = true;
  enableParam('use_flash_attention', true);
  const fa = document.getElementById('useFlashAttention');
  if (fa) fa.checked = true;
  setArchField('learningRate', 0.0002);
  setArchField('lrSchedule', 'cosine');
  setArchField('warmupSteps', 500);
  setArchField('mixedPrecision', 'bf16');
  enableParam('warmup_steps', true);
  enableParam('lr_schedule', true);
  enableParam('mixed_precision', true);
  setMixLocalJsonl(false);
  const mc = document.getElementById('mixLocalCorpora');
  if (mc) mc.checked = false;
}

function applyRunPreset(key) {
  window.vjepaPackRows = 0;
  window.vjepaPackMinChars = 4000;

  const presets = {
    mcf_recipe_v2_hls: {
      label: 'MCF v2 HLS — latent packing · 360M · 1.5B tokens',
      run() {
        applyMcfCommon();
        pickDataSource('vjepa', true);
        window.vjepaDatasetIds = VJEPA_MIX_WORLD.slice();
        window.vjepaPackRows = 128;
        window.vjepaPackMinChars = 4000;
        setHfDataset(VJEPA_MIX_WORLD[0]);
        setEpochCycles(true, 3);
        applyFootprintPreset(
          FOOTPRINT_MCF_V2,
          '~360M MCF v2 HLS · 17L×960d · batch 256 · pack 128 · 1.5B tok'
        );
        fillRunGoal(
          'MCF v2 — Hierarchical Latent Stack (HLS)',
          'Latent packing (128 frames/block) + GCT physics injector + dual-target loss target. ' +
            'Streams: 11-way world mix (CS2/ego/sokoban + LIBERO/behavior/MuJoCo/PhysSim/AgiBot/NMS). ' +
            'Stop ~1.5B tokens (~5.25 GB) — not 24 GB byte trap. ' +
            'Restart: stop → reset-session → start resume:false. ' +
            'Success: step 10 within 10 min, >50 MB/hr, loss stable-decreasing (not →0.15).'
        );
      },
    },
    mcf_recipe: {
      label: 'MCF — Metamorphic Constraint Field (dual-stream, 5 GB)',
      run() {
        applyMcfCommon();
        pickDataSource('mixed', true);
        setHfStreams({
          fable: false,
          sol: true,
          kimi: false,
          nemotron: true,
          math: true,
          pref: false,
          fineweb: false,
        });
        applyFootprintPreset(
          FOOTPRINT_MCF,
          '~360M MCF braid · 17L×960d · batch 768 · scalar 12d ON'
        );
        fillRunGoal(
          'MCF — Metamorphic Constraint Field (dual-stream smoke)',
          'L = Entropy_human + λ·PhysicsConstraintError(Δx). ' +
            'Stream B (symbolic): sol + nemotron + unsolved-math HF streams with scalar 12d bridge.'
        );
      },
    },
    mcf_recipe_vjepa: {
      label: 'MCF Stream A — V-JEPA physics field (24 GB · 360M · multi-HF)',
      run() {
        applyMcfCommon();
        pickDataSource('vjepa', true);
        window.vjepaDatasetIds = VJEPA_MIX_DEFAULT.slice();
        setHfDataset(window.defaultVjepaDataset || DEFAULT_VJEPA_DATASET);
        applyFootprintPreset(
          FOOTPRINT_MCF,
          '~360M MCF Stream A · V-JEPA mix ×' +
            VJEPA_MIX_DEFAULT.length +
            ' · batch 768 · scalar 12d ON'
        );
        fillRunGoal(
          'MCF Stream A — continuous physics-logic field (V-JEPA multi)',
          'Legacy interleaved V-JEPA (includes mamba). ~24 GB — known slow on short metadata rows.'
        );
      },
    },
    transformer_baseline_51m: {
      label: "Transformer baseline — 51M HF SFT (1.1 GB)",
      run() {
        if (typeof setScaleMode === "function") setScaleMode("finetune", true);
        pickDataSource("mixed", true);
        setMixLocalJsonl(false);
        const mc = document.getElementById("mixLocalCorpora");
        if (mc) mc.checked = false;
        const mix = document.getElementById("jsonlMix");
        if (mix) mix.value = "0";
        setHfStreams({
          fable: true,
          sol: true,
          kimi: true,
          nemotron: true,
          math: true,
          pref: false,
          fineweb: false,
        });
        enableParam("scalar_input_projection", true);
        const sp = document.getElementById("useScalarProjection");
        if (sp) sp.checked = false;
        apply51mFinetuneBase();
        setArchField("targetGb", 1.1);
        fillRunGoal(
          "Transformer baseline — 51M HF SFT (1.1 GB)",
          "Fixed-depth causal GPT (2017): parallel HF byte streams, causal mask at train time. " +
            "Mixed Fable + Sol + Nemotron + UnsolvedMath on nextaura-51m-step3898.pt. Byte stop 1.1 GB."
        );
      },
    },
    hf_qwen_finetune: {
      label: "HF native SFT — Qwen / any causal LM (2.5 GB)",
      run() {
        if (typeof setScaleMode === "function") setScaleMode("finetune", true);
        pickDataSource("mixed", true);
        setMixLocalJsonl(false);
        setHfStreams({
          fable: true,
          sol: true,
          kimi: true,
          nemotron: true,
          math: true,
          pref: true,
          fineweb: false,
        });
        enableParam("scalar_input_projection", true);
        const sp = document.getElementById("useScalarProjection");
        if (sp) sp.checked = false;
        enableParam("gradient_checkpointing", true);
        const gc = document.getElementById("useGradCkpt");
        if (gc) gc.checked = true;
        setArchField("learningRate", "0.00001");
        setArchField("lrSchedule", "constant");
        setArchField("batchSize", "512");
        setArchField("microBatchSize", "1");
        setArchField("gradAccumSteps", "512");
        setArchField("mixedPrecision", "bf16");
        setArchField("targetGb", 2.5);
        fillRunGoal(
          "HF native SFT — Qwen / any causal LM",
          "Import HF model first (checkpoints panel). Mixed HF streams, no scalar/V-JEPA. " +
            "Practical 2.5 GB SFT slice — not full Chinchilla for multi-B models."
        );
      },
    },
    nextaura_novel_51m: {
      label: "NextAura novel recipe — causal + world-model mix",
      run() {
        if (typeof setScaleMode === "function") setScaleMode("finetune", true);
        pickDataSource("mixed", true);
        setMixLocalJsonl(true);
        const mix = document.getElementById("jsonlMix");
        if (mix) mix.value = "30";
        setHfStreams({
          fable: false,
          sol: true,
          kimi: true,
          nemotron: true,
          math: true,
          pref: true,
          fineweb: false,
        });
        enableParam("scalar_input_projection", true);
        const sp = document.getElementById("useScalarProjection");
        if (sp) sp.checked = true;
        enableParam("gradient_checkpointing", true);
        const gc = document.getElementById("useGradCkpt");
        if (gc) gc.checked = true;
        apply51mFinetuneBase();
        setArchField("targetGb", 1.1);
        fillRunGoal(
          "NextAura novel recipe — 51M causal + world-model SFT",
          "Causal left-to-right GPT + HF text/math/code streams + preference RM data + scalar 12d (V-JEPA path). " +
            "Local JSONL spice 30%. Geometry/embodied sim via future distill slice — not distill-live."
        );
      },
    },
    multimodal_10m: {
      label: '10M Multimodal smoke (0.08 GB)',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('pretrain', true);
        pickDataSource('multimodal', true);
        setHfDataset(window.defaultMultimodalDataset || DEFAULT_MM_DATASET);
        setMixLocalJsonl(false);
        applyFootprint10M();
        fillRunGoal(
          '10M multimodal smoke — Zebra-CoT',
          'Byte-stop ~0.08 GB on Zebra-CoT multimodal stream. total_train_steps=0. Confirm stream hits in terminal, then scale.'
        );
      },
    },
    multimodal_quarter: {
      label: '10M · 0.25× Chinchilla (~0.2 GB)',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('pretrain', true);
        pickDataSource('multimodal', true);
        setHfDataset(window.defaultMultimodalDataset || DEFAULT_MM_DATASET);
        setMixLocalJsonl(false);
        applyFootprint10M();
        setArchField('targetGb', 0.2);
        fillRunGoal(
          '10M multimodal — 0.25× Chinchilla byte stop',
          'Target ~0.2 GB (~0.25× Chinchilla optimal for 10M). Still a short run — raise Target GB for real pretrain.'
        );
      },
    },
    vjepa_10m: {
      label: '10M V-JEPA smoke',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('pretrain', true);
        pickDataSource('vjepa', true);
        setHfDataset(window.defaultVjepaDataset || DEFAULT_VJEPA_DATASET);
        setMixLocalJsonl(false);
        enableParam('scalar_input_projection', true);
        const sp = document.getElementById('useScalarProjection');
        if (sp) sp.checked = true;
        applyFootprint10M();
        fillRunGoal(
          '10M V-JEPA smoke',
          'Byte-stop ~0.08 GB on V-JEPA latents with scalar 12d projection ON. total_train_steps=0.'
        );
      },
    },
    distill_25m: {
      label: '25M Distill merged',
      run() {
        applyFootprint25M();
        fillRunGoal(
          '25M distill student on merged corpus',
          'Finetune distill-merged.jsonl at ~25M footprint. Step cap 600 or byte target 0.2 GB.'
        );
      },
    },
    text_mixed: {
      label: 'Text mixed streams',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('pretrain', true);
        pickDataSource('mixed', true);
        setMixLocalJsonl(true);
        applyFootprint10M();
        fillRunGoal(
          '10M text mixed smoke',
          'Mixed local JSONL + HF text streams (fable/sol/kimi). Byte-stop ~0.08 GB.'
        );
      },
    },
    sft_intel: {
      label: 'SFT intelligence',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('finetune', true);
        pickDataSource('sft-intelligence', true);
        setArchField('targetGb', 0.05);
        setArchField('totalTrainSteps', 400);
        fillRunGoal(
          'SFT intelligence seeds',
          'Finetune on sft-full-merged.jsonl. Low LR, epoch cycles, step cap 400.'
        );
      },
    },
    full_sft_51m: {
      label: 'Full 51M SFT finetune',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('finetune', true);
        pickDataSource('sft-intelligence', true);
        setArchField('targetGb', 1.1);
        setArchField('learningRate', 0.00001);
        setArchField('totalTrainSteps', 0);
        setArchField('batchSize', 256);
        const lr = document.getElementById('lrSchedule');
        if (lr) lr.value = 'constant';
        const ec = document.getElementById('epochCycles');
        const ep = document.getElementById('jsonlEpochs');
        if (ec) {
          ec.checked = true;
          enableParam('epoch_cycles', true);
        }
        if (ep) {
          ep.disabled = false;
          ep.value = '15';
        }
        enableParam('learning_rate', true);
        enableParam('target_gb', true);
        enableParam('total_train_steps', true);
        enableParam('lr_schedule', true);
        enableParam('epoch_cycles', true);
        fillRunGoal(
          '51M full SFT finetune from step3898 pretrain base',
          'Load nextaura-51m-step3898.pt first. Target 1.1 GB (0.25× extended). sft-full-merged.jsonl. Not smoke.'
        );
        const note = document.getElementById('playbookFootprintNote');
        if (note) {
          note.textContent =
            'Load nextaura-51m-step3898.pt — keep 8L×512d gpt2 arch from checkpoint. Do NOT click 10M floor.';
        }
      },
    },
    muse_glimmer_sft: {
      label: 'Muse-Glimmer-30B QLoRA SFT v2',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('finetune', true);
        pickDataSource('nemotron-swe', true);
        setArchField('learningRate', 0.00001);
        setArchField('totalTrainSteps', 800);
        enableParam('learning_rate', true);
        enableParam('total_train_steps', true);
        fillRunGoal(
          'Muse-Glimmer-30B QLoRA SFT v2 — coding agent',
          'External Unsloth job (not GPT TrainEngine). GUI rules: lr 1e-5, 2 epochs, abort if <200 usable rows, early-stop if loss < 0.05 after step 40. Format Nemotron messages+tools. packing=False. Launch: muse/start_sft_v2.sh'
        );
        const note = document.getElementById('playbookFootprintNote');
        if (note) {
          note.textContent =
            'Muse SFT is /data/muse/sft_qlora.py via muse-venv — Stage 1 Start does not launch it. sft-v1 memorized 9 packed crumbs; v2 keeps unique formatted traces.';
        }
      },
    },
    nemotron_swe_finetune: {
      label: '360M Nemotron SWE v3.5 SFT',
      run() {
        if (typeof setScaleMode === 'function') setScaleMode('finetune', true);
        pickDataSource('nemotron-swe', true);
        setArchField('targetGb', 2.4);
        setArchField('learningRate', 0.00001);
        setArchField('totalTrainSteps', 0);
        setArchField('batchSize', 256);
        setArchField('microBatchSize', 64);
        setArchField('gradAccumSteps', 4);
        const lr = document.getElementById('lrSchedule');
        if (lr) lr.value = 'constant';
        const ec = document.getElementById('epochCycles');
        const ep = document.getElementById('jsonlEpochs');
        if (ec) {
          ec.checked = true;
          enableParam('epoch_cycles', true);
        }
        if (ep) {
          ep.disabled = false;
          ep.value = '3';
        }
        enableParam('learning_rate', true);
        enableParam('target_gb', true);
        enableParam('total_train_steps', true);
        enableParam('lr_schedule', true);
        enableParam('epoch_cycles', true);
        fillRunGoal(
          '360M SFT — Nemotron SWE v3.5 agentic coding',
          'Load 360M pretrain base first (17L×960d). HF stream nvidia/Nemotron-SFT-SWE-v3.5 · ~5.1k SWE trajectories · target 2.4 GB (0.1× Chinchilla).'
        );
        const note = document.getElementById('playbookFootprintNote');
        if (note) {
          note.textContent =
            'Load latest 360M .pt — keep 17L×960d from checkpoint. Dataset: messages + tools from OpenCode harness.';
        }
      },
    },
  };
  const preset = presets[key];
  if (!preset) return;
  preset.run();
  const note = document.getElementById('playbookFootprintNote');
  if (note) note.textContent = 'Preset applied: ' + preset.label + ' — review Stage 1 below or edit manually.';
  if (typeof updateHfProbeHint === 'function') updateHfProbeHint();
  if (typeof renderRunWorkflow === 'function') renderRunWorkflow();
  scrollToStage1();
}

function updateHfProbeHint(health) {
  const el = document.getElementById('hfProbeHint');
  if (!el) return;
  const probe = (health && health.hf_dataset_probe) || window.hfProbeStatus || {};
  const hf = (document.getElementById('hfDataset') || {}).value;
  const p = hf && probe[hf];
  if (p && p.ok) {
    el.textContent =
      'HF probe OK · ' + hf + (p.text_preview ? ' · ' + String(p.text_preview).slice(0, 72) + '…' : '');
    el.style.color = '#86efac';
  } else if (p && !p.ok) {
    el.textContent = 'HF probe failed for ' + hf + ': ' + String(p.error || 'no usable text').slice(0, 100);
    el.style.color = '#f87171';
  } else if (hf) {
    el.textContent = 'Probing ' + hf + '… (refresh page if this persists)';
    el.style.color = '#94a3b8';
  } else {
    el.textContent = 'Pick a dataset — verified streams show ✓ in the dropdown.';
    el.style.color = '#64748b';
  }
}

function updateRunPlaybook(health, status) {
  const mm = (health && health.multimodal_datasets_verified) || window.multimodalDatasetsVerified || (health && health.multimodal_datasets) || window.multimodalDatasets || [];
  const vj = (health && health.vjepa_datasets_verified) || window.vjepaDatasetsVerified || (health && health.vjepa_datasets) || window.vjepaDatasets || [];
  const mmEl = document.getElementById('playbookMmCount');
  const vjEl = document.getElementById('playbookVjepaCount');
  if (mmEl) mmEl.textContent = String(mm.length);
  if (vjEl) vjEl.textContent = String(vj.length);

  const ds = (document.getElementById('dataSource') || {}).value || 'mixed';
  const dsEl = document.getElementById('playbookDataSource');
  if (dsEl) {
    const labels = {
      multimodal: 'Multimodal HF stream',
      vjepa: 'V-JEPA HF stream',
      mixed: 'Mixed text streams',
      fineweb: 'FineWeb text',
      'fable-traces': 'Fable text',
      'distill-merged': 'Distill merged',
      'distill-coding': 'Distill coding',
      'distill-physics': 'Distill physics',
      'distill-math': 'Distill math',
      'sft-intelligence': 'SFT intelligence',
    };
    const hf = (document.getElementById('hfDataset') || {}).value;
    dsEl.textContent = (labels[ds] || ds) + (hf && (ds === 'multimodal' || ds === 'vjepa') ? ' · ' + hf : '');
  }

  const params = Number((document.getElementById('modelParams') || {}).value || 0);
  const footEl = document.getElementById('playbookFootprintLive');
  if (footEl && params > 0) {
    const m = params / 1e6;
    footEl.textContent = m.toFixed(1) + 'M params · ' + (document.getElementById('archHint') || {}).value;
  }

  if (status && typeof renderRunGoalLive === 'function') renderRunGoalLive(status);
  if (typeof updateHfProbeHint === 'function') updateHfProbeHint(health);

  const agentEl = document.getElementById('playbookAgentStatus');
  if (agentEl) {
    fetch('/api/agent/drive-status')
      .then((r) => r.json())
      .then((d) => {
        const phase = d.phase || d.phase_title || 'idle';
        const rm = d.last_rm != null ? ' · RM ' + Number(d.last_rm).toFixed(3) : '';
        const ck = d.last_checkpoint ? ' · ckpt ' + d.last_checkpoint : '';
        agentEl.textContent =
          'Autonomous agent (' +
          (d.agent || 'drive-agent') +
          '): phase ' +
          phase +
          ' [' +
          (d.phase_index != null ? d.phase_index : 0) +
          ']' +
          rm +
          ck;
      })
      .catch(() => {
        agentEl.textContent = 'Autonomous agent: offline (start scripts/drive_agent.py on GPU)';
      });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  updateRunPlaybook(null, null);
});
