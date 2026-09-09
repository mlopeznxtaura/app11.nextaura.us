/** Multimodal + V-JEPA recipe presets for app7 training lab UI. */
const MM_VJEPA_RECIPES = [
  {
    id: 'mm-smoke',
    family: 'multimodal',
    title: 'Multimodal smoke',
    blurb: 'Fast HF stream check — textbook captions, ~50MB.',
    lines: ['data: multimodal', 'multimodal-reasoning-lab/Zebra-CoT', '10M floor · 0.05 GB · lr 3e-4', 'batch 512 · StorySupra 8K · steps=0'],
    apply: {
      scaleMode: 'pretrain',
      dataSource: 'multimodal',
      hfDataset: 'multimodal-reasoning-lab/Zebra-CoT',
      targetGb: 0.05,
      learningRate: 0.0003,
      batchSize: 512,
      microBatchSize: 128,
      gradAccumSteps: 4,
      warmupSteps: 100,
      mixedPrecision: 'bf16',
      useFlashAttention: true,
      scalarProjection: false,
      tokenizerId: 'SupraLabs/StorySupra-10M',
      footprint10M: true,
      totalTrainSteps: 0,
    },
  },
  {
    id: 'mm-textbook',
    family: 'multimodal',
    title: 'Textbook pretrain',
    blurb: 'Default multimodal stream — Chinchilla-scale caption/text mix.',
    lines: ['data: multimodal', 'DAMO-NLP-SG/multimodal_textbook', 'pretrain · 3.33 GB · lr 3e-4', 'batch 1024 · micro 256×4 · cosine'],
    apply: {
      scaleMode: 'pretrain',
      dataSource: 'multimodal',
      hfDataset: 'multimodal-reasoning-lab/Zebra-CoT',
      targetGb: 3.33,
      learningRate: 0.0003,
      batchSize: 1024,
      microBatchSize: 256,
      gradAccumSteps: 4,
      warmupSteps: 500,
      lrSchedule: 'cosine',
      mixedPrecision: 'bf16',
      useFlashAttention: true,
      scalarProjection: false,
      tokenizerId: 'gpt2',
    },
  },
  {
    id: 'mm-zebra-cot',
    family: 'multimodal',
    title: 'Zebra-CoT reasoning',
    blurb: 'Multimodal chain-of-thought — lighter LR for SFT-style tuning.',
    lines: ['data: multimodal', 'multimodal-reasoning-lab/Zebra-CoT', 'fine-tune · 1 GB · lr 2e-5', 'batch 512 · micro 128×4'],
    apply: {
      scaleMode: 'finetune',
      dataSource: 'multimodal',
      hfDataset: 'multimodal-reasoning-lab/Zebra-CoT',
      targetGb: 1,
      learningRate: 0.00002,
      batchSize: 512,
      microBatchSize: 128,
      gradAccumSteps: 4,
      warmupSteps: 200,
      lrSchedule: 'cosine',
      mixedPrecision: 'bf16',
      useFlashAttention: true,
      scalarProjection: false,
      tokenizerId: 'gpt2',
    },
  },
  {
    id: 'vjepa-smoke',
    family: 'vjepa',
    title: 'V-JEPA smoke',
    blurb: 'Latent + metadata stream check — scalar 12d projection required.',
    lines: ['data: vjepa', 'rookierufus/epic-kitchens-vjepa', '10M floor · 0.1 GB · lr 1e-4', 'scalar proj ON · steps=0'],
    apply: {
      scaleMode: 'pretrain',
      dataSource: 'vjepa',
      hfDataset: 'rookierufus/epic-kitchens-vjepa',
      targetGb: 0.1,
      learningRate: 0.0001,
      batchSize: 512,
      microBatchSize: 128,
      gradAccumSteps: 4,
      warmupSteps: 100,
      mixedPrecision: 'bf16',
      useFlashAttention: true,
      scalarProjection: true,
      tokenizerId: 'SupraLabs/StorySupra-10M',
      footprint10M: true,
      totalTrainSteps: 0,
    },
  },
  {
    id: 'vjepa-epic',
    family: 'vjepa',
    title: 'Epic Kitchens latents',
    blurb: 'Video latent pretrain — default V-JEPA dataset on L40S.',
    lines: ['data: vjepa', 'rookierufus/epic-kitchens-vjepa', 'pretrain · 3.33 GB · lr 1e-4', 'scalar proj ON · flash attn'],
    apply: {
      scaleMode: 'pretrain',
      dataSource: 'vjepa',
      hfDataset: 'rookierufus/epic-kitchens-vjepa',
      targetGb: 3.33,
      learningRate: 0.0001,
      batchSize: 1024,
      microBatchSize: 256,
      gradAccumSteps: 4,
      warmupSteps: 500,
      lrSchedule: 'cosine',
      mixedPrecision: 'bf16',
      useFlashAttention: true,
      scalarProjection: true,
      tokenizerId: 'gpt2',
    },
  },
  {
    id: 'vjepa-libero',
    family: 'vjepa',
    title: 'LIBERO world latents',
    blurb: 'Robotics world-model latents — smaller batches for dense vectors.',
    lines: ['data: vjepa', 'Adjimavo/libero_world_latents_vjepa_m4', 'pretrain · 1 GB · lr 8e-5', 'scalar proj ON · micro 64×8'],
    apply: {
      scaleMode: 'pretrain',
      dataSource: 'vjepa',
      hfDataset: 'Adjimavo/libero_world_latents_vjepa_m4',
      targetGb: 1,
      learningRate: 0.00008,
      batchSize: 512,
      microBatchSize: 64,
      gradAccumSteps: 8,
      warmupSteps: 300,
      lrSchedule: 'cosine',
      mixedPrecision: 'bf16',
      useFlashAttention: true,
      scalarProjection: true,
      tokenizerId: 'gpt2',
    },
  },
];

function setFieldValue(id, value) {
  const el = document.getElementById(id);
  if (!el || value === undefined || value === null) return;
  el.value = String(value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function setCheckbox(id, checked) {
  const el = document.getElementById(id);
  if (!el) return;
  el.checked = !!checked;
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

function setParamEnabled(key, on) {
  const toggle = document.querySelector('.param-toggle[data-param="' + key + '"]');
  if (!toggle) return;
  toggle.checked = !!on;
  if (typeof toggleParam === 'function') toggleParam(key);
}

function applyMmVjepaRecipe(recipeId) {
  const recipe = MM_VJEPA_RECIPES.find((r) => r.id === recipeId);
  if (!recipe) return;
  const a = recipe.apply;
  if (a.scaleMode && typeof setScaleMode === 'function') setScaleMode(a.scaleMode, true);
  if (a.dataSource) {
    const ds = document.getElementById('dataSource');
    if (ds) {
      ds.value = a.dataSource;
      if (typeof dataSourceUserSet !== 'undefined') dataSourceUserSet = true;
      if (typeof toggleDataSource === 'function') toggleDataSource();
    }
  }
  if (a.hfDataset) {
    if (typeof populateHfDatasetOptions === 'function') populateHfDatasetOptions(a.dataSource);
    const hf = document.getElementById('hfDataset');
    if (hf) {
      hf.value = a.hfDataset;
      if (typeof hfDatasetUserSet !== 'undefined') hfDatasetUserSet = true;
    }
  }
  if (a.footprint10M && typeof applyFootprint10M === 'function') {
    applyFootprint10M();
  }
  setFieldValue('targetGb', a.targetGb);
  setFieldValue('learningRate', a.learningRate);
  if (typeof lrUserSet !== 'undefined') lrUserSet = true;
  setFieldValue('batchSize', a.batchSize);
  if (typeof batchUserSet !== 'undefined') batchUserSet = true;
  setParamEnabled('micro_batch_size', true);
  setFieldValue('microBatchSize', a.microBatchSize);
  setParamEnabled('grad_accum_steps', true);
  setFieldValue('gradAccumSteps', a.gradAccumSteps);
  setParamEnabled('warmup_steps', true);
  setFieldValue('warmupSteps', a.warmupSteps);
  if (a.lrSchedule) {
    setParamEnabled('lr_schedule', true);
    setFieldValue('lrSchedule', a.lrSchedule);
  }
  if (a.mixedPrecision) {
    setParamEnabled('mixed_precision', true);
    setFieldValue('mixedPrecision', a.mixedPrecision);
  }
  setParamEnabled('use_flash_attention', true);
  setCheckbox('useFlashAttention', a.useFlashAttention !== false);
  setParamEnabled('scalar_input_projection', !!a.scalarProjection);
  setCheckbox('useScalarProjection', !!a.scalarProjection);
  if (a.tokenizerId) {
    setParamEnabled('tokenizer_id', true);
    const tok = document.getElementById('tokenizerId');
    if (tok) tok.value = a.tokenizerId;
  }
  if (a.totalTrainSteps != null) {
    setParamEnabled('total_train_steps', true);
    setFieldValue('totalTrainSteps', a.totalTrainSteps);
  }
  if (typeof updateModelEstimate === 'function') updateModelEstimate();
  if (typeof onFormChange === 'function') onFormChange();
  const status = document.getElementById('mmVjepaRecipeStatus');
  if (status) status.textContent = 'Applied: ' + recipe.title + ' — review Live recipe, then Start.';
}

function renderMmVjepaRecipeCards() {
  const grid = document.getElementById('mmVjepaRecipeGrid');
  if (!grid) return;
  grid.innerHTML = MM_VJEPA_RECIPES.map((recipe) => {
    const tag = recipe.family === 'vjepa' ? 'V-JEPA' : 'Multimodal';
    const lines = recipe.lines.map((line) => '<div class="mm-vjepa-line">' + line + '</div>').join('');
    return (
      '<article class="mm-vjepa-card" data-family="' + recipe.family + '">' +
      '<div class="mm-vjepa-card-head"><span class="mm-vjepa-tag">' + tag + '</span><strong>' + recipe.title + '</strong></div>' +
      '<p class="mm-vjepa-blurb">' + recipe.blurb + '</p>' +
      '<div class="mm-vjepa-lines">' + lines + '</div>' +
      '<button type="button" class="secondary mm-vjepa-apply" onclick="applyMmVjepaRecipe(\'' + recipe.id + '\')">Apply recipe</button>' +
      '</article>'
    );
  }).join('');
}

document.addEventListener('DOMContentLoaded', renderMmVjepaRecipeCards);
