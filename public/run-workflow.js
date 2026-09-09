/** Top-to-bottom run workflow — Step 1 = mode+stream+arch, Step 2 = goal. */
(function () {
  const STORAGE_KEY = 'app7RunWorkflowLocks';
  const STEP_IDS = ['config', 'goal'];

  function loadLocks() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { config: false, goal: false };
      const o = JSON.parse(raw);
      if (o.config != null || o.goal != null) {
        return { config: !!o.config, goal: !!o.goal };
      }
      return { config: !!(o.stream && o.footprint), goal: !!o.goal };
    } catch (e) {
      return { config: false, goal: false };
    }
  }

  function saveLocks(locks) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(locks));
    } catch (e) {}
  }

  let locks = loadLocks();

  function el(id) {
    return document.getElementById(id);
  }

  function stepIndex(stepId) {
    return STEP_IDS.indexOf(stepId);
  }

  function highestUnlockedStep() {
    if (!locks.config) return 0;
    if (!locks.goal) return 1;
    return 2;
  }

  function isFinetuneSource(ds) {
    return (
      ds === 'jsonl' ||
      ds === 'local-corpus-mix' ||
      ds === 'sft-intelligence' ||
      ds === 'nemotron-swe' ||
      (typeof isSftFinetune === 'function' && isSftFinetune(ds)) ||
      (typeof isDistillTopic === 'function' && isDistillTopic(ds)) ||
      ds === 'distill-merged' ||
      ds === 'distill-live'
    );
  }

  function streamChecks() {
    const ds = (el('dataSource') || {}).value || '';
    const mode = typeof scaleMode !== 'undefined' ? scaleMode : 'pretrain';
    const issues = [];
    if (!ds) issues.push('Pick a data source.');
    // JSONL / SFT / distill — no HF stream dropdown required
    if (isFinetuneSource(ds)) {
      if (mode !== 'finetune') issues.push('Switch training mode to Fine-tuning for JSONL / SFT / distill.');
      if (ds === 'jsonl' || ds === 'local-corpus-mix') {
        const jf = (el('jsonlFile') || {}).value || '';
        if (!jf) issues.push('Pick a local JSONL corpus file.');
      }
      return { ok: issues.length === 0, issues, summary: ds };
    }
    const hfSel = el('hfDataset');
    const hf = (hfSel || {}).value || '';
    const hfCount = hfSel && hfSel.options ? hfSel.options.length : 0;
    if (ds === 'multimodal') {
      const mm = (typeof multimodalDatasets !== 'undefined' && multimodalDatasets) || window.multimodalDatasets || [];
      if (!mm.length && !hfCount) issues.push('No multimodal HF datasets — fix health or pick a text stream.');
      else if (!hf) issues.push('Select an HF dataset in the Multimodal dropdown.');
    } else if (ds === 'vjepa') {
      const vj = (typeof vjepaDatasets !== 'undefined' && vjepaDatasets) || window.vjepaDatasets || [];
      if (!vj.length && !hfCount) {
        issues.push('No V-JEPA datasets loaded — refresh page, enable scalar 12d, or pick SFT / JSONL for fine-tuning.');
      } else if (!hf) issues.push('Select a V-JEPA dataset in the dropdown.');
    } else if (
      ds === 'mixed' ||
      ds === 'fineweb' ||
      ds === 'fable-traces' ||
      ds === 'openassistant-rm' ||
      ds === 'sol-traces' ||
      ds === 'kimi-k3-traces' ||
      ds === 'unsolved-math' ||
      ds === 'nemotron-math' ||
      ds === 'nemotron-swe'
    ) {
      /* text / trace streams — data source selection is enough */
    } else if (mode === 'finetune' && ds !== 'nemotron-swe') {
      issues.push('Fine-tuning mode needs JSONL, SFT, distill, or nemotron-swe — not an HF byte stream.');
    }
    return { ok: issues.length === 0, issues, summary: ds };
  }

  function footprintChecks() {
    const params = Number((el('modelParams') || {}).value || 0);
    const nEmbd = Number((el('nEmbd') || {}).value || 0);
    const mode = typeof scaleMode !== 'undefined' ? scaleMode : 'pretrain';
    const ds = (el('dataSource') || {}).value || '';
    const issues = [];
    const finetune = mode === 'finetune' || isFinetuneSource(ds);
    if (!finetune) {
      if (params < 8e6) issues.push('Footprint too small — click Apply 10M floor or set dim/layers.');
      if (nEmbd === 256 && params < 9e6) issues.push('Click “Apply 10M floor” or set architecture explicitly.');
      const note = (el('playbookFootprintNote') || {}).textContent || '';
      if (note.indexOf('set with buttons') >= 0 && params < 9e6) {
        issues.push('Footprint not applied — use a ladder button in Step 1.');
      }
    } else if (params < 8e6) {
      issues.push('Load a checkpoint or set architecture — finetune needs a real base model.');
    }
    if (ds === 'vjepa' && !finetune) {
      const sp = el('useScalarProjection');
      if (sp && !sp.checked) issues.push('V-JEPA requires scalar 12d projection ON.');
    }
    return {
      ok: issues.length === 0,
      issues,
      summary: params > 0 ? (params / 1e6).toFixed(1) + 'M' : '—',
    };
  }

  function modeChecks() {
    const mode = typeof scaleMode !== 'undefined' ? scaleMode : 'pretrain';
    const ds = (el('dataSource') || {}).value || '';
    const issues = [];
    const finetuneDs =
      ds === 'jsonl' ||
      ds === 'sft-intelligence' ||
      (typeof isSftFinetune === 'function' && isSftFinetune(ds)) ||
      (typeof isDistillTopic === 'function' && isDistillTopic(ds)) ||
      ds === 'distill-merged' ||
      ds === 'distill-live';
    const pretrainDs =
      ds === 'multimodal' ||
      ds === 'vjepa' ||
      ds === 'mixed' ||
      ds === 'fineweb' ||
      ds === 'fable-traces';
    if (mode === 'finetune' && pretrainDs && !finetuneDs) {
      issues.push('Data source is a pretrain stream — switch mode to Pretrain or pick SFT/distill/JSONL.');
    }
    if (mode === 'pretrain' && finetuneDs && !pretrainDs) {
      issues.push('Data source is finetune — switch mode to Fine-tuning or pick a stream source.');
    }
    return {
      ok: issues.length === 0,
      issues,
      summary: mode === 'finetune' ? 'Fine-tuning' : 'Pretraining',
    };
  }

  function configChecks() {
    const stream = streamChecks();
    const foot = footprintChecks();
    const mode = modeChecks();
    const issues = stream.issues.concat(foot.issues, mode.issues);
    return {
      ok: issues.length === 0,
      issues,
      summary: [mode.summary, stream.summary, foot.summary].filter(Boolean).join(' · '),
    };
  }

  function goalChecks() {
    const title = ((el('runGoalTitle') || {}).value || '').trim();
    const detail = ((el('runGoalDetail') || {}).value || '').trim();
    const issues = [];
    if (title.length < 8) issues.push('Goal title too short — say what this run is for.');
    if (detail.length < 24) issues.push('Intention missing — success criteria + next step.');
    const stop = (document.querySelector('input[name="runStopPrimary"]:checked') || {}).value || 'bytes';
    if (stop === 'bytes') {
      const steps = Number((el('totalTrainSteps') || {}).value || 0);
      if (steps > 0) issues.push('Bytes-only stop: set total_train_steps = 0 in Advanced.');
    }
    return { ok: issues.length === 0, issues, summary: title || '—' };
  }

  function checkStep(stepId) {
    if (stepId === 'config') return configChecks();
    if (stepId === 'goal') return goalChecks();
    return { ok: true, issues: [], summary: '' };
  }

  function setLock(stepId, on) {
    if (STEP_IDS.indexOf(stepId) < 0) return;
    const chk = checkStep(stepId);
    if (on && !chk.ok) {
      alert('Cannot lock ' + stepId + ':\n\n' + chk.issues.join('\n'));
      return false;
    }
    locks[stepId] = !!on;
    if (!on) {
      const idx = stepIndex(stepId);
      STEP_IDS.slice(idx + 1).forEach((s) => {
        locks[s] = false;
      });
    }
    saveLocks(locks);
    renderWorkflow();
    return true;
  }

  function resetWorkflow() {
    locks = { config: false, goal: false };
    window.__streamConfigured = false;
    saveLocks(locks);
    renderWorkflow();
  }

  function renderStepListItem(stepId, n) {
    const li = el('playbookStep' + n);
    if (!li) return;
    const idx = stepIndex(stepId);
    const unlocked = idx <= highestUnlockedStep();
    const locked = locks[stepId];
    const chk = checkStep(stepId);
    li.classList.remove('wf-ok', 'wf-bad', 'wf-active', 'wf-locked', 'wf-pending');
    if (locked) li.classList.add('wf-ok', 'wf-locked');
    else if (!unlocked) li.classList.add('wf-pending');
    else {
      li.classList.add('wf-active');
      if (!chk.ok) li.classList.add('wf-bad');
    }
    const hint = el('playbookStep' + n + 'Hint');
    if (hint) {
      if (locked) {
        hint.textContent = 'Locked · ' + chk.summary;
        hint.className = 'run-playbook-step-hint wf-ok-text';
      } else if (!unlocked) {
        hint.textContent = 'Complete and lock the step above first.';
        hint.className = 'run-playbook-step-hint wf-muted';
      } else if (!chk.ok) {
        hint.textContent = chk.issues[0];
        hint.className = 'run-playbook-step-hint wf-bad-text';
      } else {
        hint.textContent = 'Ready — click Lock step ' + n + ' when correct.';
        hint.className = 'run-playbook-step-hint';
      }
    }
    const btn = el('playbookLock' + n);
    if (btn) {
      btn.disabled = !unlocked || (!locked && !chk.ok);
      btn.textContent = locked ? 'Locked ✓' : 'Lock step ' + n;
      btn.className = locked ? 'ok' : 'secondary';
    }
  }

  function gateSection(sectionId, minStep) {
    const node = el(sectionId);
    if (!node) return;
    const open = highestUnlockedStep() >= minStep;
    node.classList.toggle('workflow-gated', !open);
    node.classList.toggle('workflow-unlocked', open);
    const banner = el(sectionId + 'Gate');
    if (banner) {
      banner.style.display = open ? 'none' : 'block';
      const labels = ['', 'Step 1 (mode + stream + arch)', 'run goal'];
      banner.textContent = 'Locked — complete and lock ' + (labels[minStep] || 'prior steps') + ' in the playbook above.';
    }
  }

  function renderWorkflow() {
    renderStepListItem('config', 1);
    renderStepListItem('goal', 2);
    const step3 = el('playbookStep3');
    if (step3) {
      const ready = highestUnlockedStep() >= 2;
      step3.classList.toggle('wf-ok', ready);
      step3.classList.toggle('wf-pending', !ready);
    }
    gateSection('sectionRunGoal', 1);
    gateSection('chinchillaScalePanel', 1);
    gateSection('sectionRunHistory', 2);
    gateSection('sectionRecipe', 1);
    gateSection('sectionTrainControls', 2);
    const startBtn = el('startBtn');
    if (startBtn) {
      const wfReady = locks.config && locks.goal;
      if (!wfReady) {
        startBtn.disabled = true;
      } else if (!window.__app7StartBlockedByRun) {
        startBtn.disabled = false;
      }
    }
    const status = el('workflowStatus');
    if (status) {
      const parts = [];
      if (locks.config) parts.push('config ✓');
      if (locks.goal) parts.push('goal ✓');
      status.textContent = parts.length ? 'Workflow: ' + parts.join(' · ') : 'Workflow: lock Step 1 config, then Step 2 goal, before Start';
    }
  }

  window.lockWorkflowStep = function (n) {
    const id = STEP_IDS[n - 1];
    if (!id) return;
    const currently = locks[id];
    setLock(id, !currently);
  };

  window.markWorkflowFootprintApplied = function () {
    renderWorkflow();
  };

  window.markWorkflowStreamPicked = function () {
    window.__streamConfigured = true;
    renderWorkflow();
  };

  window.resetRunWorkflow = resetWorkflow;
  window.isRunWorkflowReady = function () {
    return locks.config && locks.goal;
  };
  window.renderRunWorkflow = renderWorkflow;

  document.addEventListener('DOMContentLoaded', () => {
    window.__streamConfigured = false;
    ['dataSource', 'hfDataset', 'nEmbd', 'modelParams', 'runGoalTitle', 'runGoalDetail', 'totalTrainSteps', 'targetGb'].forEach(
      (id) => {
        const node = el(id);
        if (node) node.addEventListener('input', renderWorkflow);
        if (node) node.addEventListener('change', renderWorkflow);
      }
    );
    document.querySelectorAll('input[name="runStopPrimary"]').forEach((r) => {
      r.addEventListener('change', renderWorkflow);
    });
    const pre = el('scaleModePretrain');
    const fin = el('scaleModeFinetune');
    if (pre) pre.addEventListener('click', () => setTimeout(renderWorkflow, 0));
    if (fin) fin.addEventListener('click', () => setTimeout(renderWorkflow, 0));
    renderWorkflow();
  });
})();
