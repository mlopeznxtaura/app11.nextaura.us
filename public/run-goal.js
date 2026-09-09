/** Run goal / intention tracker — set before Start, live progress above terminal. */
const RUN_GOAL_DEFAULTS = {
  title: 'App7 — 10M floor smoke → validate stream → scale to 50M',
  detail:
    'Ladder: (1) pick Multimodal or V-JEPA HF stream + 10M footprint smoke ~0.08 GB, ' +
    '(2) confirm terminal shows mm/vjepa hits + loss decreasing, (3) only then scale dim 512 / 3.33 GB. ' +
    'Stop = bytes only (total_train_steps=0). COS app7/stage1/.',
  stopPrimary: 'bytes',
  agent: 'composer-2.5',
};

function loadRunGoalForm() {
  try {
    const raw = localStorage.getItem('runGoalDraft');
    if (!raw) return { ...RUN_GOAL_DEFAULTS };
    return { ...RUN_GOAL_DEFAULTS, ...JSON.parse(raw) };
  } catch (e) {
    return { ...RUN_GOAL_DEFAULTS };
  }
}

function saveRunGoalDraft() {
  const data = {
    title: (document.getElementById('runGoalTitle') || {}).value || '',
    detail: (document.getElementById('runGoalDetail') || {}).value || '',
    stopPrimary: (document.querySelector('input[name="runStopPrimary"]:checked') || {}).value || 'bytes',
    agent: (document.getElementById('runGoalAgent') || {}).value || '',
  };
  try {
    localStorage.setItem('runGoalDraft', JSON.stringify(data));
  } catch (e) {}
  return data;
}

function initRunGoalForm() {
  const d = loadRunGoalForm();
  const title = document.getElementById('runGoalTitle');
  const detail = document.getElementById('runGoalDetail');
  const agent = document.getElementById('runGoalAgent');
  if (title) title.value = d.title;
  if (detail) detail.value = d.detail;
  if (agent) agent.value = d.agent;
  document.querySelectorAll('input[name="runStopPrimary"]').forEach((el) => {
    el.checked = el.value === d.stopPrimary;
  });
  ['runGoalTitle', 'runGoalDetail', 'runGoalAgent'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', saveRunGoalDraft);
  });
  document.querySelectorAll('input[name="runStopPrimary"]').forEach((el) => {
    el.addEventListener('change', () => {
      saveRunGoalDraft();
      syncRunGoalHints();
    });
  });
  syncRunGoalHints();
}

function syncRunGoalHints() {
  const stop = (document.querySelector('input[name="runStopPrimary"]:checked') || {}).value || 'bytes';
  const hint = document.getElementById('runGoalStopHint');
  const gb = Number((document.getElementById('targetGb') || {}).value || 3.33);
  const steps = Number((document.getElementById('totalTrainSteps') || {}).value || 0);
  if (!hint) return;
  if (stop === 'bytes') {
    hint.textContent =
      'Primary stop: ' + gb + ' GB — set total_train_steps to 0 (Advanced) so steps do not cut the run short.';
  } else if (stop === 'steps') {
    hint.textContent = 'Primary stop: ' + (steps || '?') + ' optimizer steps (Chinchilla-style cap).';
  } else {
    hint.textContent = 'Primary stop: Chinchilla token budget for current model size.';
  }
}

function runGoalPayload() {
  const d = saveRunGoalDraft();
  return {
    run_goal_title: d.title,
    run_goal_detail: d.detail,
    run_stop_primary: d.stopPrimary,
    run_goal_agent: d.agent,
  };
}

function renderRunGoalLive(s) {
  const panel = document.getElementById('runGoalLive');
  if (!panel || !s) return;
  const g = s.run_goal || {};
  const title = g.title || (document.getElementById('runGoalTitle') || {}).value || '—';
  const outcome = g.outcome_label || (s.running ? 'Waiting for start…' : 'Set goal before Start');
  const badge = document.getElementById('runGoalOutcomeBadge');
  if (badge) {
    badge.textContent = outcome;
    badge.className = 'run-goal-badge ' + (g.goal_met ? 'ok' : g.outcome === 'partial_step_cap' || g.outcome === 'step_cap_only' ? 'warn' : s.running ? 'run' : 'idle');
  }
  const titleEl = document.getElementById('runGoalLiveTitle');
  if (titleEl) titleEl.textContent = title;
  const detailEl = document.getElementById('runGoalLiveDetail');
  if (detailEl) detailEl.textContent = g.detail || '';
  const agentEl = document.getElementById('runGoalLiveAgent');
  if (agentEl) agentEl.textContent = g.agent ? 'Agent: ' + g.agent : '';
  const byteBar = document.getElementById('runGoalByteBar');
  const byteLbl = document.getElementById('runGoalByteLabel');
  const bytePct = g.byte_progress_pct != null ? g.byte_progress_pct : 0;
  if (byteBar) byteBar.style.width = Math.min(100, bytePct) + '%';
  if (byteLbl) {
    byteLbl.textContent =
      'Bytes: ' +
      (s.size_gb != null ? s.size_gb.toFixed(3) : '0') +
      ' / ' +
      (g.byte_target_gb != null ? g.byte_target_gb : s.target_gb || '?') +
      ' GB (' +
      bytePct.toFixed(1) +
      '%)' +
      (g.byte_met ? ' ✓' : '');
  }
  const stepBar = document.getElementById('runGoalStepBar');
  const stepLbl = document.getElementById('runGoalStepLabel');
  const stepPct = g.step_progress_pct != null ? g.step_progress_pct : 0;
  const stepCap = g.step_cap || s.total_train_steps || 0;
  if (stepBar) stepBar.style.width = stepCap > 0 ? Math.min(100, stepPct) + '%' : '0%';
  if (stepLbl) {
    stepLbl.textContent =
      stepCap > 0
        ? 'Steps: ' + (s.train_step || 0) + ' / ' + stepCap + ' (' + stepPct.toFixed(1) + '%)' + (g.step_met ? ' ✓' : '')
        : 'Steps: ' + (s.train_step || 0) + ' (no step cap — bytes only)';
  }
  const stopEl = document.getElementById('runGoalLiveStop');
  if (stopEl) stopEl.textContent = 'Stop mode: ' + (g.stop_primary || 'bytes');
}

document.addEventListener('DOMContentLoaded', initRunGoalForm);
