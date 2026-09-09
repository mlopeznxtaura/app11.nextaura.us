(function () {
  function showErr(msg) {
    const el = document.getElementById("inferError");
    if (!el) return;
    if (msg) {
      el.textContent = msg;
      el.style.display = "block";
    } else {
      el.textContent = "";
      el.style.display = "none";
    }
  }

  function renderInferContext(d) {
    const grid = document.getElementById("inferCtxGrid");
    const chips = document.getElementById("inferPromptChips");
    const badge = document.getElementById("trainModeBadge");
    if (!d || !grid) {
      if (grid) grid.innerHTML = '<span class="hint">No training context — start a run in Stage 1.</span>';
      if (chips) chips.innerHTML = "";
      renderInferCheatSheets(null);
      return;
    }
    const mode = d.training_mode || "pretrain";
    if (badge) {
      const smoke = d.is_smoke ? " · smoke" : "";
      badge.textContent = (d.training_mode_label || (mode === "finetune" ? "Fine-tuning" : "Pretraining")) + smoke;
      badge.className = "badge " + mode + (d.is_smoke ? " smoke" : "");
    }
    const archBits = [];
    if (d.block_size) archBits.push("block " + d.block_size);
    if (d.scalar_input_projection) archBits.push("scalar 12d proj");
    const archExtra = archBits.length ? " · " + archBits.join(" · ") : "";
    const rows = [
      ["Mode", d.training_mode_label || "—", mode === "finetune" ? "ok" : ""],
      ["Stream", d.hf_stream_label || d.data_source || "—", ""],
      ["Byte budget", (d.bytes_consumed_mb != null ? d.bytes_consumed_mb + " / " + d.bytes_target_mb + " MB" : "—") + (d.target_gb != null ? " (" + d.target_gb + " GB cap)" : ""), d.is_smoke || (d.chinchilla_ratio != null && d.chinchilla_ratio <= 0.12) ? "warn" : ""],
      ["Chinchilla", d.chinchilla_optimal_gb != null ? d.chinchilla_optimal_gb + " GB optimal · " + (d.chinchilla_train_steps || "?") + " steps" : "—", ""],
      ["Scale", d.byte_scale_note || "—", d.is_smoke || (d.chinchilla_ratio != null && d.chinchilla_ratio <= 0.12) ? "warn" : "ok"],
      ["Stream hits", d.stream_hits || "—", d.stream_hits ? "ok" : ""],
      ["Steps", (d.train_steps || d.step || 0) + (d.total_train_steps_cap ? " / cap " + d.total_train_steps_cap : ""), ""],
      ["Weights", d.weights_source || "—", ""],
      ["Tokenizer", (d.tokenizer_id || "—") + (d.vocab_size ? " · vocab " + d.vocab_size : "") + archExtra, ""],
      ["Checkpoint", (d.training_checkpoint || d.checkpoint_name || "—") + (d.loaded_checkpoint && d.checkpoint_mismatch ? " (header loaded: " + d.loaded_checkpoint + ")" : ""), d.checkpoint_mismatch ? "warn" : ""],
      ["Stop reason", d.stop_reason_label || (d.session_status === "streaming" ? "training…" : d.session_status || "—"), ""],
      ["Eval RM", d.eval_last_mean != null ? "last " + Number(d.eval_last_mean).toFixed(4) + (d.eval_best_mean != null ? " · best " + Number(d.eval_best_mean).toFixed(4) : "") : "—", d.eval_last_mean != null ? "ok" : ""],
      ["Run goal", d.run_goal_title || "—", ""],
    ];
    grid.innerHTML = rows
      .map(function (r) {
        return (
          '<div class="infer-ctx-row"><span class="infer-ctx-k">' +
          r[0] +
          '</span><span class="infer-ctx-v' +
          (r[2] ? " " + r[2] : "") +
          '">' +
          (r[1] || "—") +
          "</span></div>"
        );
      })
      .join("");
    if (chips) {
      let prompts = d.suggested_prompts || [];
      if (!prompts.length && (d.data_source === "nemotron-swe" || (d.hf_stream_label || "").indexOf("nemotron") >= 0)) {
        prompts = [
          "CI fails: test_delete_stream — AttributeError stream not found in aws.rb.storage. List 3 files to inspect, grep patterns, and one fix hypothesis.",
          "Given this repo issue, list the files you would inspect first and why.",
          "Write a minimal patch plan for a failing CI test without changing unrelated code.",
        ];
      }
      chips.innerHTML = prompts
        .map(function (p) {
          const label = p.slice(0, 48) + (p.length > 48 ? "…" : "");
          return (
            '<button type="button" class="secondary infer-prompt-chip" data-prompt="' +
            escapeHtmlAttr(p) +
            '">' +
            escapeHtml(label) +
            "</button>"
          );
        })
        .join("");
    }
    renderInferCheatSheets(d);
  }

  function escapeHtmlAttr(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;");
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  }

  function fmtParamsM(d) {
    if (!d) return null;
    if (d.model_params_m != null) return Number(d.model_params_m);
    const p = d.params || d.model_params;
    if (p) return Math.round((p / 1e6) * 10) / 10;
    return null;
  }

  function fmtParamsLabel(d) {
    const m = fmtParamsM(d);
    return m != null ? "~" + m + "M" : "unknown size";
  }

  function inferArchLabel(d) {
    if (!d) return "—";
    if (d.n_layer && d.n_embd) return d.n_layer + "L×" + d.n_embd + "d";
    return "—";
  }

  function inferStreamLabel(d) {
    if (!d) return "—";
    return d.hf_stream_label || d.data_source || d.dataset || "current stream";
  }

  function inferTaskProfile(d) {
    if (!d) return "general";
    const ds = String(d.data_source || "").toLowerCase();
    const goal = String(d.run_goal_title || "").toLowerCase();
    const stream = String(d.hf_stream_label || "").toLowerCase();
    if (ds === "nemotron-swe" || goal.indexOf("swe") >= 0 || stream.indexOf("swe") >= 0) return "swe";
    if (ds === "multimodal" || ds === "vjepa") return "vision";
    if (ds.indexOf("distill") >= 0) return "distill";
    if (d.training_mode === "pretrain" || ds === "fineweb" || ds === "nemotron") return "pretrain";
    if (ds.indexOf("trace") >= 0 || ds === "mixed" || ds.indexOf("sol") >= 0 || ds.indexOf("kimi") >= 0)
      return "reasoning";
    if (d.training_mode === "finetune") return "finetune";
    return "general";
  }

  function trainingSignalLabel(d) {
    const profile = inferTaskProfile(d);
    const stream = inferStreamLabel(d);
    if (profile === "swe") return "cross-entropy on <strong>" + escapeHtml(stream) + "</strong> agent/coding trajectories";
    if (profile === "pretrain") return "cross-entropy on next-token prediction over <strong>" + escapeHtml(stream) + "</strong>";
    if (profile === "vision") return "cross-entropy on <strong>" + escapeHtml(stream) + "</strong> multimodal / video latents";
    if (profile === "distill") return "cross-entropy matching teacher outputs on <strong>" + escapeHtml(stream) + "</strong>";
    if (profile === "reasoning") return "cross-entropy on reasoning traces from <strong>" + escapeHtml(stream) + "</strong>";
    return "cross-entropy on <strong>" + escapeHtml(stream) + "</strong> training rows";
  }

  function promptStyleHint(d) {
    const profile = inferTaskProfile(d);
    if (profile === "swe") return "Use <strong>repo / CI / patch-plan prompts</strong> aligned with the SWE stream — not unrelated domains.";
    if (profile === "vision") return "Use <strong>scene / motion / visual reasoning prompts</strong> aligned with the multimodal stream.";
    if (profile === "pretrain") return "Use <strong>completion or short Q&amp;A</strong> — the model has not been instruction-tuned yet.";
    if (profile === "distill") return "Use prompts from the <strong>distill topic</strong> you trained on (code, math, physics, etc.).";
    if (profile === "reasoning") return "Use <strong>step-by-step reasoning or code-debug prompts</strong> matching the trace mix.";
    return "Use prompts that match the <strong>current data stream</strong> shown in the context panel above.";
  }

  function tempBandForProfile(profile) {
    if (profile === "swe") {
      return {
        rec: "0.25–0.45",
        note: "recommended for live SWE / agent SFT smoke tests",
        troubleshoot: "0.2–0.35",
      };
    }
    if (profile === "pretrain") {
      return { rec: "0.7–0.9", note: "pretrain checkpoints are not instruction-tuned — higher temp explores the distribution", troubleshoot: "0.5–0.7" };
    }
    if (profile === "vision") {
      return { rec: "0.4–0.6", note: "balanced for captioning / VQA-style answers", troubleshoot: "0.3–0.45" };
    }
    if (profile === "distill" || profile === "reasoning") {
      return { rec: "0.3–0.5", note: "good for structured reasoning without wild drift", troubleshoot: "0.2–0.35" };
    }
    return { rec: "0.35–0.55", note: "reasonable default for general finetune smoke tests", troubleshoot: "0.25–0.4" };
  }

  function maxTokenBandForProfile(profile, block) {
    const cap = Math.max(16, block - 8);
    const mid = Math.min(cap, Math.round(block * 0.65));
    if (profile === "swe") return { lo: Math.min(120, mid), hi: Math.min(220, cap), long: cap };
    if (profile === "pretrain") return { lo: 40, hi: Math.min(120, cap), long: Math.min(180, cap) };
    if (profile === "vision") return { lo: 60, hi: Math.min(160, cap), long: cap };
    return { lo: 80, hi: mid, long: cap };
  }

  function normalizeInferContext(raw) {
    if (!raw) return null;
    const step = raw.train_step || raw.step || 0;
    const params = raw.model_params || raw.params;
    return {
      step: step,
      loss: raw.last_loss != null ? raw.last_loss : raw.loss,
      params: params,
      model_params_m: params ? Math.round((params / 1e6) * 10) / 10 : raw.model_params_m,
      n_layer: raw.n_layer,
      n_embd: raw.n_embd,
      n_head: raw.n_head,
      data_source: raw.data_source,
      dataset: raw.dataset,
      hf_dataset_id: raw.hf_dataset_id,
      checkpoint_name: raw.checkpoint_name || raw.training_checkpoint,
      training: raw.running != null ? raw.running : raw.training,
      model_label: raw.model_label,
      training_mode: raw.training_mode,
      training_mode_label: raw.training_mode_label,
      is_smoke: raw.is_smoke,
      target_gb: raw.target_gb,
      bytes_consumed_mb: raw.bytes_consumed_mb != null ? raw.bytes_consumed_mb : raw.size_mb,
      bytes_target_mb: raw.bytes_target_mb,
      chinchilla_optimal_gb: raw.chinchilla_optimal_gb,
      chinchilla_ratio: raw.chinchilla_ratio,
      byte_scale_note: raw.byte_scale_note,
      stop_reason: raw.stop_reason,
      stop_reason_label: raw.stop_reason_label,
      session_status: raw.status || raw.session_status,
      tokenizer_id: raw.tokenizer_id,
      vocab_size: raw.vocab_size,
      block_size: raw.block_size,
      scalar_input_projection: raw.scalar_input_projection,
      weights_source: raw.weights_source,
      loaded_checkpoint: raw.loaded_checkpoint,
      training_checkpoint: raw.training_checkpoint,
      checkpoint_mismatch: raw.checkpoint_mismatch,
      stream_hits: raw.stream_hits,
      hf_stream_label: raw.hf_stream_label,
      eval_last_mean: raw.eval_last_mean != null ? raw.eval_last_mean : raw.eval && raw.eval.last_mean,
      eval_best_mean: raw.eval_best_mean != null ? raw.eval_best_mean : raw.eval && raw.eval.best_mean,
      train_steps: raw.train_step || step,
      chinchilla_train_steps: raw.chinchilla_train_steps,
      total_train_steps_cap: raw.total_train_steps_cap,
      run_goal_title: raw.run_goal_title || (raw.run_goal && raw.run_goal.title),
      suggested_prompts: raw.suggested_prompts,
      device: raw.device,
    };
  }

  function setCheatHtml(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  }

  function renderInferCheatSheets(d) {
    const sectionHint = document.getElementById("inferSectionHint");
    if (!d || (!d.step && !d.checkpoint_name && !fmtParamsM(d))) {
      const idle =
        "Load or train a checkpoint — sampling cheat sheets below will auto-fill from the <strong>loaded model</strong> (params, block size, stream, mode).";
      if (sectionHint) sectionHint.innerHTML = idle;
      return;
    }

    const paramsLabel = fmtParamsLabel(d);
    const paramsM = fmtParamsM(d);
    const arch = inferArchLabel(d);
    const block = Number(d.block_size) || 256;
    const cap = Math.max(16, block - 8);
    const examplePromptTok = Math.min(48, Math.round(block * 0.18));
    const exampleMax = Math.max(16, cap - examplePromptTok);
    const stream = inferStreamLabel(d);
    const profile = inferTaskProfile(d);
    const tokBand = maxTokenBandForProfile(profile, block);
    const tempBand = tempBandForProfile(profile);
    const ckpt = d.checkpoint_name || d.training_checkpoint || ("step " + (d.step || "?"));
    const modeLabel = d.training_mode_label || d.training_mode || "training";
    const tokenizer = d.tokenizer_id ? d.tokenizer_id + (d.vocab_size ? " · vocab " + d.vocab_size : "") : "current tokenizer";
    const weights = d.weights_source || (d.training ? "live GPU weights" : "loaded checkpoint");

    if (sectionHint) {
      sectionHint.innerHTML =
        "Test <strong>" +
        escapeHtml(weights) +
        "</strong> — <strong>" +
        paramsLabel +
        "</strong> · <strong>" +
        escapeHtml(arch) +
        "</strong> · block <strong>" +
        block +
        "</strong> · <em>" +
        escapeHtml(stream) +
        "</em>. Sampling knobs affect generation only; they do <strong>not</strong> change training loss. " +
        promptStyleHint(d);
    }

    setCheatHtml(
      "inferCheatMaxTokens",
      "<strong>Max tokens</strong> = how many <em>new</em> tokens the model may emit after your prompt (decode budget — not model size).<br><br>" +
        "<strong>Loaded model:</strong> <strong>" +
        paramsLabel +
        " parameters</strong> (" +
        escapeHtml(arch) +
        ") · checkpoint <code>" +
        escapeHtml(ckpt) +
        "</code> · " +
        escapeHtml(tokenizer) +
        ".<br><br>" +
        "<strong>Block size " +
        block +
        "</strong> is the hard context window: prompt + generated tokens must fit in one forward pass. " +
        "Cap for this model: <strong>≤" +
        cap +
        "</strong> new tokens (block − 8 safety margin).<br><br>" +
        "<strong>Rule:</strong> max_tokens ≤ " +
        block +
        " − prompt_tokens − 8. Example: ~" +
        examplePromptTok +
        "-token prompt → stay under <strong>~" +
        exampleMax +
        "</strong>.<br><br>" +
        "<strong>For " +
        escapeHtml(stream) +
        ":</strong> try <strong>" +
        tokBand.lo +
        "–" +
        tokBand.hi +
        "</strong> for focused answers; up to <strong>" +
        tokBand.long +
        "</strong> only if you accept truncation at block " +
        block +
        "."
    );

    setCheatHtml(
      "inferCheatTemp",
      "<strong>Temperature</strong> scales logits before softmax at <em>each</em> generated token on <strong>" +
        paramsLabel +
        " · " +
        escapeHtml(arch) +
        "</strong> (block " +
        block +
        "). Lower = sharper; higher = more random.<br><br>" +
        "<strong>0.05–0.2</strong> — near-greedy. Exact recall, short completions, format reproduction.<br>" +
        "<strong>" +
        tempBand.rec +
        "</strong> — <em>" +
        tempBand.note +
        "</em> for <strong>" +
        escapeHtml(stream) +
        "</strong>.<br>" +
        "<strong>0.6–0.9</strong> — exploratory; often too noisy for code/agent tasks.<br>" +
        "<strong>1.0+</strong> — rarely useful except pretrain probing.<br><br>" +
        "<strong>Does not train the model.</strong> Training (" +
        escapeHtml(modeLabel) +
        ") uses teacher-forced cross-entropy on <em>" +
        escapeHtml(stream) +
        "</em>, not sampling.<br><br>" +
        "<strong>If output loops or echoes the prompt:</strong> drop to <strong>" +
        tempBand.troubleshoot +
        "</strong>, enable repetition penalty, use a stream-aligned prompt (chips above)."
    );

    setCheatHtml(
      "inferCheatRepPenalty",
      "<strong>Repetition penalty</strong> down-weights tokens already in prompt + answer (1.0 = off). Applies only at <strong>Ask</strong> time on the loaded <strong>" +
        paramsLabel +
        "</strong> weights — never during gradient steps.<br><br>" +
        "<strong>1.0</strong> — off (verbatim completion tests only).<br>" +
        "<strong>1.05–1.15</strong> — light anti-loop; good while <strong>" +
        escapeHtml(modeLabel) +
        "</strong> is still running on <em>" +
        escapeHtml(stream) +
        "</em>.<br>" +
        "<strong>1.2–1.35</strong> — stronger when output stutters or copies your question.<br>" +
        "<strong>1.4+</strong> — aggressive; may suppress valid repeated identifiers in code.<br><br>" +
        "<strong>Does not reinforce training.</strong> Improvement comes from " +
        trainingSignalLabel(d) +
        " (watch terminal loss). Rep penalty only makes smoke tests less annoying.<br><br>" +
        "<strong>Pair with temperature:</strong> high temp + low rep → loops on block-" +
        block +
        " models; low temp + rep ~1.15 → stable answers for <em>" +
        escapeHtml(stream) +
        "</em>."
    );

    const rmSummary = document.getElementById("inferRmSummary");
    if (rmSummary) {
      rmSummary.innerHTML =
        "Training improves via " +
        trainingSignalLabel(d) +
        " (terminal loss). RM is a separate preference judge — run after Ask or on checkpoint saves.";
    }

    const lossHint =
      d.training_mode === "pretrain"
        ? "Pretrain loss in the 2–4 range is normal early; judge with completion quality + RM trend."
        : profile === "swe"
          ? "Mid-SFT loss ~1.3–1.7 is common; judge with SWE-style prompts + RM across checkpoints."
          : "Watch loss ↓ in the terminal for the active stream; use RM to compare checkpoints.";

    setCheatHtml(
      "inferCheatRm",
      "<strong>Two loops — don't confuse them.</strong> Loaded: <strong>" +
        paramsLabel +
        "</strong> · <strong>" +
        escapeHtml(arch) +
        "</strong> · <code>" +
        escapeHtml(ckpt) +
        "</code> · step <strong>" +
        (d.train_steps || d.step || "—") +
        "</strong>" +
        (d.loss != null ? " · loss <strong>" + Number(d.loss).toFixed(4) + "</strong>" : "") +
        ".<br><br>" +
        "<strong>1. Training signal (what updates weights)</strong><br>Each step: sample from <em>" +
        escapeHtml(stream) +
        "</em> → tokenize (block " +
        block +
        ") → forward → " +
        trainingSignalLabel(d) +
        " → optimizer step. <strong>Terminal loss ↓</strong> = the model is fitting this stream.<br><br>" +
        "<strong>2. Reward model (RM) — eval only</strong><br>OpenAssistant DeBERTa scores (prompt, answer) for preference-like quality. It does <strong>not</strong> backprop into your <strong>" +
        paramsLabel +
        "</strong> model during this run.<br><br>" +
        "<strong>Score last answer (RM)</strong> — quick smoke on your last Ask pair.<br>" +
        "<strong>RM panel (20 prompts)</strong> — benchmark mean RM; compare checkpoints in the eval leaderboard. Auto-runs on save when enabled.<br><br>" +
        "<strong>How to read progress:</strong> " +
        lossHint +
        "<br><br>RL / DPO is <strong>not</strong> wired here — RM is the keep/discard and compare-checkpoint signal."
    );
  }

  function bindInferPromptChips() {
    const chips = document.getElementById("inferPromptChips");
    if (!chips || chips._inferChipBound) return;
    chips._inferChipBound = true;
    chips.addEventListener("click", function (ev) {
      const btn = ev.target.closest(".infer-prompt-chip");
      if (!btn) return;
      const p = btn.getAttribute("data-prompt");
      if (!p) return;
      const ta = document.getElementById("inferQ");
      if (ta) {
        ta.value = p;
        ta.focus();
      }
    });
  }

  window.updateTrainModeBadge = function (d) {
    if (d) renderInferContext(d);
  };

  function formatInferLabel(d) {
    if (!d) return "current checkpoint";
    if (d.model_label) return d.model_label;
    const parts = [];
    if (d.model_params_m != null) parts.push(d.model_params_m + "M");
    else if (d.params) parts.push((d.params / 1e6).toFixed(1) + "M");
    if (d.n_layer && d.n_embd) parts.push(d.n_layer + "L×" + d.n_embd + "d");
    if (d.data_source) parts.push(d.data_source);
    if (d.checkpoint_name) parts.push(d.checkpoint_name);
    else if (d.step) parts.push("step " + d.step);
    return parts.join(" · ") || d.model || "current checkpoint";
  }

  function updateInferRmLive(d, lastScore) {
    const el = document.getElementById("inferRmLive");
    if (!el) return;
    const parts = [];
    if (lastScore != null && !isNaN(lastScore)) {
      parts.push("Last Ask RM: <strong>" + Number(lastScore).toFixed(4) + "</strong>");
    }
    if (d && d.eval_last_mean != null) {
      parts.push("Panel last: " + Number(d.eval_last_mean).toFixed(4));
    }
    if (d && d.eval_best_mean != null) {
      parts.push("best " + Number(d.eval_best_mean).toFixed(4));
    }
    if (d && d.train_steps) {
      parts.push("step " + d.train_steps);
    }
    el.innerHTML = parts.length ? parts.join(" · ") : "RM: — (Ask a coding prompt, then Score last answer)";
  }

  function applySamplingDefaults(d) {
    const tokInp = document.getElementById("inferTokens");
    const hint = document.getElementById("inferMaxTokensHint");
    const block = d && d.block_size ? Number(d.block_size) : 256;
    const suggestedCap = Math.min(512, Math.max(64, block - 32));
    if (tokInp) {
      tokInp.max = String(Math.min(512, block - 8));
      if (!tokInp.dataset.userTouched) {
        tokInp.value = String(Math.min(180, suggestedCap));
      }
    }
    if (hint) {
      const pLabel = fmtParamsLabel(d);
      hint.textContent = "≤" + (block - 8) + " (block " + block + ")";
      hint.title =
        pLabel +
        " params are weights; max tokens is decode budget. Prompt + answer must fit block_size " +
        block +
        ".";
    }
    const profile = inferTaskProfile(d);
    const tempInp = document.getElementById("inferTemp");
    const repInp = document.getElementById("inferRepPenalty");
    const tempDefaults = { swe: "0.35", pretrain: "0.8", vision: "0.5", distill: "0.4", reasoning: "0.4", finetune: "0.45", general: "0.5" };
    const repDefaults = { swe: "1.15", pretrain: "1.1", vision: "1.1", distill: "1.12", reasoning: "1.12", finetune: "1.1", general: "1.1" };
    if (tempInp && !tempInp.dataset.userTouched) tempInp.value = tempDefaults[profile] || "0.5";
    if (repInp && !repInp.dataset.userTouched) repInp.value = repDefaults[profile] || "1.1";
  }

  function bindSamplingUserTouch() {
    ["inferTokens", "inferTemp", "inferRepPenalty"].forEach(function (id) {
      const el = document.getElementById(id);
      if (!el || el.dataset.touchBound) return;
      el.dataset.touchBound = "1";
      el.addEventListener("change", function () {
        el.dataset.userTouched = "1";
      });
    });
  }

  function applyInferLabels(d) {
    const label = formatInferLabel(d);
    const title = document.getElementById("inferStageTitle");
    const ta = document.getElementById("inferQ");
    const paramsM =
      d && d.model_params_m != null
        ? String(d.model_params_m)
        : d && d.params
          ? (d.params / 1e6).toFixed(1)
          : "?";
    if (title) {
      title.textContent = d && d.training ? "Stage 2 — inference (live weights)" : "Stage 2 — inference";
    }
    if (ta) {
      const profile = inferTaskProfile(d);
      const placeholders = {
        swe: "CI / repo issue prompt — failing test, files to inspect, grep patterns, patch plan.",
        vision: "Describe the scene or motion in this clip; step-by-step visual reasoning.",
        pretrain: "Short completion or factual Q&A — model is not instruction-tuned yet.",
        distill: "Prompt matching your distill topic (code, math, physics, …).",
        reasoning: "Step-by-step reasoning or debug prompt matching the trace mix.",
      };
      ta.placeholder =
        placeholders[profile] ||
        "Prompt to test " + paramsM + "M · " + inferArchLabel(d) + (d && d.checkpoint_name ? " · " + d.checkpoint_name : "") + "…";
    }
    window._inferLabel = label;
    return label;
  }

  async function inferApi(path, opts) {
    opts = opts || {};
    const r = await fetch(path, opts);
    let d = {};
    try {
      d = await r.json();
    } catch (_) {
      d = { error: r.statusText || "request failed" };
    }
    if (!r.ok) throw new Error(d.error || r.statusText || "HTTP " + r.status);
    return d;
  }

  function appendChat(role, text, meta) {
    const box = document.getElementById("inferChat");
    if (!box) return;
    const el = document.createElement("div");
    el.className = role === "q" ? "iq" : "ia";
    el.textContent = (role === "q" ? "Q: " : "A: ") + text;
    box.appendChild(el);
    if (meta) {
      const m = document.createElement("div");
      m.className = "imeta";
      m.textContent = meta;
      box.appendChild(m);
    }
    box.scrollTop = box.scrollHeight;
  }

  window.updateInferFromStatus = function (s) {
    if (!s) return;
    const el = document.getElementById("inferReady");
    const askBtn = document.getElementById("inferAskBtn");
    if (askBtn) {
      askBtn.disabled = s.inference_available === false;
      askBtn.title =
        s.inference_available === false
          ? "Muse-Glimmer is training; generation unlocks from its saved QLoRA adapter."
          : "";
    }
    const step = s.train_step || s.step || 0;
    if (!step && !s.checkpoint_name) {
      if (el) {
        el.textContent = "No checkpoint yet — train a few steps in Stage 1.";
        el.style.color = "#f87171";
      }
      applyInferLabels(null);
      renderInferContext(null);
      renderInferCheatSheets(null);
      return;
    }
    const d = normalizeInferContext({
      step: step,
      loss: s.last_loss,
      params: s.model_params,
      model_params_m: s.model_params ? Math.round((s.model_params / 1e6) * 10) / 10 : null,
      n_layer: s.n_layer,
      n_embd: s.n_embd,
      data_source: s.data_source,
      hf_dataset_id: s.hf_dataset_id,
      checkpoint_name: s.checkpoint_name,
      training: s.running,
      model_label: s.model_label || null,
      training_mode: s.training_mode,
      training_mode_label: s.training_mode_label,
      is_smoke: s.is_smoke,
      target_gb: s.target_gb,
      bytes_consumed_mb: s.bytes_consumed_mb != null ? s.bytes_consumed_mb : s.size_mb,
      bytes_target_mb: s.bytes_target_mb,
      chinchilla_optimal_gb: s.chinchilla_optimal_gb,
      chinchilla_ratio: s.chinchilla_ratio,
      byte_scale_note: s.byte_scale_note,
      stop_reason: s.stop_reason,
      stop_reason_label: s.stop_reason_label,
      session_status: s.status,
      tokenizer_id: s.tokenizer_id,
      vocab_size: s.vocab_size,
      block_size: s.block_size,
      scalar_input_projection: s.scalar_input_projection,
      weights_source: s.weights_source,
      loaded_checkpoint: s.loaded_checkpoint,
      training_checkpoint: s.training_checkpoint,
      checkpoint_mismatch: s.checkpoint_mismatch,
      stream_hits: s.stream_hits,
      hf_stream_label: s.hf_stream_label,
      eval_last_mean: s.eval_last_mean != null ? s.eval_last_mean : s.eval && s.eval.last_mean,
      eval_best_mean: s.eval_best_mean != null ? s.eval_best_mean : s.eval && s.eval.best_mean,
      train_steps: s.train_step || step,
      chinchilla_train_steps: s.chinchilla_train_steps,
      total_train_steps_cap: s.total_train_steps_cap,
      run_goal_title: s.run_goal_title || (s.run_goal && s.run_goal.title),
      suggested_prompts: s.suggested_prompts,
    });
    window._inferStatusCache = d;
    if (!d.model_label && d.data_source === "multimodal" && d.hf_dataset_id) {
      d.model_label =
        d.model_params_m + "M · " + d.n_layer + "L×" + d.n_embd + "d · multimodal:" + d.hf_dataset_id.split("/").pop();
    } else if (!d.model_label && d.data_source === "vjepa" && d.hf_dataset_id) {
      d.model_label =
        d.model_params_m + "M · " + d.n_layer + "L×" + d.n_embd + "d · vjepa:" + d.hf_dataset_id.split("/").pop();
    }
    renderInferContext(d);
    applySamplingDefaults(d);
    updateInferRmLive(d, window._lastRmScore);
    const label = applyInferLabels(d);
    if (el) {
      const loss = s.last_loss != null ? Number(s.last_loss).toFixed(4) : "—";
      const prefix = s.running ? "Live training weights · " : "Ready · ";
      el.textContent =
        prefix +
        label +
        " · step " +
        step +
        " · loss " +
        loss +
        (s.inference_available === false ? " · generation locked during training" : "");
      el.style.color = s.running ? "#93c5fd" : "#86efac";
    }
  };

  window.refreshInferStatus = async function () {
    const el = document.getElementById("inferReady");
    try {
      const raw = await inferApi("/api/infer/status");
      const askBtn = document.getElementById("inferAskBtn");
      if (askBtn) {
        askBtn.disabled = raw.inference_available === false;
        askBtn.title =
          raw.inference_available === false
            ? raw.error || "Live model is training; generation is temporarily locked."
            : "";
      }
      if (!raw.ready) {
        if (el) {
          el.textContent = raw.error || "No checkpoint yet — train a few steps in Stage 1.";
          el.style.color = "#f87171";
        }
        applyInferLabels(null);
        renderInferContext(null);
        renderInferCheatSheets(null);
        return;
      }
      const d = normalizeInferContext(raw);
      if (!d || (!d.step && !d.checkpoint_name)) {
        if (el) {
          el.textContent = "No checkpoint yet — train a few steps in Stage 1.";
          el.style.color = "#f87171";
        }
        applyInferLabels(null);
        renderInferContext(null);
        renderInferCheatSheets(null);
        return;
      }
      d.training = d.training != null ? d.training : !!raw.training;
      const label = applyInferLabels(d);
      renderInferContext(d);
      applySamplingDefaults(d);
      updateInferRmLive(d, window._lastRmScore);
      if (el) {
        const prefix = d.training ? "Live training weights · " : "Ready · ";
        el.textContent =
          prefix +
          label +
          " · step " +
          (d.step || 0) +
          " · loss " +
          (d.loss ? Number(d.loss).toFixed(4) : "—") +
          " · " +
          (d.device || "cpu") +
          (raw.inference_available === false ? " · generation locked during training" : "");
        el.style.color = d.training ? "#93c5fd" : "#86efac";
      }
    } catch (_) {
      if (el) {
        el.textContent = "Inference API not reachable.";
        el.style.color = "#94a3b8";
      }
    }
  };

  let _inferModelsCache = [];

  function inferModelOptionLabel(m) {
    if (!m) return "";
    if (m.label) return m.label;
    const src = m.source === "upcloud" ? "UpCloud" : "local";
    return m.name + " (" + src + (m.size_mb != null ? " · " + m.size_mb + " MB" : "") + ")";
  }

  window.refreshInferModels = async function () {
    const sel = document.getElementById("inferModelSelect");
    const syncBtn = document.getElementById("inferSyncUpcloudBtn");
    if (!sel) return;
    try {
      const d = await inferApi("/api/infer/models");
      _inferModelsCache = d.models || [];
      const loaded = d.loaded || "";
      const prev = sel.value;
      sel.innerHTML = "";
      if (!_inferModelsCache.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No checkpoints yet";
        sel.appendChild(opt);
      } else {
        _inferModelsCache.forEach(function (m) {
          const opt = document.createElement("option");
          opt.value = m.id || m.name;
          opt.textContent = inferModelOptionLabel(m) + (m.loaded ? " ✓ loaded" : "");
          opt.dataset.name = m.name;
          opt.dataset.source = m.source || "local";
          sel.appendChild(opt);
        });
      }
      const match = _inferModelsCache.find(function (m) {
        return m.loaded || m.name === loaded;
      });
      if (match) sel.value = match.id || match.name;
      else if (prev) sel.value = prev;
      const hdr = document.getElementById("inferHeaderStatus");
      if (hdr) {
        const loadedModel = _inferModelsCache.find(function (m) {
          return m.loaded;
        });
        hdr.textContent = loadedModel
          ? "loaded: " + loadedModel.name
          : d.loaded
            ? "loaded: " + d.loaded
            : "no checkpoint";
        hdr.style.color = loadedModel || d.loaded ? "#86efac" : "#94a3b8";
      }
      if (syncBtn) {
        const st = d.storage || {};
        syncBtn.disabled = !st.upcloud_configured;
        syncBtn.title = st.upcloud_configured
          ? "Archive all local .pt files to UpCloud object storage"
          : "UpCloud storage not configured on server";
      }
    } catch (e) {
      sel.innerHTML = '<option value="">— models unavailable —</option>';
      if (syncBtn) syncBtn.disabled = true;
    }
  };

  window.switchInferModel = async function () {
    showErr("");
    const sel = document.getElementById("inferModelSelect");
    if (!sel || !sel.value) return;
    const opt = sel.options[sel.selectedIndex];
    const name = (opt && opt.dataset.name) || sel.value.split(":").pop();
    const source = (opt && opt.dataset.source) || (sel.value.indexOf("upcloud:") === 0 ? "upcloud" : "local");
    const loadedOpt = Array.prototype.find.call(sel.options, function (o) {
      return o.textContent.indexOf("✓ loaded") >= 0;
    });
    if (loadedOpt && loadedOpt.value === sel.value) return;
    const prev = sel.value;
    sel.disabled = true;
    const ready = document.getElementById("inferReady");
    if (ready) {
      ready.textContent = "Loading " + name + (source === "upcloud" ? " from UpCloud…" : "…");
      ready.style.color = "#93c5fd";
    }
    try {
      await inferApi("/api/infer/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, source: source }),
      });
      await window.refreshInferModels();
      await window.refreshInferStatus();
      if (typeof line === "function") line("[GREEN] Inference model → " + name, "green");
    } catch (e) {
      showErr(e.message || String(e));
      sel.value = prev;
      if (typeof line === "function") line("[RED] Load failed: " + (e.message || e), "red");
    } finally {
      sel.disabled = false;
    }
  };

  window.syncUpcloudCheckpoints = async function () {
    showErr("");
    const btn = document.getElementById("inferSyncUpcloudBtn");
    const prev = btn ? btn.textContent : "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Syncing…";
    }
    try {
      const d = await inferApi("/api/checkpoints/sync-upcloud", { method: "POST" });
      const n = (d.uploaded || []).length;
      const msg = "UpCloud sync: " + n + " uploaded" + (d.errors && d.errors.length ? " · " + d.errors.length + " errors" : "");
      if (typeof line === "function") line("[CYAN] " + msg, "cyan");
      await window.refreshInferModels();
    } catch (e) {
      showErr(e.message || String(e));
    } finally {
      if (btn) {
        btn.textContent = prev || "Sync → UpCloud";
        btn.disabled = false;
      }
    }
  };

  window.toggleRepetitionPenalty = function () {
    const on = !!(document.getElementById("useRepetitionPenalty") || {}).checked;
    const inp = document.getElementById("inferRepPenalty");
    if (inp) inp.disabled = !on;
    const cell = document.getElementById("repPenaltyCell");
    if (cell) cell.classList.toggle("field-off", !on);
  };

  function appendScoreMeta(text) {
    const box = document.getElementById("inferChat");
    if (!box) return;
    const m = document.createElement("div");
    m.className = "imeta";
    m.textContent = text;
    box.appendChild(m);
    box.scrollTop = box.scrollHeight;
  }

  window.scoreLastAnswer = async function () {
    showErr("");
    if (!window.lastInferQ || !window.lastInferA) {
      showErr("Ask a question first.");
      return;
    }
    const btn = document.getElementById("inferScoreBtn");
    const prevLabel = btn ? btn.textContent : "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Scoring…";
    }
    try {
      const d = await inferApi("/api/eval/reward", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: window.lastInferQ, answer: window.lastInferA }),
      });
      const ck = window._inferLabel || "current checkpoint";
      const label = "RM " + d.score.toFixed(4) + " · " + ck;
      window._lastRmScore = d.score;
      updateInferRmLive(window._inferStatusCache || null, d.score);
      appendScoreMeta(label);
      if (typeof line === "function") line("[CYAN] " + label, "cyan");
    } catch (e) {
      const msg = "RM: " + (e.message || String(e));
      showErr(msg);
      if (typeof line === "function") line("[RED] " + msg, "red");
    } finally {
      if (btn) {
        btn.textContent = prevLabel || "Score with RM";
        btn.disabled = !window.lastInferA;
      }
    }
  };

  window.runInferRmPanel = async function () {
    showErr("");
    const btn = document.getElementById("inferRmPanelBtn");
    const prev = btn ? btn.textContent : "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "RM panel running…";
    }
    try {
      const d = await inferApi("/api/eval/reward-panel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 20, live: true, record: true }),
      });
      const msg =
        "RM panel: mean " +
        Number(d.mean_reward).toFixed(4) +
        (d.delta != null ? " · Δ " + Number(d.delta).toFixed(4) : "") +
        " · " +
        (d.checkpoint || "live") +
        " · " +
        (d.n_scored || d.results && d.results.length || "?") +
        " prompts";
      updateInferRmLive(
        {
          eval_last_mean: d.mean_reward,
          eval_best_mean: d.baseline_mean,
          train_steps: d.step,
        },
        window._lastRmScore
      );
      appendScoreMeta(msg);
      if (typeof line === "function") line("[GREEN] " + msg, "green");
      if (typeof window.refreshEvalPanel === "function") window.refreshEvalPanel();
    } catch (e) {
      const msg = "RM panel: " + (e.message || String(e));
      showErr(msg);
      if (typeof line === "function") line("[RED] " + msg, "red");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prev || "RM panel (20 prompts)";
      }
    }
  };

  window.inferAsk = async function () {
    showErr("");
    const q = ((document.getElementById("inferQ") || {}).value || "").trim();
    if (!q) {
      showErr("Type a prompt first.");
      return;
    }
    const tokens = parseInt((document.getElementById("inferTokens") || {}).value, 10) || 120;
    const temp = parseFloat((document.getElementById("inferTemp") || {}).value);
    const useRep = !!(document.getElementById("useRepetitionPenalty") || {}).checked;
    const repPenalty = parseFloat((document.getElementById("inferRepPenalty") || {}).value);
    const btn = document.getElementById("inferAskBtn");
    if (btn) btn.disabled = true;
    appendChat("q", q);
    try {
      const d = await inferApi("/api/infer/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          max_new_tokens: tokens,
          temperature: isNaN(temp) ? 0.35 : temp,
          use_repetition_penalty: useRep,
          repetition_penalty: useRep && !isNaN(repPenalty) ? repPenalty : 1.0,
        }),
      });
      const metaLabel = formatInferLabel(d);
      appendChat("a", d.answer || "(empty)", metaLabel + " · " + (d.latency_ms || "?") + " ms");
      window.lastInferQ = q;
      window.lastInferA = d.answer || "";
      const scoreBtn = document.getElementById("inferScoreBtn");
      if (scoreBtn) scoreBtn.disabled = !window.lastInferA;
    } catch (e) {
      showErr(e.message || String(e));
      appendChat("a", "error: " + (e.message || e));
    } finally {
      if (btn) btn.disabled = false;
    }
  };

  window.inferClear = function () {
    const box = document.getElementById("inferChat");
    if (box) box.innerHTML = "";
    showErr("");
    window.lastInferQ = "";
    window.lastInferA = "";
    const scoreBtn = document.getElementById("inferScoreBtn");
    if (scoreBtn) scoreBtn.disabled = true;
  };

  const q = document.getElementById("inferQ");
  if (q) {
    q.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
        ev.preventDefault();
        window.inferAsk();
      }
    });
  }

  setInterval(window.refreshInferStatus, 8000);
  setInterval(window.refreshInferModels, 15000);
  bindInferPromptChips();
  bindSamplingUserTouch();
  renderInferCheatSheets(null);
  window.refreshInferStatus();
  window.refreshInferModels();
  window.toggleRepetitionPenalty();
})();
