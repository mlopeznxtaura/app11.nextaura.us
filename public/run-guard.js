(function () {
  function showPreflightBlock(pf) {
    const lines = (pf.errors || []).concat(pf.warnings || []);
    const msg = lines.join("\n\n");
    if (typeof line === "function") {
      (pf.errors || []).forEach(function (e) {
        line("[RED] Preflight: " + e, "red");
      });
      (pf.warnings || []).forEach(function (w) {
        line("[YELLOW] Preflight: " + w, "yellow");
      });
    }
    alert(msg || "Preflight blocked Start.");
  }

  const origStart = window.startStream;
  if (typeof origStart === "function") {
    window.startStream = async function (resume) {
      if (!resume) {
        if (typeof window.isRunWorkflowReady === "function" && !window.isRunWorkflowReady()) {
          alert("Workflow incomplete — lock steps 1–3 in the Run playbook (top of page) before Start.");
          if (typeof window.renderRunWorkflow === "function") window.renderRunWorkflow();
          return;
        }
        try {
          const payload =
            typeof bodyPayload === "function"
              ? { ...bodyPayload(), ...(typeof runGoalPayload === "function" ? runGoalPayload() : {}) }
              : {};
          const r = await fetch("/api/preflight", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const pf = await r.json().catch(function () {
            return { ok: false, errors: [r.statusText] };
          });
          if (!pf.ok) {
            showPreflightBlock(pf);
            return;
          }
          if (pf.warnings && pf.warnings.length) {
            const ok = confirm(
              "Preflight warnings:\n\n" + pf.warnings.join("\n\n") + "\n\nStart anyway?"
            );
            if (!ok) return;
          }
        } catch (e) {
          if (typeof line === "function") line("[RED] Preflight: " + e.message, "red");
          return;
        }
      }
      return await origStart(resume);
    };
  }

  const origRenderCkpt = window.renderCkptStatus;
  window.renderCkptStatus = function (s) {
    const el = document.getElementById("ckptStatus");
    if (!el || !s) return;
    const loaded = s.loaded_checkpoint || (s.checkpoint_name && s.checkpoint_name.endsWith(".pt") ? s.checkpoint_name : "");
    const storage = s.checkpoint_storage || (s.checkpoint_name && !s.checkpoint_name.endsWith(".pt") ? s.checkpoint_name : "");
    if (loaded) {
      let text = "Loaded: " + loaded + " · step " + (s.train_step || 0).toLocaleString() + " · loss " + (s.last_loss != null ? Number(s.last_loss).toFixed(4) : "—");
      if (storage) text += " · archive " + storage;
      if (s.status === "complete" && s.stop_reason_label) text += " · " + s.stop_reason_label;
      el.textContent = text;
    } else if (s.export_ready) {
      el.textContent = "In-memory weights from current run (step " + (s.train_step || 0).toLocaleString() + ").";
    } else {
      el.textContent = "No checkpoint loaded — upload a .pt or train a few steps.";
    }
    if (origRenderCkpt && origRenderCkpt !== window.renderCkptStatus) {
      /* no-op — we replaced fully */
    }
  };

  const origRenderStatus = window.renderStatus;
  if (typeof origRenderStatus === "function") {
    window.renderStatus = function (s) {
      origRenderStatus(s);
      const startBtn = document.getElementById("startBtn");
      const wfOk = typeof window.isRunWorkflowReady === "function" ? window.isRunWorkflowReady() : true;
      window.__app7StartBlockedByRun = !!s.running || s.status === "complete";
      if (startBtn) startBtn.disabled = window.__app7StartBlockedByRun || !wfOk;
      if (typeof window.renderRunWorkflow === "function") window.renderRunWorkflow();
      const importBtn = document.getElementById("importHfBtn");
      const archLocked = !!s.running || (s.train_step || 0) > 0;
      if (importBtn) importBtn.disabled = archLocked;
      const banner = document.getElementById("runCompleteBanner");
      if (banner) {
        if (s.status === "complete") {
          banner.style.display = "block";
          banner.textContent =
            "Run complete" +
            (s.stop_reason_label ? " — " + s.stop_reason_label : "") +
            " · Reset session to start a new run.";
        } else {
          banner.style.display = "none";
        }
      }
    };
  }
})();
