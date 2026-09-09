(function () {
  async function evalApi(path, opts) {
    const r = await fetch(path, opts || {});
    const d = await r.json().catch(function () {
      return { error: r.statusText };
    });
    if (!r.ok) throw new Error(d.error || r.statusText || "HTTP " + r.status);
    return d;
  }

  function fmtRm(v) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toFixed(4);
  }

  function renderTeachersRef(distill) {
    const el = document.getElementById("evalTeachersRef");
    if (!el || !distill) return;
    const targets = distill.target_teachers || {};
    const wired = distill.teachers || {};
    const catalog = distill.catalog || [];
    const rows = catalog
      .filter(function (t) {
        return t.id !== "merged";
      })
      .map(function (t) {
        const target = targets[t.id];
        const wiredTeacher = (wired[t.id] && wired[t.id].label) || "?";
        const rowsNote = t.exists ? t.rows + " rows" : "no corpus";
        const targetNote =
          target && target !== (wired[t.id] && wired[t.id].repo) ? " · target " + target : "";
        return (
          "<strong>" +
          t.id +
          "</strong>: wired " +
          wiredTeacher +
          " (" +
          rowsNote +
          ")" +
          targetNote
        );
      });
    const merged = catalog.find(function (t) {
      return t.id === "merged";
    });
    if (merged) {
      rows.push("<strong>merged</strong>: " + (merged.exists ? merged.rows + " rows" : "—"));
    }
    el.innerHTML =
      rows.join(" · ") ||
      "Distill teachers (offline JSONL generation only — not used when training on HF streams).";
  }

  function renderLeaderboard(d) {
    const body = document.getElementById("evalLeaderboardBody");
    if (!body) return;
    const rows = (d.our_checkpoints && d.our_checkpoints.checkpoints) || d.checkpoints || [];
    if (!rows.length) {
      body.innerHTML =
        '<tr><td colspan="4" class="hint">No RM evals yet — train until a checkpoint saves (every 500 steps) or compare .pt files manually.</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map(function (r) {
        const ck = r.checkpoint || "—";
        const loaded = d.loaded === ck ? " ✓" : "";
        return (
          "<tr><td>" +
          ck +
          loaded +
          "</td><td>" +
          fmtRm(r.mean_reward) +
          "</td><td>" +
          (r.delta != null ? fmtRm(r.delta) : "—") +
          "</td><td class='hint'>" +
          (r.ts || "").slice(0, 19) +
          "</td></tr>"
        );
      })
      .join("");
  }

  window.refreshEvalPanel = async function () {
    try {
      const d = await evalApi("/api/eval/benchmark");
      renderTeachersRef(d.distill);
      renderLeaderboard(d);
      const live = d.eval || {};
      const status = document.getElementById("evalLiveStatus");
      if (status) {
        status.textContent =
          "Best RM: " +
          (live.best_checkpoint || "—") +
          " · score " +
          fmtRm(live.best_mean) +
          " · loaded " +
          (d.loaded || "—");
      }
    } catch (e) {
      const status = document.getElementById("evalLiveStatus");
      if (status) status.textContent = "Eval API: " + (e.message || e);
    }
  };

  window.compareSelectedCheckpoints = async function () {
    const sel = document.getElementById("ckptSaved");
    const names = [];
    if (sel && sel.value) names.push(sel.value);
    const extra = (document.getElementById("evalCompareList") || {}).value || "";
    extra
      .split(/[\s,]+/)
      .map(function (s) {
        return s.trim();
      })
      .filter(Boolean)
      .forEach(function (n) {
        if (names.indexOf(n) < 0) names.push(n);
      });
    if (names.length < 2) {
      alert("Pick a checkpoint and/or list 2+ .pt names to compare.");
      return;
    }
    const btn = document.getElementById("evalCompareBtn");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Comparing…";
    }
    try {
      const d = await evalApi("/api/eval/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ names: names, record: true, restore_loaded: true }),
      });
      const best = d.best;
      if (typeof line === "function") {
        line(
          "[GREEN] Compare best: " +
            (best && best.checkpoint) +
            " RM " +
            fmtRm(best && best.mean_reward),
          "green"
        );
      }
      await window.refreshEvalPanel();
      if (typeof refreshInferModels === "function") refreshInferModels();
    } catch (e) {
      if (typeof line === "function") line("[RED] Compare: " + e.message, "red");
      alert(e.message || String(e));
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Compare checkpoints (RM)";
      }
    }
  };

  setInterval(window.refreshEvalPanel, 20000);
  window.refreshEvalPanel();
})();
