(function () {
  function isDistillSource(ds) {
    return (
      ds === "distill-merged" ||
      ds === "distill-live" ||
      ds === "distill-coding" ||
      ds === "distill-physics" ||
      ds === "distill-math"
    );
  }

  function isDistillLive(ds) {
    return ds === "distill-live";
  }

  function selectedDistillTopics() {
    const out = [];
    ["distillTopicCoding", "distillTopicPhysics", "distillTopicMath"].forEach(function (id, i) {
      const el = document.getElementById(id);
      if (el && el.checked) out.push(["coding", "physics", "math"][i]);
    });
    return out;
  }

  window.toggleDistillGenerator = function () {
    const ds = (document.getElementById("dataSource") || {}).value || "";
    const row = document.getElementById("distillGeneratorRow");
    const btn = document.getElementById("distillGenerateBtn");
    if (row) row.style.display = isDistillSource(ds) ? "grid" : "none";
    if (btn) btn.style.display = isDistillLive(ds) ? "none" : "";
    const el = document.getElementById("distillTopicStatus");
    if (el && isDistillLive(ds)) {
      el.textContent =
        "Live distill: teachers generate answers during training → distill-live.jsonl. No pre-generate step.";
    }
  };

  window.refreshDistillTopicStatus = async function () {
    const el = document.getElementById("distillTopicStatus");
    if (!el) return;
    try {
      const d = await fetch("/api/distill/teachers").then(function (r) {
        return r.json();
      });
      const parts = (d.catalog || [])
        .filter(function (t) {
          return t.id !== "merged";
        })
        .map(function (t) {
          const teacher = t.teacher ? " · " + t.teacher : "";
          return t.id + (t.exists ? ": " + (t.rows || 0) + " rows" + teacher : ": —");
        });
      const merged = (d.catalog || []).find(function (t) {
        return t.id === "merged";
      });
      if (merged) {
        parts.push(
          "merged" + (merged.exists ? ": " + (merged.rows || 0) + " rows" : ": —")
        );
      }
      el.textContent = parts.length ? parts.join(" · ") : "No distill files yet";
      el.style.color = "#94a3b8";
    } catch (_) {
      el.textContent = "Distill API unavailable";
    }
  };

  window.generateDistillTopics = async function () {
    const btn = document.getElementById("distillGenerateBtn");
    const status = document.getElementById("distillTopicStatus");
    const topics = selectedDistillTopics();
    if (!topics.length) {
      if (status) {
        status.textContent = "Select at least one topic";
        status.style.color = "#f87171";
      }
      return;
    }
    const prev = btn ? btn.textContent : "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Generating…";
    }
    if (status) {
      status.textContent = "Generating distill " + topics.join(", ") + " (teachers on GPU)…";
      status.style.color = "#93c5fd";
    }
    try {
      const r = await fetch("/api/distill/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topics: topics,
          benchmark_limit: 12,
          write_merged: topics.length > 1,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || d.detail || "generate failed");
      const summary = (d.topics || []).map(function (b) {
        return b.topic + " " + b.kept + " (" + (b.teacher || "?") + ")";
      });
      if (status) {
        status.textContent = "Generated: " + summary.join(", ");
        status.style.color = "#86efac";
      }
      if (typeof line === "function") {
        line("[GREEN] Distill corpus → " + summary.join(", "), "green");
      }
      window.refreshDistillTopicStatus();
      if (typeof onFormChange === "function") onFormChange();
    } catch (e) {
      if (status) {
        status.textContent = e.message || String(e);
        status.style.color = "#f87171";
      }
      if (typeof line === "function") {
        line("[RED] Distill generate: " + (e.message || e), "red");
      }
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prev || "Generate distill";
      }
    }
  };

  window.refreshDistillTopicStatus();
})();
