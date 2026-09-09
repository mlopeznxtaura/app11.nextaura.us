(function () {
  function isSftTopicSource(ds) {
    return ds === "sft-coding" || ds === "sft-physics" || ds === "sft-math";
  }

  function selectedSftTopics() {
    const out = [];
    ["sftTopicCoding", "sftTopicPhysics", "sftTopicMath"].forEach(function (id, i) {
      const el = document.getElementById(id);
      if (el && el.checked) out.push(["coding", "physics", "math"][i]);
    });
    return out;
  }

  window.toggleSftGenerator = function () {
    const ds = (document.getElementById("dataSource") || {}).value || "";
    const row = document.getElementById("sftGeneratorRow");
    const isSft = isSftTopicSource(ds) || ds === "sft-intelligence";
    if (row) row.style.display = isSft ? "grid" : "none";
  };

  window.refreshSftTopicStatus = async function () {
    const el = document.getElementById("sftTopicStatus");
    if (!el) return;
    try {
      const d = await fetch("/api/sft/topics").then(function (r) {
        return r.json();
      });
      const parts = (d.topics || [])
        .filter(function (t) {
          return t.id !== "merged";
        })
        .map(function (t) {
          return t.id + (t.exists ? ": " + (t.rows || 0) + " rows" : ": —");
        });
      el.textContent = parts.length ? parts.join(" · ") : "No topic files yet";
      el.style.color = "#94a3b8";
    } catch (_) {
      el.textContent = "SFT topic API unavailable";
    }
  };

  window.generateSftTopics = async function () {
    const btn = document.getElementById("sftGenerateBtn");
    const status = document.getElementById("sftTopicStatus");
    const topics = selectedSftTopics();
    if (!topics.length) {
      if (status) {
        status.textContent = "Select at least one topic";
        status.style.color = "#f87171";
      }
      return;
    }
    const hfAug = !!(document.getElementById("sftHfAugment") || {}).checked;
    const prev = btn ? btn.textContent : "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Generating…";
    }
    if (status) {
      status.textContent = "Generating " + topics.join(", ") + "…";
      status.style.color = "#93c5fd";
    }
    try {
      const r = await fetch("/api/sft/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topics: topics,
          hf_augment: hfAug,
          hf_limit: 40,
          write_merged: topics.length > 1,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || d.detail || "generate failed");
      const summary = (d.topics || []).map(function (b) {
        return b.topic + " " + b.kept;
      });
      if (status) {
        status.textContent = "Generated: " + summary.join(", ");
        status.style.color = "#86efac";
      }
      if (typeof line === "function") line("[GREEN] SFT topics → " + summary.join(", "), "green");
      window.refreshSftTopicStatus();
      if (typeof onFormChange === "function") onFormChange();
    } catch (e) {
      if (status) {
        status.textContent = e.message || String(e);
        status.style.color = "#f87171";
      }
      if (typeof line === "function") line("[RED] SFT generate: " + (e.message || e), "red");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prev || "Generate SFT";
      }
    }
  };

  window.refreshSftTopicStatus();
})();
