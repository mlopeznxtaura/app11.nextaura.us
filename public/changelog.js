/** Footer changelog — loaded from /static/changelog.json (built from git log). */
(function () {
  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (e) {
      return iso;
    }
  }

  async function loadChangelog() {
    const body = document.getElementById("changelogBody");
    const meta = document.getElementById("changelogMeta");
    if (!body) return;
    try {
      const r = await fetch("/static/changelog.json?v=" + Date.now());
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      const entries = data.entries || [];
      if (meta) {
        const head = data.head_short || (data.head || "").slice(0, 7);
        meta.textContent =
          (data.repo || "app7") +
          " · " +
          (data.branch || "main") +
          " @ " +
          (head || "—") +
          " · updated " +
          fmtDate(data.generated_at);
      }
      if (!entries.length) {
        body.innerHTML = '<p class="hint">No changelog entries yet.</p>';
        return;
      }
      body.innerHTML =
        '<ul class="changelog-list">' +
        entries
          .map(function (e) {
            return (
              '<li><span class="changelog-date">' +
              esc(fmtDate(e.date)) +
              '</span> <code class="changelog-sha">' +
              esc(e.short_sha || (e.sha || "").slice(0, 7)) +
              '</code> <a href="' +
              esc(e.url || "#") +
              '" target="_blank" rel="noopener">' +
              esc(e.subject) +
              "</a> <span class=\"changelog-author\">— " +
              esc(e.author) +
              "</span></li>"
            );
          })
          .join("") +
        "</ul>";
    } catch (err) {
      body.innerHTML =
        '<p class="hint" style="color:#f87171">Changelog unavailable: ' + esc(err.message) + "</p>";
    }
  }

  document.addEventListener("DOMContentLoaded", loadChangelog);
})();
