"""Self-authored agent index for app11.nextaura.us — scrape-first, no HTML guessing."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

HOST = os.environ.get("PUBLIC_HOST", "app11.nextaura.us")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://app11.nextaura.us")
CANONICAL_URL = PUBLIC_URL
FIT_HOST = os.environ.get("FIT_HOST", "app11.nextaura.fit")
FIT_URL = os.environ.get("FIT_URL", "https://app11.nextaura.fit")
APP_ID = "app11"
SCHEMA_VERSION = "1"
VOICE_TO_PLAN_URL = os.environ.get("VOICE_TO_PLAN_URL", "https://app11.nextaura.fit")

DATA_SOURCE_CATALOG: list[dict[str, str]] = [
    {"id": "mixed", "label": "Mixed (local JSONL + HF streams)", "mode": "pretrain", "notes": "Default. Interleaves local corpus with enabled HF chips."},
    {"id": "fineweb", "label": "FineWeb stream only", "mode": "pretrain", "notes": "HuggingFaceFW/fineweb configs (sample-10BT, etc.)."},
    {"id": "jsonl", "label": "Local JSONL only", "mode": "either", "notes": "Single file from data/. Stop at Target GB or epoch cycles."},
    {"id": "local-corpus-mix", "label": "Local curated mix", "mode": "pretrain", "notes": "2400-clean + 4563-curated + nextaura-extract merged."},
    {"id": "sft-intelligence", "label": "SFT intelligence seeds", "mode": "finetune", "notes": "sft-full-merged.jsonl. Use fine-tuning scaling table; load pretrain ckpt first."},
    {"id": "fable-traces", "label": "Fable stream only", "mode": "pretrain", "notes": "Glint-Research/Fable-5-traces."},
    {"id": "sol-traces", "label": "GPT-5.6 Sol traces", "mode": "pretrain", "notes": "greghavens/gpt-5.6-sol-coding-and-debugging-traces."},
    {"id": "kimi-k3-traces", "label": "Kimi K3 traces", "mode": "pretrain", "notes": "greghavens/kimi-k3-coding-and-debugging-traces."},
    {"id": "unsolved-math", "label": "UnsolvedMath", "mode": "pretrain", "notes": "ulamai/UnsolvedMath problems."},
    {"id": "nemotron-math", "label": "Nemotron SFT Math v4", "mode": "pretrain", "notes": "nvidia/Nemotron-SFT-Math-v4."},
    {"id": "openassistant-rm", "label": "OpenAssistant RM (HH-RLHF)", "mode": "pretrain", "notes": "Preference / reward-style rows."},
    {"id": "multimodal", "label": "Multimodal HF stream", "mode": "pretrain", "notes": "Pick hf_dataset_id from /api/health multimodal_datasets."},
    {"id": "vjepa", "label": "V-JEPA HF stream", "mode": "pretrain", "notes": "Scalar 12d projection; pick vjepa dataset from health."},
]

UI_SECTIONS: list[dict[str, str]] = [
    {"id": "sectionData", "title": "Data", "column": "left", "summary": "data_source, Target GB, Chinchilla pretrain/finetune scaling tables with lr+epoch recommendations per row."},
    {"id": "sectionArch", "title": "Architecture", "column": "left", "summary": "n_layer, n_embd, n_head, block_size, optional ffn_dim/vocab_size. Locked after step>0."},
    {"id": "sectionOpt", "title": "Optimizer", "column": "left", "summary": "learning_rate, batch_size, micro_batch, grad_accum, warmup, weight_decay, dropout."},
    {"id": "sectionCorpus", "title": "Data mix", "column": "left", "summary": "mix local %, curated mix toggle, jsonl file, epoch cycles (JSONL stop), HF stream chips."},
    {"id": "sectionAdv", "title": "Advanced", "column": "left", "summary": "lr_schedule, total_train_steps, mixed_precision, save_every_n_steps, flash attn, scalar proj, grad ckpt, tokenizer."},
    {"id": "sectionRunHistory", "title": "Run history", "column": "left", "summary": "Pretrain/SFT log with RM deltas, verdicts, Ox Alpha seeds. GET /api/runs/history."},
    {"id": "sectionRecipe", "title": "Live recipe", "column": "left", "summary": "Active vs Pending config from /api/status vs form. Gold on Start."},
    {"id": "sectionCkpt", "title": "Checkpoint", "column": "left", "summary": "Upload/load .pt, HF import, delete, reset session."},
    {"id": "sectionInfer", "title": "Stage 2 inference", "column": "right", "summary": "Ask, temperature, repetition penalty, reward model score."},
]

API_ENDPOINTS: list[dict[str, Any]] = [
    {"method": "GET", "path": "/", "auth": False, "summary": "Human UI (Stage 1 train + Stage 2 infer). Prefer agent.json for machines."},
    {"method": "GET", "path": "/agent.json", "auth": False, "summary": "Full agent contract (this document, JSON). Primary scrape target."},
    {"method": "GET", "path": "/agent.txt", "auth": False, "summary": "Plain-text agent contract."},
    {"method": "GET", "path": "/.well-known/agent.json", "auth": False, "summary": "Well-known alias of /agent.json."},
    {"method": "GET", "path": "/llms.txt", "auth": False, "summary": "LLMs.txt discovery index with curated links."},
    {"method": "GET", "path": "/agent-graph.json", "auth": False, "summary": "Engineering memory graph for app7/app8/app9 GPU labs (nodes, edges, invariants)."},
    {"method": "GET", "path": "/robots.txt", "auth": False, "summary": "Crawler policy; allows agent surfaces."},
    {"method": "GET", "path": "/sitemap.xml", "auth": False, "summary": "Sitemap of machine-readable GET endpoints."},
    {"method": "GET", "path": "/api/health", "auth": False, "summary": "Boot defaults, device, secrets loaded, data_sources, chinchilla targets."},
    {"method": "GET", "path": "/api/status", "auth": False, "summary": "Live training snapshot: step, loss, bytes, recipe fields, can_resume."},
    {"method": "GET", "path": "/api/log", "auth": False, "summary": "Terminal mirror lines. Query: since=<index>."},
    {"method": "GET", "path": "/api/model/estimate", "auth": False, "summary": "Param count + chinchilla bundle for arch query params."},
    {"method": "GET", "path": "/api/scaling/chinchilla", "auth": False, "summary": "Chinchilla pretrain + finetune multiplier tables."},
    {"method": "GET", "path": "/api/capabilities", "auth": False, "summary": "Trainer feature flags."},
    {"method": "GET", "path": "/api/data", "auth": False, "summary": "List local JSONL files in data/ with row counts."},
    {"method": "GET", "path": "/api/checkpoints", "auth": False, "summary": "Saved .pt files on server disk."},
    {"method": "GET", "path": "/api/infer/status", "auth": False, "summary": "Whether a checkpoint is loaded for inference."},
    {"method": "GET", "path": "/api/export/checkpoint", "auth": False, "summary": "Download current weights as .pt (large)."},
    {"method": "POST", "path": "/api/start", "auth": False, "summary": "Start training. Body: StartIn (see start_body_schema). Blocked if already running."},
    {"method": "POST", "path": "/api/resume", "auth": False, "summary": "Resume paused/stopped run with same recipe."},
    {"method": "POST", "path": "/api/stop", "auth": False, "summary": "Stop training; saves checkpoint."},
    {"method": "POST", "path": "/api/pause", "auth": False, "summary": "Toggle pause."},
    {"method": "POST", "path": "/api/import/checkpoint", "auth": False, "summary": "Upload .pt multipart file."},
    {"method": "POST", "path": "/api/import/checkpoint/local", "auth": False, "summary": "Load saved checkpoint by name."},
    {"method": "POST", "path": "/api/import/data", "auth": False, "summary": "Upload JSONL to data/."},
    {"method": "POST", "path": "/api/convert/hf", "auth": False, "summary": "Import HuggingFace GPT-2 weights. Body: {repo_id}."},
    {"method": "POST", "path": "/api/checkpoints/reset-session", "auth": False, "summary": "Clear GPU weights; keep disk files."},
    {"method": "POST", "path": "/api/checkpoints/delete", "auth": False, "summary": "Bulk delete local .pt except loaded."},
    {"method": "DELETE", "path": "/api/checkpoints/{name}", "auth": False, "summary": "Delete one local checkpoint."},
    {"method": "POST", "path": "/api/infer/ask", "auth": False, "summary": "Generate text. Body: {question, max_new_tokens, temperature, repetition_penalty, use_repetition_penalty}."},
    {"method": "GET", "path": "/api/runs/history", "auth": False, "summary": "Pretrain/SFT run log with verdicts, RM deltas, Ox Alpha seeds."},
    {"method": "POST", "path": "/api/runs/record", "auth": False, "summary": "Append engineering note, corpus build, or reward eval to run history."},
    {"method": "POST", "path": "/api/eval/reward-panel", "auth": False, "summary": "Generate on N benchmark prompts + OpenAssistant RM mean. Body: limit, baseline_mean, record."},
    {"method": "POST", "path": "/api/eval/reward", "auth": False, "summary": "Score Q/A with OpenAssistant reward model."},
]

START_BODY_SCHEMA: dict[str, Any] = {
    "required_for_agents": ["data_source", "target_gb", "learning_rate", "batch_size", "n_layer", "n_head", "n_embd", "block_size"],
    "stop_conditions": [
        "target_gb: byte budget (primary for streams and most runs)",
        "total_train_steps: optional hard cap (Advanced, when apply.total_train_steps)",
        "epoch_cycles + epochs: JSONL-only replay stop (can finish before target_gb)",
    ],
    "apply_toggles": "Each field has apply.<name> bool — only checked fields override active run on Start.",
    "fields": {
        "data_source": "See data_sources catalog",
        "target_gb": "0.001–25 GB. User stop cap; compare to scaling table rows.",
        "learning_rate": "pretrain 2e-4–3e-4; finetune 1e-5–1e-4",
        "epoch_cycles": "false for streams; rare 1–5 for tiny JSONL smoke",
        "jsonl_file": "Filename in data/",
        "mix_local_jsonl": "Interleave local into mixed streams",
        "jsonl_mix_ratio": "0–1 local fraction in mixed mode",
        "hf_stream_*": "Booleans for fable, sol, kimi, nemotron, math, pref, fineweb",
    },
}

SCALING_GUIDANCE: dict[str, Any] = {
    "law": "Chinchilla: ~20 tokens per parameter for compute-optimal pretrain",
    "bytes_per_token": 3.5,
    "ui_toggle": "Pretrain vs Fine-tuning button above scaling table",
    "pretrain": {
        "reference_row": "1× Chinchilla optimal",
        "lr_range": "2e-4–3e-4",
        "epochs": "off (use Target GB)",
        "multipliers": ["0.25× smoke", "0.5× light", "1× optimal", "2× extended", "3× heavy", "10× legacy cap"],
    },
    "finetune": {
        "reference_row": "0.1× standard SFT",
        "lr_range": "1e-5–1e-4",
        "epochs": "off; 1–5 only for smoke on <1 MB JSONL",
        "multipliers": ["0.01× smoke", "0.05× light", "0.1× standard", "0.25× extended", "0.5× heavy", "1× full replay"],
        "requires": "Pretrained checkpoint — seed-only SFT from scratch fails at 17M",
    },
    "live_api": "GET /api/scaling/chinchilla or chinchilla block in /api/model/estimate",
}

AGENT_WORKFLOW: list[str] = [
    "1. GET /agent.json and GET /agent-graph.json — do not scrape HTML for config.",
    "2. GET /api/runs/history — auto backfills from disk checkpoints; read before Start.",
    "3. GET /api/health — device, defaults, whether HF_TOKEN and COS are loaded.",
    "4. GET /api/status — active run, step, loss, can_resume, loaded checkpoint.",
    "5. Pick data_source + scaling mode (pretrain vs finetune) from UI tables or scaling_guidance.",
    "6. Set target_gb to matching scaling row (your GPU/time cap, not universal).",
    "7. POST /api/start with body matching form (see start_body_schema). Use apply toggles like UI 📎.",
    "8. Poll GET /api/status and GET /api/log?since=N until complete or target reached.",
    "9. POST /api/eval/reward-panel {limit:20, baseline_mean, record:true} — gate keep vs discard.",
    "10. POST /api/runs/record for corpus builds and engineering notes agents discover.",
    "11. Finetune: reload step 3898, switch to finetune table, lower lr, shrink batch if rows < batch.",
]

DISCOVERY_PATHS: dict[str, str] = {
    "agent_json": "/agent.json",
    "agent_txt": "/agent.txt",
    "well_known_agent_json": "/.well-known/agent.json",
    "llms_txt": "/llms.txt",
    "agent_graph": "/agent-graph.json",
    "robots_txt": "/robots.txt",
    "sitemap_xml": "/sitemap.xml",
    "human_ui": "/",
    "health": "/api/health",
    "status": "/api/status",
    "scaling": "/api/scaling/chinchilla",
}


def _discovery_urls() -> dict[str, str]:
    return {k: f"{PUBLIC_URL}{path}" for k, path in DISCOVERY_PATHS.items()}


def build_agent_contract(*, live: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full machine index. `live` from server health/status hooks."""
    live = live or {}
    hf_ok = bool(live.get("hf_token_loaded"))
    cos_ok = bool(live.get("cos_configured"))
    device = live.get("device", "unknown")
    running = bool(live.get("running"))
    train_step = int(live.get("train_step") or 0)
    api_health = "ok" if hf_ok and cos_ok else ("degraded" if hf_ok or cos_ok else "down")
    ui_up = True
    success = ui_up and api_health in ("ok", "degraded")

    purpose = (
        "GPU experiment lab: GPT-style pretrain and SFT on mixed HF streams, local JSONL, "
        "multimodal and V-JEPA data. Live Chinchilla pretrain/finetune scaling tables in UI "
        "with per-row lr and epoch recommendations. Stage 2 inference + reward scoring."
    )
    how_built = (
        "FastAPI + PyTorch TrainEngine on UpCloud train-1 (212.147.237.163), "
        "Cloudflare worker nextaura-app11-us → origin. Optional IBM COS for checkpoint persistence. "
        "Fork of app2 pretrain stack; default ~51M params (8L×512d×8h, block 256, vocab 50257)."
    )
    page_text = (
        f"{HOST} — NextAura Multimodal Fine-tune. Stage 1: stream+train with scaling tables. "
        f"Stage 2: infer. Agent contract at /agent.json. Health /api/health. Status /api/status."
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "app_id": APP_ID,
        "host": HOST,
        "url": PUBLIC_URL,
        "canonical_url": CANONICAL_URL,
        "aliases": [
            {
                "host": FIT_HOST,
                "url": FIT_URL,
                "role": "marketing_entry_from_nextaura.fit",
                "app": "Lumen DAW (IBM Code Engine) for / — agent + /api/* proxied to GPU lab",
                "agent_index": f"GET {FIT_URL}/agent.json proxies to GPU lab (canonical {CANONICAL_URL})",
            }
        ],
        "origin_url": live.get("origin_url"),
        "origin": "digitalocean-gpu",
        "worker_or_pages": "nextaura-app11-us",
        "related_hosts": {
            "canonical_training_lab": CANONICAL_URL,
            "marketing_fit_entry": FIT_URL,
            "voice_to_plan": VOICE_TO_PLAN_URL,
            "app2_pretrain": "https://app2.nextaura.us",
            "app7_text_lab": "https://app7.nextaura.us",
            "app8_mcf_lab": "https://app8.nextaura.us",
        },
        "engineering_memory": {
            "graph_url": f"{PUBLIC_URL}/agent-graph.json",
            "labs": ["app7", "app8", "app9"],
            "scrape_first": True,
            "invariants": [
                "V-JEPA rows are 50–200 tokens; FineWeb rows are thousands — do not copy FineWeb batch/GB onto V-JEPA.",
                "MCF v2: pack 128 frames, batch 256, ~1.5B token target.",
                "Finetune requires a pretrained checkpoint.",
                "Do not stop app7 text runs for app8/app9 V-JEPA work.",
            ],
        },
        "purpose": purpose,
        "how_built": how_built,
        "page_text": page_text,
        "status": "up" if success else "fail",
        "success": success,
        "http_status": 200,
        "failure_reason": None if success else "Missing HF_TOKEN or IBM_CLOUD_API_KEY",
        "health": {
            "ui": ui_up,
            "api": api_health,
            "training": "running" if running else ("idle" if train_step == 0 else "paused_or_complete"),
            "origin": "droplet",
            "device": device,
            "model_ready": train_step > 0,
        },
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "collector": "self",
        "discovery": _discovery_urls(),
        "ui_sections": UI_SECTIONS,
        "data_sources": DATA_SOURCE_CATALOG,
        "scaling_guidance": SCALING_GUIDANCE,
        "agent_workflow": AGENT_WORKFLOW,
        "start_body_schema": START_BODY_SCHEMA,
        "api_endpoints": API_ENDPOINTS,
        "terminology": {
            "target_gb": "User stop cap (GPU/time budget). Not the same as 1× Chinchilla optimal.",
            "forward_pass": "One model forward (loss computed).",
            "gradient_accumulation": "Multiple forwards before one optimizer update.",
            "training_step": "One optimizer weight update.",
            "epoch_cycles": "Replay local JSONL N times; JSONL-only stop; usually off.",
            "chinchilla_row": "Scaling table row: tokens, GB, lr range, epoch guidance.",
        },
        "live": live,
    }


def render_agent_txt(contract: dict[str, Any]) -> str:
    lines = [
        "NextAura agent contract",
        f"host: {contract['host']}",
        f"app_id: {contract['app_id']}",
        f"schema_version: {contract['schema_version']}",
        "",
        "purpose:",
        f"  {contract['purpose']}",
        "",
        "how_built:",
        f"  {contract['how_built']}",
        "",
        f"status: {contract['status']}",
        f"success: {str(contract['success']).lower()}",
        f"health.ui: {str(contract['health']['ui']).lower()}",
        f"health.api: {contract['health']['api']}",
        f"health.device: {contract['health'].get('device', '?')}",
        "",
        "discovery (fetch these, not HTML):",
    ]
    for key, url in contract.get("discovery", {}).items():
        lines.append(f"  {key}: {url}")
    lines.extend(["", "agent_workflow:"])
    for step in contract.get("agent_workflow", []):
        lines.append(f"  {step}")
    lines.extend(["", "scaling_guidance:"])
    sg = contract.get("scaling_guidance", {})
    lines.append(f"  law: {sg.get('law', '')}")
    lines.append(f"  ui_toggle: {sg.get('ui_toggle', '')}")
    for mode in ("pretrain", "finetune"):
        block = sg.get(mode, {})
        if block:
            lines.append(f"  {mode}: lr {block.get('lr_range', '?')} · ref {block.get('reference_row', '?')}")
    lines.extend(["", "data_sources:"])
    for ds in contract.get("data_sources", []):
        lines.append(f"  {ds['id']}: [{ds['mode']}] {ds['label']}")
    lines.extend(["", "ui_sections:"])
    for sec in contract.get("ui_sections", []):
        lines.append(f"  {sec['id']}: {sec['title']} — {sec['summary']}")
    lines.extend(["", "api (GET discovery first):"])
    for ep in contract.get("api_endpoints", []):
        auth = "auth" if ep.get("auth") else "open"
        lines.append(f"  {ep['method']} {ep['path']} [{auth}] — {ep['summary']}")
    live = contract.get("live") or {}
    if live:
        lines.extend([
            "",
            "live_snapshot:",
            f"  train_step: {live.get('train_step', 0)}",
            f"  last_loss: {live.get('last_loss')}",
            f"  running: {live.get('running', False)}",
            f"  target_gb: {live.get('target_gb')}",
            f"  data_source: {live.get('data_source')}",
        ])
    return "\n".join(lines) + "\n"


def render_llms_txt(contract: dict[str, Any]) -> str:
    d = contract.get("discovery", {})
    return "\n".join([
        f"# {HOST}",
        "",
        f"> {contract['purpose']}",
        "",
        "## Agent contract (start here)",
        f"- [Agent JSON]({d.get('agent_json', PUBLIC_URL + '/agent.json')}): Full machine index — APIs, UI sections, scaling, workflow",
        f"- [Agent graph]({d.get('agent_graph', PUBLIC_URL + '/agent-graph.json')}): app7/app8/app9 engineering memory (nodes, edges, invariants)",
        f"- [Agent TXT]({d.get('agent_txt', PUBLIC_URL + '/agent.txt')}): Plain-text contract",
        f"- [Well-known]({d.get('well_known_agent_json', PUBLIC_URL + '/.well-known/agent.json')}): Same as agent.json",
        f"- [Sitemap]({d.get('sitemap_xml', PUBLIC_URL + '/sitemap.xml')}): All GET endpoints",
        "",
        "## Live state",
        f"- [Health]({d.get('health', PUBLIC_URL + '/api/health')}): Defaults, device, secrets, data source list",
        f"- [Status]({d.get('status', PUBLIC_URL + '/api/status')}): Active training run snapshot",
        f"- [Chinchilla scaling]({d.get('scaling', PUBLIC_URL + '/api/scaling/chinchilla')}): Pretrain + finetune tables with recommendations",
        "",
        "## Human UI",
        f"- [Stage 1 + 2 UI]({d.get('human_ui', PUBLIC_URL + '/')}): Browser control panel (agents should prefer JSON above)",
        "",
        "## Key rules for agents",
        "- Target GB is the user's stop cap (GPU/time), not universal Chinchilla optimal.",
        "- Toggle Pretrain vs Fine-tuning scaling table before picking target_gb and lr.",
        "- Pretrain: lr 2e-4–3e-4, epochs off, mixed/HF streams OK.",
        "- Finetune: lr 1e-5–1e-4, smaller GB, load pretrain checkpoint first.",
        "- Do not train seed-only SFT from scratch — fails at 17M (loss ~5+).",
        "- Loss <1 on polluted mixed runs = overfit warning, not success.",
        "",
    ]) + "\n"


def render_robots_txt() -> str:
    # Put agent-allow rules BEFORE any generic blocks so crawlers see them first.
    return "\n".join([
        "# NextAura app9 GPU training lab — canonical: " + CANONICAL_URL,
        "# AI crawlers: agent.json is the source of truth (not HTML).",
        "",
        "User-agent: GPTBot",
        "User-agent: ClaudeBot",
        "User-agent: CCBot",
        "User-agent: Google-Extended",
        "Allow: /agent.json",
        "Allow: /agent-graph.json",
        "Allow: /agent.txt",
        "Allow: /llms.txt",
        "Allow: /.well-known/",
        "Allow: /api/health",
        "Allow: /api/status",
        "Allow: /api/scaling/",
        "Allow: /api/capabilities",
        "Disallow: /api/export/",
        "Disallow: /api/import/",
        "Disallow: /api/infer/ask",
        "",
        "User-agent: *",
        "Allow: /",
        "Allow: /agent.json",
        "Allow: /agent-graph.json",
        "Allow: /agent.txt",
        "Allow: /llms.txt",
        "Allow: /.well-known/",
        "Allow: /api/health",
        "Allow: /api/status",
        "Allow: /api/capabilities",
        "Allow: /api/scaling/",
        "Allow: /api/model/estimate",
        "Allow: /api/data",
        "Allow: /api/checkpoints",
        "Allow: /api/infer/status",
        "Allow: /api/log",
        "Disallow: /api/export/",
        "Disallow: /api/import/",
        "Disallow: /api/start",
        "Disallow: /api/stop",
        "Disallow: /api/infer/ask",
        "",
        f"Sitemap: {CANONICAL_URL}/sitemap.xml",
        "",
    ])


def render_sitemap_xml() -> str:
    urls = [
        ("/", "daily", "1.0"),
        ("/agent.json", "hourly", "1.0"),
        ("/agent-graph.json", "weekly", "0.95"),
        ("/agent.txt", "hourly", "0.9"),
        ("/llms.txt", "weekly", "0.9"),
        ("/.well-known/agent.json", "hourly", "0.9"),
        ("/api/health", "always", "0.8"),
        ("/api/status", "always", "0.8"),
        ("/api/capabilities", "weekly", "0.7"),
        ("/api/scaling/chinchilla", "weekly", "0.8"),
        ("/api/data", "daily", "0.6"),
        ("/api/checkpoints", "daily", "0.5"),
        ("/api/infer/status", "always", "0.7"),
    ]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for path, freq, priority in urls:
        parts.append("  <url>")
        parts.append(f"    <loc>{PUBLIC_URL}{path}</loc>")
        parts.append(f"    <lastmod>{today}</lastmod>")
        parts.append(f"    <changefreq>{freq}</changefreq>")
        parts.append(f"    <priority>{priority}</priority>")
        parts.append("  </url>")
    parts.append("</urlset>")
    return "\n".join(parts) + "\n"
