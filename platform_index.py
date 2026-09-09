"""Platform-wide agent scrape index — all hosts, all model checkpoints."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "platform" / "agent_registry.yaml"
if not REGISTRY_PATH.is_file():
    REGISTRY_PATH = ROOT.parent / "platform" / "agent_registry.yaml"
CACHE_PATH = ROOT / "data" / "platform-agent-index.json"

# Inline fallback if registry YAML missing on GPU host
DEFAULT_HOSTS: list[dict[str, Any]] = [
    {
        "id": "app9",
        "name": "app9.nextaura.us",
        "url": "https://app9.nextaura.us",
        "role": "gpu-training-lab",
        "primary": True,
        "paths": {
            "agent": "/agent.json",
            "health": "/api/health",
            "status": "/api/status",
            "checkpoints": "/api/checkpoints",
            "infer_models": "/api/infer/models",
            "benchmark": "/api/eval/benchmark",
            "drive_status": "/api/agent/drive-status",
            "models": "/api/agent/models",
        },
    },
    {
        "id": "app7",
        "name": "app7.nextaura.us",
        "url": "https://app7.nextaura.us",
        "role": "retired-archive",
        "primary": False,
        "paths": {
            "agent": "/agent.json",
            "health": "/agent.json",
            "status": "/agent.json",
        },
    },
    {
        "id": "app8",
        "name": "app8.nextaura.us",
        "url": "https://app8.nextaura.us",
        "role": "gpu-training-lab",
        "primary": True,
        "paths": {
            "agent": "/agent.json",
            "health": "/api/health",
            "status": "/api/status",
            "checkpoints": "/api/checkpoints",
            "infer_models": "/api/infer/models",
            "benchmark": "/api/eval/benchmark",
            "models": "/api/agent/models",
        },
    },
    {
        "id": "app8_fit",
        "name": "app8.nextaura.fit",
        "url": "https://app8.nextaura.fit",
        "role": "lumen-daw",
        "paths": {"health": "/health"},
    },
    {
        "id": "qvrmv3",
        "name": "nextaura-qvrmv3",
        "url": "https://nextaura-qvrmv3.284w7l87aq94.us-south.codeengine.appdomain.cloud",
        "role": "orchestrator",
        "paths": {"health": "/health"},
    },
    {
        "id": "stage1",
        "name": "nextaura-stage1-capture",
        "url": "https://nextaura-stage1-capture.284w7l87aq94.us-south.codeengine.appdomain.cloud",
        "role": "stage1-capture",
        "paths": {"health": "/health"},
    },
    {
        "id": "app3_atp",
        "name": "app3-nextaura-us",
        "url": "https://app3-nextaura-us.284w7l87aq94.us-south.codeengine.appdomain.cloud",
        "role": "atp-agent",
        "paths": {"health": "/health"},
    },
    {
        "id": "golias_live",
        "name": "golias-live",
        "url": "https://golias-live.2b02drai9gwy.us-east.codeengine.appdomain.cloud",
        "role": "golias-live-agent",
        "paths": {"health": "/health"},
    },
]


def load_registry() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.is_file():
        return DEFAULT_HOSTS
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
        return list(data.get("hosts") or DEFAULT_HOSTS)
    except Exception:
        return DEFAULT_HOSTS


def _fetch_url(url: str, timeout: float = 25.0, user_agent: str = "NextAura-AgentScraper/1.0") -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"Accept": "*/*", "User-Agent": user_agent},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(8192).decode("utf-8", errors="replace")
            ct = (resp.headers.get("content-type") or "").split(";")[0].strip()
            out: dict[str, Any] = {
                "http_status": resp.status,
                "content_type": ct,
                "bytes": len(raw),
            }
            if "json" in ct or raw.lstrip().startswith(("{", "[")):
                try:
                    out["json"] = json.loads(raw)
                except json.JSONDecodeError:
                    out["preview"] = raw[:200]
            else:
                out["preview"] = raw[:200]
            return out
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        return {"_error": f"HTTP {exc.code}", "http_status": exc.code, "_body": body[:200]}
    except Exception as exc:
        return {"_error": str(exc)}


def _fetch_json(url: str, timeout: float = 25.0, user_agent: str = "NextAura-AgentScraper/1.0") -> dict[str, Any]:
    data = _fetch_url(url, timeout=timeout, user_agent=user_agent)
    if "_error" in data:
        return data
    if "json" in data:
        return data["json"]
    return {"_error": "not_json", "http_status": data.get("http_status"), "content_type": data.get("content_type")}


def scrape_host(host: dict[str, Any], *, self_url: str | None = None) -> dict[str, Any]:
    base = str(host.get("url") or "").rstrip("/")
    paths: dict[str, str] = dict(host.get("paths") or {})
    root_path = paths.get("root", "/")
    probe_paths = {k: v for k, v in paths.items() if k != "root"}
    out: dict[str, Any] = {
        "id": host.get("id"),
        "name": host.get("name"),
        "url": base,
        "role": host.get("role"),
        "primary": bool(host.get("primary")),
        "live": False,
        "reachable": False,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": {},
        "errors": [],
    }
    if self_url and base.rstrip("/") == self_url.rstrip("/"):
        out["self"] = True

    root_data = _fetch_url(f"{base}{root_path}")
    out["endpoints"]["root"] = root_data
    if root_data.get("http_status") == 200:
        out["live"] = True
        out["reachable"] = True
    elif "_error" in root_data:
        out["errors"].append({"path": root_path, "error": root_data.get("_error")})

    for key, path in probe_paths.items():
        if not path:
            continue
        if key == "platform" and out.get("self"):
            continue
        if key in ("health", "agent", "status", "checkpoints", "infer_models", "benchmark", "drive_status", "models"):
            data = _fetch_json(f"{base}{path}")
        else:
            data = _fetch_url(f"{base}{path}")
        out["endpoints"][key] = data
        if "_error" not in data:
            out["reachable"] = True
            if key == "health":
                out["live"] = True
        else:
            out["errors"].append({"path": path, "error": data.get("_error")})

    return out


def extract_models_from_host(host_scrape: dict[str, Any]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    host_id = str(host_scrape.get("id") or "")
    host_name = str(host_scrape.get("name") or host_id)
    eps = host_scrape.get("endpoints") or {}

    for ckpt in (eps.get("checkpoints") or {}).get("checkpoints") or []:
        models.append(
            {
                "host": host_name,
                "host_id": host_id,
                "kind": "checkpoint",
                "name": ckpt.get("name"),
                "size_mb": ckpt.get("size_mb"),
                "loaded": ckpt.get("loaded"),
                "source": "local",
            }
        )
    for remote in (eps.get("checkpoints") or {}).get("remote_checkpoints") or []:
        models.append(
            {
                "host": host_name,
                "host_id": host_id,
                "kind": "checkpoint",
                "name": remote.get("name"),
                "size_mb": remote.get("size_mb"),
                "loaded": False,
                "source": "remote",
            }
        )
    agent_models = eps.get("models") or {}
    for row in agent_models.get("checkpoints") or []:
        models.append({**row, "host": host_name, "host_id": host_id})
    for row in agent_models.get("teachers") or []:
        models.append({**row, "host": host_name, "host_id": host_id})
    for row in agent_models.get("tokenizers") or []:
        models.append({**row, "host": host_name, "host_id": host_id})

    bench = eps.get("benchmark") or {}
    for row in bench.get("leaderboard") or []:
        models.append(
            {
                "host": host_name,
                "host_id": host_id,
                "kind": "leaderboard",
                "name": row.get("checkpoint") or row.get("name"),
                "mean_reward": row.get("mean_reward"),
                "step": row.get("step"),
            }
        )
    return models


def build_local_models_payload(
    *,
    host: str,
    checkpoints: list[dict[str, Any]],
    remote_checkpoints: list[dict[str, Any]],
    infer_catalog: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
    tokenizers: dict[str, str] | list[dict[str, Any]],
    loaded: str | None,
    loaded_step: int,
    recommended_base: str = "nextaura-51m-step3898.pt",
) -> dict[str, Any]:
    ckpt_rows = []
    for c in checkpoints:
        ckpt_rows.append(
            {
                "kind": "checkpoint",
                "name": c.get("name"),
                "size_mb": c.get("size_mb"),
                "loaded": c.get("loaded"),
                "source": "local",
                "host": host,
            }
        )
    for r in remote_checkpoints:
        ckpt_rows.append(
            {
                "kind": "checkpoint",
                "name": r.get("name"),
                "size_mb": r.get("size_mb"),
                "loaded": False,
                "source": "remote",
                "host": host,
            }
        )
    teacher_rows = []
    for t in teachers:
        teacher_rows.append(
            {
                "kind": "teacher",
                "topic": t.get("id") or t.get("topic"),
                "name": t.get("teacher") or t.get("label"),
                "repo": t.get("teacher_repo") or t.get("repo"),
                "host": host,
            }
        )
    tok_rows = []
    if isinstance(tokenizers, dict):
        for tid, label in tokenizers.items():
            tok_rows.append({"kind": "tokenizer", "id": tid, "label": label, "host": host})
    else:
        for t in tokenizers:
            tok_rows.append({"kind": "tokenizer", **t, "host": host})

    return {
        "host": host,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "loaded_checkpoint": loaded,
        "loaded_step": loaded_step,
        "recommended_pretrain_base": recommended_base,
        "checkpoints": ckpt_rows,
        "teachers": teacher_rows,
        "tokenizers": tok_rows,
        "infer_catalog": infer_catalog,
        "model_count": len(ckpt_rows) + len(teacher_rows) + len(tok_rows),
    }


def build_platform_index(*, self_url: str | None = None, local_models: dict[str, Any] | None = None) -> dict[str, Any]:
    hosts_cfg = load_registry()
    host_results = [scrape_host(h, self_url=self_url) for h in hosts_cfg]

    all_models: list[dict[str, Any]] = []
    for hr in host_results:
        all_models.extend(extract_models_from_host(hr))

    if local_models and self_url:
        for row in local_models.get("checkpoints") or []:
            all_models.append({**row, "host": self_url.replace("https://", "").split("/")[0], "host_id": "self"})
        for row in local_models.get("teachers") or []:
            all_models.append({**row, "host_id": "self"})

    # Dedupe checkpoint names per host
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for m in all_models:
        key = (str(m.get("host_id")), str(m.get("kind")), str(m.get("name")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    down = [h["name"] for h in host_results if not h.get("live")]
    index = {
        "schema_version": "2",
        "platform": "nextaura-agent-platform",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "scraper": "platform_index.py",
        "scope": "app1-app9.nextaura.us",
        "entrypoints": {
            "platform_index": f"{self_url.rstrip('/')}/api/agent/platform" if self_url else None,
            "models_index": f"{self_url.rstrip('/')}/api/agent/models" if self_url else None,
        },
        "hosts": host_results,
        "models": deduped,
        "model_count": len(deduped),
        "hosts_live": sum(1 for h in host_results if h.get("live")),
        "hosts_down": down,
        "hosts_reachable": sum(1 for h in host_results if h.get("reachable")),
        "hosts_total": len(host_results),
    }
    return index


def save_platform_index(index: dict[str, Any], path: Path | None = None) -> str:
    dest = path or CACHE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(dest)


def load_cached_platform_index() -> dict[str, Any] | None:
    if not CACHE_PATH.is_file():
        return None
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
