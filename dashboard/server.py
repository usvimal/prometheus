"""
Prometheus — Dashboard API server.

Exposes agent state, logs, and metrics via FastAPI.
Runs alongside the main launcher on a separate port.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

app = FastAPI(title="Prometheus Dashboard", version="0.1.0")

# Data directory (same as launcher's DRIVE_ROOT)
DATA_DIR = pathlib.Path(
    os.environ.get("PROMETHEUS_DATA_DIR", str(pathlib.Path.home() / "prometheus" / "data"))
).resolve()


def _load_state() -> Dict[str, Any]:
    state_path = DATA_DIR / "state" / "state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_queue_snapshot() -> List[Dict[str, Any]]:
    path = DATA_DIR / "state" / "queue_snapshot.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("pending", [])
    except (json.JSONDecodeError, OSError):
        return []


def _tail_jsonl(path: pathlib.Path, max_lines: int = 100) -> List[Dict[str, Any]]:
    """Read last N lines from a JSONL file."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        result = []
        for line in lines[-max_lines:]:
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return result
    except OSError:
        return []


# ----- API Routes -----

@app.get("/api/status")
def get_status():
    """Current agent state: budget, workers, session info."""
    st = _load_state()
    total_budget = float(os.environ.get("TOTAL_BUDGET", "0") or "0")
    spent = float(st.get("spent_usd") or 0.0)
    return {
        "session_id": st.get("session_id"),
        "owner_id": st.get("owner_id"),
        "branch": st.get("current_branch"),
        "sha": st.get("current_sha"),
        "budget": {
            "total": total_budget,
            "spent": round(spent, 4),
            "remaining": round(max(0, total_budget - spent), 4),
            "pct": round((spent / total_budget * 100) if total_budget > 0 else 0, 1),
        },
        "tokens": {
            "prompt": st.get("spent_tokens_prompt", 0),
            "completion": st.get("spent_tokens_completion", 0),
            "cached": st.get("spent_tokens_cached", 0),
        },
        "calls": st.get("spent_calls", 0),
        "evolution": {
            "enabled": bool(st.get("evolution_mode_enabled")),
            "cycle": st.get("evolution_cycle", 0),
        },
        "last_owner_message": st.get("last_owner_message_at"),
    }


@app.get("/api/queue")
def get_queue():
    """Current task queue."""
    return _load_queue_snapshot()


@app.get("/api/logs/supervisor")
def get_supervisor_logs(limit: int = 50):
    """Recent supervisor log entries."""
    return _tail_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", max_lines=min(limit, 500))


@app.get("/api/logs/events")
def get_event_logs(limit: int = 50):
    """Recent event log entries."""
    return _tail_jsonl(DATA_DIR / "logs" / "events.jsonl", max_lines=min(limit, 500))


@app.get("/api/logs/chat")
def get_chat_logs(limit: int = 50):
    """Recent chat log entries."""
    return _tail_jsonl(DATA_DIR / "logs" / "chat.jsonl", max_lines=min(limit, 500))


@app.get("/api/codex/status")
def get_codex_status():
    """Codex OAuth authentication status."""
    auth_path = pathlib.Path.home() / "prometheus" / "data" / "auth.json"
    if not auth_path.exists():
        return {"authenticated": False}
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
        tokens = data.get("tokens", {})
        return {
            "authenticated": bool(tokens.get("access_token")),
            "auth_mode": data.get("auth_mode"),
            "last_refresh": data.get("last_refresh"),
            "has_refresh_token": bool(tokens.get("refresh_token")),
        }
    except (json.JSONDecodeError, OSError):
        return {"authenticated": False}


# ----- Static files / Dashboard UI -----

DASHBOARD_DIR = pathlib.Path(__file__).parent / "static"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the dashboard HTML."""
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Prometheus Dashboard</h1><p>Static files not found.</p>")


def run_dashboard(host: str = "127.0.0.1", port: int = 8080):
    """Run the dashboard server standalone. Binds to localhost by default (use Caddy for external access)."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
