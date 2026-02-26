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
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
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


def _build_budget_info(total_budget: float, spent: float) -> dict:
    """Build budget dict for API, with MiniMax quota in subscription mode."""
    if total_budget > 0:
        return {
            "mode": "usage",
            "total": total_budget,
            "spent": round(spent, 4),
            "remaining": round(max(0, total_budget - spent), 4),
            "pct": round((spent / total_budget * 100), 1),
        }
    # Subscription mode
    info: dict = {"mode": "subscription", "spent": round(spent, 4)}
    try:
        from prometheus.llm import fetch_minimax_quota
        quota = fetch_minimax_quota()
        if quota:
            primary = os.environ.get("PROMETHEUS_MODEL", "MiniMax-M2.5")
            mq = quota.get(primary) or next(iter(quota.values()), None)
            if mq:
                info["calls_remaining"] = mq["remaining"]
                info["calls_total"] = mq["total"]
                info["calls_used"] = mq["used"]
                info["window_resets_in_min"] = mq["window_remaining_sec"] // 60
    except Exception:
        pass
    return info


# ----- API Routes -----

def _get_kimi_usage() -> Dict[str, Any]:
    """Get Kimi usage stats (shares process with launcher)."""
    try:
        from prometheus.llm import get_kimi_usage
        return get_kimi_usage()
    except Exception:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "window_remaining_sec": 0}


def _read_md(name: str) -> str:
    """Read a markdown file from memory directory."""
    path = DATA_DIR / "memory" / name
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


@app.get("/api/status")
def get_status():
    """Current agent state: budget, workers, session info, kimi usage, uptime."""
    st = _load_state()
    total_budget = float(os.environ.get("TOTAL_BUDGET", "0") or "0")
    spent = float(st.get("spent_usd") or 0.0)

    # Uptime from persisted launch_time_unix
    launch_ts = float(st.get("launch_time_unix") or time.time())
    uptime_sec = int(time.time() - launch_ts)

    return {
        "session_id": st.get("session_id"),
        "owner_id": st.get("owner_id"),
        "branch": st.get("current_branch"),
        "sha": st.get("current_sha"),
        "budget": _build_budget_info(total_budget, spent),
        "tokens": {
            "prompt": st.get("spent_tokens_prompt", 0),
            "completion": st.get("spent_tokens_completion", 0),
            "cached": st.get("spent_tokens_cached", 0),
        },
        "calls": st.get("spent_calls", 0),
        "evolution": {
            "enabled": bool(st.get("evolution_mode_enabled")),
            "cycle": st.get("evolution_cycle", 0),
            "consecutive_failures": st.get("evolution_consecutive_failures", 0),
        },
        "kimi": _get_kimi_usage(),
        "uptime_sec": uptime_sec,
        "dashboard_enabled": bool(st.get("dashboard_enabled")),
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


@app.get("/api/logs/progress")
def get_progress_logs(limit: int = 50):
    """Recent LLM progress/thinking messages."""
    return _tail_jsonl(DATA_DIR / "logs" / "progress.jsonl", max_lines=min(limit, 500))


@app.get("/api/memory/scratchpad")
def get_scratchpad():
    """Current scratchpad contents."""
    return {"content": _read_md("scratchpad.md")}


@app.get("/api/memory/identity")
def get_identity():
    """Current identity."""
    return {"content": _read_md("identity.md")}


@app.get("/api/memory/goals")
def get_goals():
    """Current goals."""
    return {"content": _read_md("goals.md")}


@app.get("/api/evolution/stats")
def get_evolution_stats(limit: int = 20):
    """Evolution cycle stats - recent cycles, success rate, trends."""
    events = _tail_jsonl(DATA_DIR / "logs" / "events.jsonl", max_lines=min(limit * 10, 1000))
    
    # Filter evolution task events
    ev_tasks = [e for e in events if e.get("type") in ("task_done", "task_start") and e.get("task_type") == "evolution"]
    
    # Build stats
    total = 0
    successes = 0
    failures = 0
    recent = []
    
    for e in ev_tasks:
        if e.get("type") == "task_start":
            total += 1
            recent.insert(0, {
                "task_id": e.get("task_id"),
                "status": "started",
                "ts": e.get("ts"),
            })
        elif e.get("type") == "task_done":
            status = e.get("status", "unknown")
            if status == "success":
                successes += 1
            elif status == "failed":
                failures += 1
            # Find matching start event
            for r in recent:
                if r.get("task_id") == e.get("task_id") and r.get("status") == "started":
                    r["status"] = status
                    r["duration_sec"] = e.get("duration_sec")
                    r["rounds"] = e.get("rounds", 0)
                    break
    
    # Calculate success rate
    completed = successes + failures
    success_rate = round((successes / completed * 100), 1) if completed > 0 else 0
    
    return {
        "total_cycles": total,
        "successes": successes,
        "failures": failures,
        "success_rate_pct": success_rate,
        "recent_cycles": recent[:limit],
    }


# ----- Group Management API -----

def _load_full_state() -> Dict[str, Any]:
    """Load full state.json for group config operations."""
    state_path = DATA_DIR / "state" / "state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_full_state(st: Dict[str, Any]) -> None:
    """Atomic write state.json."""
    state_path = DATA_DIR / "state" / "state.json"
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    import os as _os
    _os.replace(str(tmp), str(state_path))


@app.get("/api/groups")
def get_groups():
    """List all allowed groups with their config."""
    st = _load_full_state()
    allowed = list(st.get("allowed_groups", []))
    group_config = st.get("group_config", {})
    groups = []
    for gid in allowed:
        cfg = group_config.get(str(gid), {})
        groups.append({
            "id": gid,
            "config": {
                "policy": cfg.get("policy", "allowlist"),
                "require_mention": cfg.get("require_mention", True),
                "allowed_users": cfg.get("allowed_users", []),
                "history_limit": cfg.get("history_limit", 50),
            }
        })
    return {"groups": groups}


@app.post("/api/groups")
async def post_groups(request: Request):
    """Manage groups: add, remove, update config, add/remove users."""
    body = await request.json()
    action = body.get("action")
    gid = body.get("group_id")

    if not action:
        raise HTTPException(400, "Missing 'action'")

    st = _load_full_state()

    if action == "add":
        if not gid:
            raise HTTPException(400, "Missing 'group_id'")
        gid = int(gid)
        rt_groups = list(st.get("allowed_groups", []))
        if gid not in rt_groups:
            rt_groups.append(gid)
        st["allowed_groups"] = rt_groups
        gc = st.setdefault("group_config", {})
        if str(gid) not in gc:
            gc[str(gid)] = {"policy": "allowlist", "require_mention": True, "allowed_users": [], "history_limit": 50}
        _save_full_state(st)
        return {"ok": True, "message": f"Group {gid} added"}

    if action == "remove":
        if not gid:
            raise HTTPException(400, "Missing 'group_id'")
        gid = int(gid)
        st["allowed_groups"] = [g for g in st.get("allowed_groups", []) if int(g) != gid]
        st.get("group_config", {}).pop(str(gid), None)
        _save_full_state(st)
        return {"ok": True, "message": f"Group {gid} removed"}

    if action == "update":
        if not gid:
            raise HTTPException(400, "Missing 'group_id'")
        gid = int(gid)
        config = body.get("config", {})
        gc = st.setdefault("group_config", {}).setdefault(str(gid), {})
        for k in ("policy", "require_mention", "history_limit"):
            if k in config:
                gc[k] = config[k]
        _save_full_state(st)
        return {"ok": True, "message": f"Group {gid} config updated"}

    if action == "add_user":
        if not gid:
            raise HTTPException(400, "Missing 'group_id'")
        uid = body.get("user_id")
        if not uid:
            raise HTTPException(400, "Missing 'user_id'")
        gid, uid = int(gid), int(uid)
        gc = st.setdefault("group_config", {}).setdefault(str(gid), {})
        users = list(gc.get("allowed_users", []))
        if uid not in users:
            users.append(uid)
        gc["allowed_users"] = users
        _save_full_state(st)
        return {"ok": True, "message": f"User {uid} added to group {gid}"}

    if action == "remove_user":
        if not gid:
            raise HTTPException(400, "Missing 'group_id'")
        uid = body.get("user_id")
        if not uid:
            raise HTTPException(400, "Missing 'user_id'")
        gid, uid = int(gid), int(uid)
        gc = st.setdefault("group_config", {}).setdefault(str(gid), {})
        gc["allowed_users"] = [u for u in gc.get("allowed_users", []) if u != uid]
        _save_full_state(st)
        return {"ok": True, "message": f"User {uid} removed from group {gid}"}

    raise HTTPException(400, f"Unknown action: {action}")


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
