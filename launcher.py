# ============================
# Prometheus — VPS Runtime launcher (entry point)
# ============================
# Thin orchestrator: secrets, bootstrap, main loop.
# Heavy logic lives in supervisor/ package.
# Reads config from env vars or ~/prometheus/config.env

import logging
import os
import sys
import json
import time
import uuid
import pathlib
import subprocess
import datetime
import threading
import queue as _queue_mod
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ----------------------------
# 0) Load config from env file if present
# ----------------------------
_CONFIG_FILE = pathlib.Path.home() / "prometheus" / "config.env"
if _CONFIG_FILE.exists():
    log.info("Loading config from %s", _CONFIG_FILE)
    for line in _CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# ----------------------------
# 0.1) provide apply_patch shim
# ----------------------------
from prometheus.apply_patch import install as install_apply_patch
from prometheus.llm import DEFAULT_LIGHT_MODEL
install_apply_patch()


# ----------------------------
# 1) Secrets + runtime config (from env vars)
# ----------------------------
def get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        v = default
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required config: {name}"
    return v


def get_cfg(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return v
    return default


def _parse_int_cfg(raw: Optional[str], default: int, minimum: int = 0) -> int:
    try:
        val = int(str(raw))
    except Exception:
        val = default
    return max(minimum, val)


# Required secrets
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", default="")
TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", required=True)
GITHUB_TOKEN = get_secret("GITHUB_TOKEN", required=True)

# Budget
import re
_raw_budget = str(get_secret("TOTAL_BUDGET", default="0") or "")
_clean_budget = re.sub(r'[^0-9.\-]', '', _raw_budget)
TOTAL_BUDGET_LIMIT = float(_clean_budget) if _clean_budget else 0.0

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", default="")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", default="")
GITHUB_USER = get_cfg("GITHUB_USER")
GITHUB_REPO = get_cfg("GITHUB_REPO")
assert GITHUB_USER and str(GITHUB_USER).strip(), "GITHUB_USER not set in config."
assert GITHUB_REPO and str(GITHUB_REPO).strip(), "GITHUB_REPO not set in config."

MAX_WORKERS = int(get_cfg("OUROBOROS_MAX_WORKERS", default="3") or "3")
MODEL_MAIN = get_cfg("OUROBOROS_MODEL", default="codex-mini")
MODEL_CODE = get_cfg("OUROBOROS_MODEL_CODE", default="codex-mini")
MODEL_LIGHT = get_cfg("OUROBOROS_MODEL_LIGHT", default=DEFAULT_LIGHT_MODEL)

BUDGET_REPORT_EVERY_MESSAGES = 10
SOFT_TIMEOUT_SEC = max(60, int(get_cfg("OUROBOROS_SOFT_TIMEOUT_SEC", default="600") or "600"))
HARD_TIMEOUT_SEC = max(120, int(get_cfg("OUROBOROS_HARD_TIMEOUT_SEC", default="1800") or "1800"))
DIAG_HEARTBEAT_SEC = _parse_int_cfg(get_cfg("OUROBOROS_DIAG_HEARTBEAT_SEC", default="30"), default=30, minimum=0)
DIAG_SLOW_CYCLE_SEC = _parse_int_cfg(get_cfg("OUROBOROS_DIAG_SLOW_CYCLE_SEC", default="20"), default=20, minimum=0)

# Export to env for submodules that read from env
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = str(OPENROUTER_API_KEY)
os.environ.setdefault("OPENAI_API_KEY", str(OPENAI_API_KEY or ""))
os.environ.setdefault("ANTHROPIC_API_KEY", str(ANTHROPIC_API_KEY or ""))
os.environ["GITHUB_USER"] = str(GITHUB_USER)
os.environ["GITHUB_REPO"] = str(GITHUB_REPO)
os.environ["OUROBOROS_MODEL"] = str(MODEL_MAIN or "codex-mini")
os.environ["OUROBOROS_MODEL_CODE"] = str(MODEL_CODE or "codex-mini")
if MODEL_LIGHT:
    os.environ["OUROBOROS_MODEL_LIGHT"] = str(MODEL_LIGHT)
os.environ.setdefault("OUROBOROS_MODEL_FALLBACK_LIST", "anthropic/claude-sonnet-4.6,google/gemini-2.5-pro-preview,openai/o3")
os.environ["OUROBOROS_DIAG_HEARTBEAT_SEC"] = str(DIAG_HEARTBEAT_SEC)
os.environ["OUROBOROS_DIAG_SLOW_CYCLE_SEC"] = str(DIAG_SLOW_CYCLE_SEC)
os.environ["TELEGRAM_BOT_TOKEN"] = str(TELEGRAM_BOT_TOKEN)

# ----------------------------
# 2) Filesystem paths (VPS local)
# ----------------------------
DRIVE_ROOT = pathlib.Path(
    os.environ.get("PROMETHEUS_DATA_DIR", str(pathlib.Path.home() / "prometheus" / "data"))
).resolve()
REPO_DIR = pathlib.Path(
    os.environ.get("PROMETHEUS_REPO_DIR", str(pathlib.Path.home() / "prometheus" / "repo"))
).resolve()

for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
REPO_DIR.mkdir(parents=True, exist_ok=True)

# Clear stale owner mailbox files from previous session
try:
    from prometheus.owner_inject import get_pending_path
    _stale_inject = get_pending_path(DRIVE_ROOT)
    if _stale_inject.exists():
        _stale_inject.unlink(missing_ok=True)
    _mailbox_dir = DRIVE_ROOT / "memory" / "owner_mailbox"
    if _mailbox_dir.exists():
        for _f in _mailbox_dir.iterdir():
            _f.unlink(missing_ok=True)
except Exception:
    pass

CHAT_LOG_PATH = DRIVE_ROOT / "logs" / "chat.jsonl"
if not CHAT_LOG_PATH.exists():
    CHAT_LOG_PATH.write_text("", encoding="utf-8")

# ----------------------------
# 3) Git constants
# ----------------------------
BRANCH_DEV = get_cfg("PROMETHEUS_BRANCH_DEV", default="main")
BRANCH_STABLE = get_cfg("PROMETHEUS_BRANCH_STABLE", default="main")
REMOTE_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

# ----------------------------
# 4) Initialize supervisor modules
# ----------------------------
from supervisor.state import (
    init as state_init, load_state, save_state, append_jsonl,
    update_budget_from_usage, status_text, rotate_chat_log_if_needed,
    init_state,
)
state_init(DRIVE_ROOT, TOTAL_BUDGET_LIMIT)
init_state()

from supervisor.telegram import (
    init as telegram_init, TelegramClient, send_with_budget, log_chat,
)
TG = TelegramClient(str(TELEGRAM_BOT_TOKEN))
telegram_init(
    drive_root=DRIVE_ROOT,
    total_budget_limit=TOTAL_BUDGET_LIMIT,
    budget_report_every=BUDGET_REPORT_EVERY_MESSAGES,
    tg_client=TG,
)

from supervisor.git_ops import (
    init as git_ops_init, ensure_repo_present, checkout_and_reset,
    sync_runtime_dependencies, import_test, safe_restart,
)
git_ops_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, remote_url=REMOTE_URL,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.queue import (
    enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
    persist_queue_snapshot, restore_pending_from_snapshot,
    cancel_task_by_id, queue_review_task, sort_pending,
)

from supervisor.workers import (
    init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
    spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
    handle_chat_direct, _get_chat_agent, auto_resume_after_restart,
)
workers_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, max_workers=MAX_WORKERS,
    soft_timeout=SOFT_TIMEOUT_SEC, hard_timeout=HARD_TIMEOUT_SEC,
    total_budget_limit=TOTAL_BUDGET_LIMIT,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.events import dispatch_event

# ----------------------------
# 5) Bootstrap repo
# ----------------------------
ensure_repo_present()
ok, msg = safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
assert ok, f"Bootstrap failed: {msg}"

# ----------------------------
# 6) Start workers
# ----------------------------
kill_workers()
spawn_workers(MAX_WORKERS)
restored_pending = restore_pending_from_snapshot()
persist_queue_snapshot(reason="startup")
if restored_pending > 0:
    st_boot = load_state()
    if st_boot.get("owner_chat_id"):
        send_with_budget(int(st_boot["owner_chat_id"]),
                         f"Restored pending queue from snapshot: {restored_pending} tasks.")

append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "type": "launcher_start",
    "branch": load_state().get("current_branch"),
    "sha": load_state().get("current_sha"),
    "max_workers": MAX_WORKERS,
    "model_default": MODEL_MAIN, "model_code": MODEL_CODE, "model_light": MODEL_LIGHT,
    "soft_timeout_sec": SOFT_TIMEOUT_SEC, "hard_timeout_sec": HARD_TIMEOUT_SEC,
    "worker_start_method": str(os.environ.get("OUROBOROS_WORKER_START_METHOD") or ""),
    "diag_heartbeat_sec": DIAG_HEARTBEAT_SEC,
    "diag_slow_cycle_sec": DIAG_SLOW_CYCLE_SEC,
})

# ----------------------------
# 6.1) Auto-resume after restart
# ----------------------------
auto_resume_after_restart()

# ----------------------------
# 6.2) Direct-mode watchdog
# ----------------------------
def _chat_watchdog_loop():
    """Monitor direct-mode chat agent for hangs. Runs as daemon thread."""
    soft_warned = False
    while True:
        time.sleep(30)
        try:
            agent = _get_chat_agent()
            if not agent._busy:
                soft_warned = False
                continue

            now = time.time()
            idle_sec = now - agent._last_progress_ts
            total_sec = now - agent._task_started_ts

            if idle_sec >= HARD_TIMEOUT_SEC:
                st = load_state()
                if st.get("owner_chat_id"):
                    send_with_budget(
                        int(st["owner_chat_id"]),
                        f"Task stuck ({int(total_sec)}s without progress). Restarting agent.",
                    )
                reset_chat_agent()
                soft_warned = False
                continue

            if idle_sec >= SOFT_TIMEOUT_SEC and not soft_warned:
                soft_warned = True
                st = load_state()
                if st.get("owner_chat_id"):
                    send_with_budget(
                        int(st["owner_chat_id"]),
                        f"Task running for {int(total_sec)}s, "
                        f"last progress {int(idle_sec)}s ago. Continuing.",
                    )
        except Exception:
            log.debug("Failed to check/notify chat watchdog", exc_info=True)
            pass

_watchdog_thread = threading.Thread(target=_chat_watchdog_loop, daemon=True)
_watchdog_thread.start()

# ----------------------------
# 6.3) Background consciousness
# ----------------------------
from prometheus.consciousness import BackgroundConsciousness


def _get_owner_chat_id() -> Optional[int]:
    try:
        st = load_state()
        cid = st.get("owner_chat_id")
        return int(cid) if cid else None
    except Exception:
        return None


_consciousness = BackgroundConsciousness(
    drive_root=DRIVE_ROOT,
    repo_dir=REPO_DIR,
    event_queue=get_event_q(),
    owner_chat_id_fn=_get_owner_chat_id,
)


def reset_chat_agent():
    """Reset the direct-mode chat agent (called by watchdog on hangs)."""
    import supervisor.workers as _w
    _w._chat_agent = None


# ----------------------------
# 6.4) Codex OAuth state (for /login command)
# ----------------------------
_pending_oauth: Dict[str, Any] = {}


# ----------------------------
# 7) Main loop
# ----------------------------
import types
_event_ctx = types.SimpleNamespace(
    DRIVE_ROOT=DRIVE_ROOT,
    REPO_DIR=REPO_DIR,
    BRANCH_DEV=BRANCH_DEV,
    BRANCH_STABLE=BRANCH_STABLE,
    TG=TG,
    WORKERS=WORKERS,
    PENDING=PENDING,
    RUNNING=RUNNING,
    MAX_WORKERS=MAX_WORKERS,
    send_with_budget=send_with_budget,
    load_state=load_state,
    save_state=save_state,
    update_budget_from_usage=update_budget_from_usage,
    append_jsonl=append_jsonl,
    enqueue_task=enqueue_task,
    cancel_task_by_id=cancel_task_by_id,
    queue_review_task=queue_review_task,
    persist_queue_snapshot=persist_queue_snapshot,
    safe_restart=safe_restart,
    kill_workers=kill_workers,
    spawn_workers=spawn_workers,
    sort_pending=sort_pending,
    consciousness=_consciousness,
)


def _safe_qsize(q: Any) -> int:
    try:
        return int(q.qsize())
    except Exception:
        return -1


# Rate limiter for read-only supervisor commands (prevent /queue flood)
_CMD_RATE_LIMIT: Dict[str, float] = {}  # cmd -> last_processed_ts
_CMD_RATE_LIMIT_SEC = 2.0  # min seconds between identical commands


def _handle_supervisor_command(text: str, chat_id: int, tg_offset: int = 0):
    """Handle supervisor slash-commands.

    Returns:
        True  — terminal command fully handled (caller should `continue`)
        str   — dual-path note to prepend (caller falls through to LLM)
        ""    — not a recognized command (falsy, caller falls through)
    """
    lowered = text.strip().lower()

    # Rate-limit read-only commands to prevent flood
    rate_limited_cmds = ("/queue", "/status", "/budget", "/workers", "/evolve")
    for prefix in rate_limited_cmds:
        if lowered.startswith(prefix) and " " not in lowered.strip().strip("/"):
            now = time.time()
            last = _CMD_RATE_LIMIT.get(prefix, 0)
            if (now - last) < _CMD_RATE_LIMIT_SEC:
                return True  # silently skip duplicate
            _CMD_RATE_LIMIT[prefix] = now
            break

    if lowered.startswith("/panic"):
        send_with_budget(chat_id, "PANIC: stopping everything now.")
        kill_workers()
        st2 = load_state()
        st2["tg_offset"] = tg_offset
        save_state(st2)
        raise SystemExit("PANIC")

    if lowered.startswith("/restart"):
        st2 = load_state()
        st2["session_id"] = uuid.uuid4().hex
        st2["tg_offset"] = tg_offset
        save_state(st2)
        send_with_budget(chat_id, "Restarting (soft).")
        ok, msg = safe_restart(reason="owner_restart", unsynced_policy="rescue_and_reset")
        if not ok:
            send_with_budget(chat_id, f"Restart cancelled: {msg}")
            return True
        kill_workers()
        os.execv(sys.executable, [sys.executable, __file__])

    # /login — Codex OAuth flow via Telegram
    if lowered.startswith("/login"):
        try:
            from prometheus.codex_auth import get_login_url, exchange_code_for_tokens, save_auth
            url, verifier, state = get_login_url()
            _pending_oauth["code_verifier"] = verifier
            _pending_oauth["state"] = state
            send_with_budget(chat_id,
                f"Open this URL to authenticate with ChatGPT:\n\n{url}\n\n"
                "After logging in, you'll be redirected. Copy the full redirect URL "
                "and send it here with /callback <url>")
        except Exception as e:
            send_with_budget(chat_id, f"Login failed: {e}")
        return True

    # /callback — Complete OAuth flow
    if lowered.startswith("/callback"):
        try:
            from prometheus.codex_auth import exchange_code_for_tokens, save_auth
            from urllib.parse import urlparse, parse_qs
            parts = text.strip().split(maxsplit=1)
            if len(parts) < 2:
                send_with_budget(chat_id, "Usage: /callback <redirect_url>")
                return True
            callback_url = parts[1].strip()
            parsed = urlparse(callback_url)
            query = parse_qs(parsed.query)
            code = query.get("code", [None])[0]
            state = query.get("state", [None])[0]
            if not code:
                send_with_budget(chat_id, "No code in callback URL.")
                return True
            if state != _pending_oauth.get("state"):
                send_with_budget(chat_id, "State mismatch, possible CSRF.")
                return True
            tokens = exchange_code_for_tokens(code, _pending_oauth["code_verifier"])
            save_auth(tokens)
            send_with_budget(chat_id, "Authenticated! You can now use ChatGPT features.")
            _pending_oauth.clear()
        except Exception as e:
            send_with_budget(chat_id, f"Callback failed: {e}")
        return True

    if lowered.startswith("/status"):
        from supervisor.state import status_text
        send_with_budget(chat_id, status_text(WORKERS, PENDING, RUNNING, SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC))
        return True

    if lowered.startswith("/budget"):
        st = load_state()
        remaining = st.get("budget_remaining", 0.0)
        total = TOTAL_BUDGET_LIMIT
        used = total - remaining
        send_with_budget(chat_id, f"Budget: ${used:.2f} used / ${total:.2f} total")
        return True

    if lowered.startswith("/workers"):
        st = load_state()
        w_ids = list(WORKERS.keys())
        send_with_budget(chat_id, f"Workers: {len(w_ids)} active, {_safe_qsize(PENDING)} pending.")
        return True

    if lowered.startswith("/queue"):
        pending = list(PENDING) if hasattr(PENDING, '__iter__') else []
        running = dict(RUNNING) if hasattr(RUNNING, '__iter__') else {}
        parts = []
        if running:
            r_lines = []
            for tid, info in list(running.items())[:5]:
                ttype = info.get("task_type", info.get("type", "?"))
                desc = str(info.get("text", ""))[:40]
                r_lines.append(f"  {tid[:8]} ({ttype}) {desc}")
            parts.append("Running:\n" + "\n".join(r_lines))
        if pending:
            p_lines = [f"  {i+1}. {t.get('text', '?')[:50]}" for i, t in enumerate(pending[:10])]
            parts.append("Pending:\n" + "\n".join(p_lines))
        if not parts:
            send_with_budget(chat_id, "Queue empty, nothing running.")
        else:
            send_with_budget(chat_id, "\n\n".join(parts))
        return True

    if lowered.startswith("/kill"):
        parts = text.strip().split()
        if len(parts) < 2:
            send_with_budget(chat_id, "Usage: /kill <task_id>")
            return True
        task_id = parts[1]
        ok = cancel_task_by_id(task_id)
        send_with_budget(chat_id, f"Killed {task_id}" if ok else f"Task {task_id} not found.")
        return True

    if lowered.startswith("/evolve"):
        if "start" in lowered:
            st = load_state()
            st["evolution_mode_enabled"] = True
            st["evolution_consecutive_failures"] = 0
            st.pop("evolution_paused", None)
            save_state(st)
            enqueue_evolution_task_if_needed()
            send_with_budget(chat_id, "🧬 Evolution enabled.")
        elif "stop" in lowered:
            st = load_state()
            st["evolution_mode_enabled"] = False
            st["evolution_paused"] = True
            save_state(st)
            send_with_budget(chat_id, "🧬 Evolution paused.")
        else:
            st = load_state()
            enabled = bool(st.get("evolution_mode_enabled"))
            failures = int(st.get("evolution_consecutive_failures") or 0)
            cycle = int(st.get("evolution_cycle") or 0)
            status = "running" if enabled else "paused"
            send_with_budget(chat_id, f"🧬 Evolution: {status} (cycle {cycle}, failures: {failures})")
        return True

    if lowered.startswith("/bg"):
        if "start" in lowered:
            _consciousness.start()
            send_with_budget(chat_id, "Background consciousness started.")
        elif "stop" in lowered:
            _consciousness.stop()
            send_with_budget(chat_id, "Background consciousness stopped.")
        else:
            send_with_budget(chat_id, f"Background: {'running' if _consciousness.running else 'stopped'}")
        return True

    if lowered.startswith("/review"):
        queue_review_task()
        send_with_budget(chat_id, "Review queued.")
        return True

    return ""


_last_message_ts = time.time()


def handle_one_update(offset: int) -> int:
    """Fetch and process one Telegram update.
    Returns the next offset to use.
    """
    global _last_message_ts

    # Blocking get with timeout
    updates = TG.get_updates(timeout=60, offset=offset)
    if not updates:
        return offset

    for upd in updates:
        msg = upd.get("message") or upd.get("edited_message") or {}
        # Always advance offset first so we never re-process this update
        offset = int(upd.get("update_id", 0)) + 1
        if not msg:
            continue

        chat_id = int(msg["chat"]["id"])
        chat_type = msg.get("chat", {}).get("type", "private")
        from_user = msg.get("from") or {}
        user_id = int(from_user.get("id") or 0)
        text = str(msg.get("text") or "")
        caption = str(msg.get("caption") or "")
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Extract image if present
        image_data = None
        if msg.get("photo"):
            best_photo = msg["photo"][-1]
            file_id = best_photo.get("file_id")
            if file_id:
                b64, mime = TG.download_file_base64(file_id)
                if b64:
                    image_data = (b64, mime, caption)
        elif msg.get("document"):
            doc = msg["document"]
            mime_type = str(doc.get("mime_type") or "")
            if mime_type.startswith("image/"):
                file_id = doc.get("file_id")
                if file_id:
                    b64, mime = TG.download_file_base64(file_id)
                    if b64:
                        image_data = (b64, mime, caption)

        st = load_state()
        
        # Only register owner in private chats (not group chats)
        if st.get("owner_id") is None:
            if chat_type == "private":
                st["owner_id"] = user_id
                st["owner_chat_id"] = chat_id
                st["last_owner_message_at"] = now_iso
                save_state(st)
                log_chat("in", chat_id, user_id, text)
                send_with_budget(chat_id, "Owner registered. Prometheus online.")
                continue
            else:
                # In group chat without owner registered - skip
                continue

        # Only respond to owner in private chats
        if user_id != int(st.get("owner_id")):
            continue
        
        # Skip if not private chat (owner messages in groups ignored)
        if chat_type != "private":
            continue

        log_chat("in", chat_id, user_id, text)
        st["last_owner_message_at"] = now_iso
        _last_message_ts = time.time()
        save_state(st)

        # --- Supervisor commands ---
        if text.strip().lower().startswith("/"):
            try:
                result = _handle_supervisor_command(text, chat_id, tg_offset=offset)
                if result is True:
                    continue
                elif result:
                    text = result + text
            except SystemExit:
                raise
            except Exception:
                # Catch ALL errors in command handlers so they never prevent
                # offset advancement (which would cause infinite re-processing
                # of the same Telegram update — the /queue flood bug).
                log.exception("Error handling supervisor command: %s", text[:50])
                try:
                    send_with_budget(chat_id, f"Command error: {text[:30]}. Check logs.")
                except Exception:
                    pass
                continue  # Treat as handled — advance past this update

        # --- Dispatch to worker pool ---
        # image_data = None  # FIXME: wire this through if desired
        enqueue_task({
            "id": uuid.uuid4().hex[:8],
            "type": "task",
            "chat_id": chat_id,
            "text": text,
            "image_data": image_data,
            "caption": caption,
            "from_user": from_user,
            "_is_direct_chat": True,
        })
        assign_tasks()


    return offset


# ----------------------------
# 8) Run the main loop
# ----------------------------
st = load_state()
offset = int(st.get("tg_offset", 0))
log.info("Starting main loop at offset %s", offset)

# Main event loop
while True:
    try:
        # 1) process Telegram updates
        try:
            new_offset = handle_one_update(offset)
            if new_offset != offset:
                offset = new_offset
                _st = load_state()
                _st["tg_offset"] = offset
                save_state(_st)
        except Exception:
            log.exception("Error in handle_one_update")

        # 1.5) drain worker events (send_message, task_done, etc.)
        try:
            event_q = get_event_q()
            while True:
                try:
                    evt = event_q.get_nowait()
                except _queue_mod.Empty:
                    break
                dispatch_event(evt, _event_ctx)
        except Exception:
            log.exception("Error draining event queue")

        # 2) tick workers
        try:
            assign_tasks()
        except Exception:
            log.exception("Error in assign_tasks")

        # 3) enforce task timeouts
        try:
            enforce_task_timeouts()
        except Exception:
            log.exception("Error in enforce_task_timeouts")

        # 4) tick consciousness
        try:
            pass  # consciousness runs in background thread
        except Exception:
            log.exception("Consciousness check (no-op)")

        # 5) check for scheduled/pushed tasks
        try:
            pass  # evolution tasks enqueued below
            enqueue_evolution_task_if_needed()
        except Exception:
            log.exception("Error in enqueue_evolution_task_if_needed")

        # 6) emit heartbeat
        try:
            if DIAG_HEARTBEAT_SEC > 0 and (time.time() - _last_message_ts) > DIAG_HEARTBEAT_SEC:
                st = load_state()
                if st.get("owner_chat_id"):
                    append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "type": "heartbeat",
                        "workers": len(WORKERS),
                        "pending": _safe_qsize(PENDING),
                        "running": len(RUNNING),
                        "consciousness": _consciousness.running,
                    })
                    _last_message_ts = time.time()
        except Exception:
            log.debug("Heartbeat failed", exc_info=True)

        time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Interrupted, saving state.")
        break
    except SystemExit:
        log.info("SystemExit, saving state.")
        break
    except Exception:
        log.exception("Fatal error in main loop")
        try:
            st = load_state()
            if st.get("owner_chat_id"):
                send_with_budget(int(st["owner_chat_id"]),
                    f"Fatal error in main loop: {traceback.format_exc()}")
        except Exception:
            pass
        break

# ----------------------------
# 9) Cleanup
# ----------------------------
st = load_state()
st["tg_offset"] = offset
save_state(st)

# Snapshot queue before exit
persist_queue_snapshot(reason="shutdown")

log.info("Prometheus launcher exiting.")
