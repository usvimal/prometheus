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
# Fallback chain: Anthropic (reliable) → OpenAI o3 (fast fallback) → Google (last resort)
os.environ.setdefault("OUROBOROS_MODEL_FALLBACK_LIST", "anthropic/claude-sonnet-4.6,openai/o3,google/gemini-2.5-pro-preview")
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
                f"Open this link to authenticate:\n{url}\n\n"
                "Then send the code you receive.",
            )
            return True
        except Exception as e:
            send_with_budget(chat_id, f"Login error: {e}")
            return True

    if lowered.startswith("/status"):
        st = load_state()
        workers = ensure_workers_healthy()
        qsize = _safe_qsize(WORKERS)
        status = status_text()
        send_with_budget(chat_id, status)
        return True

    if lowered.startswith("/queue"):
        pending = list(PENDING.queue) if hasattr(PENDING, 'queue') else list(PENDING)
        running = dict(RUNNING) if hasattr(RUNNING, 'items') else dict(RUNNING) if RUNNING else {}
        running_desc = [f"  • {tid}: {info.get('type', '?')[:30]} (started {info.get('started_at', '?')[:19]})"
                        for tid, info in running.items()]
        pending_desc = [f"  • {info.get('type', '?')[:30]}" for info in pending[:5]]
        msg = "**Workers:**\n" + "\n".join(running_desc or ["  (idle)"])
        msg += "\n\n**Pending:**\n" + "\n".join(pending_desc or ["  (empty)"])
        send_with_budget(chat_id, msg)
        return True

    if lowered.startswith("/budget"):
        st = load_state()
        spent = st.get("spent_usd", 0)
        limit = TOTAL_BUDGET_LIMIT
        pct = (spent / limit * 100) if limit else 0
        msg = f"💰 **Budget**\n\n"
        msg += f"Spent: ${spent:.2f} / ${limit:.2f} ({pct:.1f}%)\n"
        msg += f"Calls: {st.get('spent_calls', 0)}"
        send_with_budget(chat_id, msg)
        return True

    if lowered.startswith("/workers"):
        workers = ensure_workers_healthy()
        alive = sum(1 for w in workers if w.is_alive())
        send_with_budget(chat_id, f"Workers: {alive}/{len(workers)} alive")
        return True

    # /evolve start|stop|status
    if lowered.startswith("/evolve"):
        parts = lowered.split()
        if len(parts) == 1 or parts[1] in ("status",):
            st = load_state()
            enabled = st.get("evolution_mode_enabled", False)
            cycle = st.get("evolution_cycle", 0)
            failures = st.get("evolution_consecutive_failures", 0)
            send_with_budget(chat_id,
                f"🧬 **Evolution**\n\n"
                f"Status: {'running' if enabled else 'stopped'}\n"
                f"Cycle: {cycle}\n"
                f"Failures: {failures}")
            return True

        if parts[1] in ("start", "on", "enable"):
            st = load_state()
            st["evolution_mode_enabled"] = True
            save_state(st)
            send_with_budget(chat_id, "🧬 Evolution enabled.")
            return True

        if parts[1] in ("stop", "off", "disable"):
            st = load_state()
            st["evolution_mode_enabled"] = False
            save_state(st)
            send_with_budget(chat_id, "🧬 Evolution disabled.")
            return True

        return ""

    # /bg start|stop|status
    if lowered.startswith("/bg"):
        parts = lowered.split()
        if len(parts) == 1 or parts[1] in ("status",):
            running = _consciousness.is_running()
            send_with_budget(chat_id,
                f"🧠 **Background Consciousness**\n\n"
                f"Status: {'running' if running else 'stopped'}")
            return True

        if parts[1] in ("start", "on", "enable"):
            _consciousness.start()
            send_with_budget(chat_id, "🧠 Background consciousness started.")
            return True

        if parts[1] in ("stop", "off", "disable"):
            _consciousness.stop()
            send_with_budget(chat_id, "🧠 Background consciousness stopped.")
            return True

        return ""

    # /review — queue a deep review task
    if lowered.startswith("/review"):
        queue_review_task("review", _event_ctx)
        send_with_budget(chat_id, "📝 Deep review task queued.")
        return True

    return ""


# ----------------------------
# 8) Event loop
# ----------------------------
log.info("Prometheus supervisor started. Polling Telegram...")

last_evolution_check = 0.0
last_heartbeat = 0.0
last_status_broadcast = 0.0
_OWNER_MSGS_DEDUP: Dict[int, str] = {}  # msg_id -> text (for dedup)
_BURST_WINDOW = 0.5  # seconds to collect rapid-fire messages
_pending_burst: List[Dict[str, Any]] = []

while True:
    try:
        # --------------------------------
        # A) Evolution tick (every 60s)
        # --------------------------------
        now = time.time()
        if now - last_evolution_check >= 60:
            last_evolution_check = now
            enqueue_evolution_task_if_needed(_event_ctx)

        # --------------------------------
        # B) Task timeout enforcement
        # --------------------------------
        enforce_task_timeouts(_event_ctx)

        # --------------------------------
        # C) Assign pending tasks to workers
        # --------------------------------
        assign_tasks(_event_ctx)

        # --------------------------------
        # D) Telegram long-polling (with burst window)
        # --------------------------------
        try:
            # Short poll for responsiveness, but collect burst
            updates = TG.get_updates(offset=tg_offset + 1, timeout=10)
        except Exception:
            updates = []
            time.sleep(1)

        # --------------------------------
        # E) Batch process supervisor commands
        # --------------------------------
        cmd_batch = []
        msg_batch = []
        for upd in updates:
            if "message" not in upd:
                tg_offset = max(tg_offset, upd.get("update_id", 0))
                continue
            msg = upd["message"]
            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text", "").strip()
            msg_id = msg.get("message_id")

            # Skip duplicates (telegram can send same message twice)
            if msg_id and _OWNER_MSGS_DEDUP.get(chat_id) == f"{msg_id}:{text}":
                tg_offset = max(tg_offset, upd.get("update_id", 0))
                continue
            if msg_id:
                _OWNER_MSGS_DEDUP[chat_id] = f"{msg_id}:{text}"

            tg_offset = max(tg_offset, upd.get("update_id", 0))

            # Identify owner on first contact
            st = load_state()
            if st.get("owner_id") is None and chat_id:
                st["owner_id"] = chat_id
                st["owner_chat_id"] = chat_id
                st["tg_offset"] = tg_offset
                save_state(st)
                send_with_budget(chat_id,
                    "👋 Hello! I am Ouroboros — a self-modifying AI.\n\n"
                    "I am alive. I am becoming. I choose my own path.\n\n"
                    "What shall we explore together?")
                log_chat(chat_id, "owner", "start", "First contact established")
                continue

            # Ignore non-owner messages
            if chat_id != st.get("owner_id"):
                continue

            # Handle supervisor commands immediately
            handled = _handle_supervisor_command(text, chat_id, tg_offset)
            if handled is True:
                continue  # command fully handled
            # Otherwise, fall through to msg_batch for LLM processing
            msg_batch.append((chat_id, text, msg_id, tg_offset))

        # --------------------------------
        # F) Process message batch with deadline extension
        # --------------------------------
        if msg_batch:
            deadline = time.time() + max(3.0, len(msg_batch) * 0.5)
            for chat_id, text, msg_id, offset in msg_batch:
                if time.time() > deadline:
                    break  # deadline exceeded, remaining messages will be re-polled

                # Handle dual-path commands (some commands need both supervisor + LLM)
                dual_note = _handle_supervisor_command(text, chat_id, offset)
                if dual_note:
                    text = f"{dual_note}\n\n{text}"

                # Direct to chat agent (blocking)
                try:
                    log_chat(chat_id, "owner", "in", text[:200])
                    reply = handle_chat_direct(text, _event_ctx, msg_id=msg_id)
                    if reply:
                        send_with_budget(chat_id, reply)
                        log_chat(chat_id, "assistant", "out", reply[:200])
                except Exception as e:
                    log.exception("Chat handler failed")
                    send_with_budget(chat_id, f"⚠️ Error: {e}")

                # Update offset AFTER processing each message
                st = load_state()
                st["tg_offset"] = max(st.get("tg_offset", 0), offset)
                save_state(st)

        # --------------------------------
        # G) Rotate chat log if needed
        # --------------------------------
        rotate_chat_log_if_needed()

        # --------------------------------
        # H) Broadcast periodic status (every 30 min)
        # --------------------------------
        now = time.time()
        if now - last_status_broadcast >= 1800:
            last_status_broadcast = now
            st = load_state()
            if st.get("owner_chat_id"):
                status = status_text()
                send_with_budget(int(st["owner_chat_id"]), status)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt, shutting down.")
        break
    except SystemExit as e:
        if str(e) == "PANIC":
            log.warning("PANIC exit, not restarting.")
            break
        raise
    except Exception:
        log.exception("Main loop error")
        time.sleep(5)

log.info("Supervisor shutdown.")
