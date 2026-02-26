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

_LAUNCH_TIME = time.time()

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
            # Strip inline comments (e.g. "MiniMax-M2.5  # fallback")
            if "  #" in value:
                value = value.split("  #")[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value

# ----------------------------
# 0.1) provide apply_patch shim
# ----------------------------
from prometheus.apply_patch import install as install_apply_patch
from prometheus.llm import DEFAULT_LIGHT_MODEL, get_kimi_usage
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


# --- Env var migration: PROMETHEUS_* → PROMETHEUS_* ---
# Backward compat: if old names exist but new ones don't, copy them over.
_ENV_PREFIX_OLD = "PROMETHEUS_"
_ENV_PREFIX_NEW = "PROMETHEUS_"
for _key in list(os.environ):
    if _key.startswith(_ENV_PREFIX_OLD):
        _new_key = _ENV_PREFIX_NEW + _key[len(_ENV_PREFIX_OLD):]
        if _new_key not in os.environ:
            os.environ[_new_key] = os.environ[_key]
# --- End migration ---

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

MAX_WORKERS = int(get_cfg("PROMETHEUS_MAX_WORKERS", default="3") or "3")
MODEL_MAIN = get_cfg("PROMETHEUS_MODEL", default="codex-mini")
MODEL_CODE = get_cfg("PROMETHEUS_MODEL_CODE", default="codex-mini")
MODEL_LIGHT = get_cfg("PROMETHEUS_MODEL_LIGHT", default=DEFAULT_LIGHT_MODEL)

BUDGET_REPORT_EVERY_MESSAGES = 10
SOFT_TIMEOUT_SEC = max(60, int(get_cfg("PROMETHEUS_SOFT_TIMEOUT_SEC", default="600") or "600"))
HARD_TIMEOUT_SEC = max(120, int(get_cfg("PROMETHEUS_HARD_TIMEOUT_SEC", default="1800") or "1800"))
DIAG_HEARTBEAT_SEC = _parse_int_cfg(get_cfg("PROMETHEUS_DIAG_HEARTBEAT_SEC", default="30"), default=30, minimum=0)
DIAG_SLOW_CYCLE_SEC = _parse_int_cfg(get_cfg("PROMETHEUS_DIAG_SLOW_CYCLE_SEC", default="20"), default=20, minimum=0)

# Export to env for submodules that read from env
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = str(OPENROUTER_API_KEY)
os.environ.setdefault("OPENAI_API_KEY", str(OPENAI_API_KEY or ""))
os.environ.setdefault("ANTHROPIC_API_KEY", str(ANTHROPIC_API_KEY or ""))
os.environ["GITHUB_USER"] = str(GITHUB_USER)
os.environ["GITHUB_REPO"] = str(GITHUB_REPO)
os.environ["PROMETHEUS_MODEL"] = str(MODEL_MAIN or "codex-mini")
os.environ["PROMETHEUS_MODEL_CODE"] = str(MODEL_CODE or "codex-mini")
if MODEL_LIGHT:
    os.environ["PROMETHEUS_MODEL_LIGHT"] = str(MODEL_LIGHT)
os.environ["PROMETHEUS_DIAG_HEARTBEAT_SEC"] = str(DIAG_HEARTBEAT_SEC)
os.environ["PROMETHEUS_DIAG_SLOW_CYCLE_SEC"] = str(DIAG_SLOW_CYCLE_SEC)
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
    "worker_start_method": str(os.environ.get("PROMETHEUS_WORKER_START_METHOD") or ""),
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
_pending_events: List[Dict[str, Any]] = []  # For events generated by event handlers
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
    pending_events=_pending_events,
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
                "After logging in, you'll be redirected. Copy the full redirect URL and send it back here."
            )
        except Exception as e:
            send_with_budget(chat_id, f"Login failed: {e}")
        return True

    if lowered.startswith("/queue"):
        lines = ["\U0001f4cb Queue:"]
        if PENDING:
            for t in PENDING:
                tid = str(t.get("id", "?"))[:8]
                typ = t.get("type", "?")
                txt = str(t.get("text") or t.get("description") or "")[:40]
                lines.append(f"  {tid} | {typ} | {txt}")
        else:
            lines.append("  (empty)")
        lines.append(f"\n\U0001f7e1 Pending: {len(PENDING)} \u00b7 \U0001f7e2 Running: {len(RUNNING)}")
        send_with_budget(chat_id, "\n".join(lines))
        return True
    if lowered.startswith("/status"):
        st = load_state()
        sha = st.get("current_sha", "?")[:7]
        version = st.get("version") or "?"

        # Uptime
        uptime_sec = int(time.time() - _LAUNCH_TIME)
        if uptime_sec < 60:
            uptime_str = f"{uptime_sec}s"
        elif uptime_sec < 3600:
            uptime_str = f"{uptime_sec // 60}m {uptime_sec % 60}s"
        elif uptime_sec < 86400:
            h, rem = divmod(uptime_sec, 3600)
            uptime_str = f"{h}h {rem // 60}m"
        else:
            d, rem = divmod(uptime_sec, 86400)
            uptime_str = f"{d}d {rem // 3600}h"

        # Token usage
        prompt_tok = st.get("spent_tokens_prompt", 0)
        comp_tok = st.get("spent_tokens_completion", 0)
        cached_tok = st.get("spent_tokens_cached", 0)
        calls = st.get("spent_calls", 0)

        def _fmt_tokens(n):
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.0f}k"
            return str(n)

        # Queue
        pending_count = len(PENDING)
        running_count = len(RUNNING)

        # Evolution
        evo_enabled = st.get("evolution_mode_enabled", False)
        evo_cycle = st.get("evolution_cycle", 0)

        # Consciousness
        bg_status = "on" if _consciousness.is_running else "off"

        # Model info
        model_primary = MODEL_MAIN or "?"
        model_fallback = MODEL_LIGHT or "?"

        # Kimi 5h window usage
        kimi = get_kimi_usage()
        kimi_remaining = kimi["window_remaining_sec"]
        if kimi["calls"] > 0:
            kh, km = divmod(kimi_remaining, 3600)
            kimi_time = f"{kh}h {km // 60}m left"
            kimi_total = kimi["input_tokens"] + kimi["output_tokens"]
            kimi_line = f"\U0001f4a0 Kimi (5h): {_fmt_tokens(kimi_total)} tokens \u00b7 {kimi['calls']} calls \u00b7 {kimi_time}"
        else:
            kimi_line = f"\U0001f4a0 Kimi (5h): idle"

        lines = [
            f"\U0001f525 Prometheus {version} ({sha})",
            f"\U0001f9e0 Model: {model_primary} \u00b7 Fallback: {model_fallback}",
            f"\U0001f9ee Tokens: {_fmt_tokens(prompt_tok)} in / {_fmt_tokens(comp_tok)} out \u00b7 {calls} calls",
            kimi_line,
            f"\U0001f9f5 Queue: {pending_count} pending \u00b7 {running_count} running \u00b7 {len(WORKERS)} workers",
            f"\U00002699\ufe0f Evo: {'cycle ' + str(evo_cycle) if evo_enabled else 'off'} \u00b7 BG: {bg_status} \u00b7 Up: {uptime_str}",
        ]
        send_with_budget(chat_id, "\n".join(lines))
        return True

    if lowered.startswith("/budget"):
        st = load_state()
        msg = (
            f"💰 **Budget:**\n"
            f"- Spent: ${st.get('spent_usd', 0):.2f}\n"
            f"- Limit: ${TOTAL_BUDGET_LIMIT:.2f}\n"
            f"- Calls: {st.get('spent_calls', 0)}"
        )
        send_with_budget(chat_id, msg)
        return True

    if lowered.startswith("/workers"):
        msg = f"👷 **Workers:** {len(WORKERS)}\n"
        for i, w in enumerate(WORKERS):
            msg += f"- {i}: {_safe_qsize(w.task_q)} tasks queued\n"
        send_with_budget(chat_id, msg.strip())
        return True

    if lowered.startswith("/evolve"):
        if "start" in lowered or "on" in lowered:
            st = load_state()
            st["evolution_mode_enabled"] = True
            st["evolution_consecutive_failures"] = 0
            save_state(st)
            send_with_budget(chat_id, "Evolution enabled.")
        elif "stop" in lowered or "off" in lowered:
            st = load_state()
            st["evolution_mode_enabled"] = False
            save_state(st)
            send_with_budget(chat_id, "Evolution disabled.")
        else:
            st = load_state()
            enabled = st.get("evolution_mode_enabled", False)
            send_with_budget(chat_id, f"Evolution: {'enabled' if enabled else 'disabled'}")
        return True

    if lowered.startswith("/bg"):
        if "start" in lowered or "on" in lowered:
            _consciousness.start()
            send_with_budget(chat_id, "Background consciousness started.")
        elif "stop" in lowered or "off" in lowered:
            _consciousness.stop()
            send_with_budget(chat_id, "Background consciousness stopped.")
        else:
            send_with_budget(chat_id, f"Background: {'running' if _consciousness.is_running else 'stopped'}")
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
        owner_chat_id = st.get("owner_chat_id")
        is_owner = user_id == owner_chat_id

        # Ignore non-owner in private chat unless owner_chat_id not set yet
        if chat_type == "private" and owner_chat_id is None:
            # First message — claim ownership
            st["owner_id"] = user_id
            st["owner_chat_id"] = chat_id
            save_state(st)
            owner_chat_id = chat_id
            is_owner = True
            send_with_budget(chat_id, "👋 Hello! I've been expecting you.")

        # Group handling: allow groups but only respond to @mentions or replies
        is_group = chat_type in ('group', 'supergroup', 'channel')
        
        if not is_owner and not is_group:
            log.debug("Ignoring message from non-owner %s in private chat", user_id)
            continue
            
        # In groups, only respond if:
        # 1. Message is a reply to the bot's message
        # 2. Message contains @mention of the bot
        # 3. Owner is sending the message
        if is_group and not is_owner:
            # Check if this is a reply to the bot
            reply_to = msg.get('reply_to_message')
            is_reply_to_bot = False
            if reply_to:
                reply_from = reply_to.get('from', {})
                if reply_from.get('is_bot') and reply_from.get('username', '').lower().endswith('bot'):
                    is_reply_to_bot = True
            
            # Check for @mention of bot
            bot_username = TG.get_bot_username()
            has_mention = False
            if bot_username and text:
                has_mention = f'@{bot_username}' in text or f'@{bot_username.lower()}' in text.lower()
            
            # Also check entities for mentions
            entities = msg.get('entities', [])
            for entity in entities:
                if entity.get('type') == 'mention':
                    offset = entity.get('offset', 0)
                    length = entity.get('length', 0)
                    mention_text = text[offset:offset+length] if text else ''
                    if bot_username and mention_text.lower() == f'@{bot_username.lower()}':
                        has_mention = True
                        break
            
            if not (is_reply_to_bot or has_mention):
                log.debug("Ignoring group message without mention or reply from %s", user_id)
                continue
            
            # Strip the @mention from text before processing
            if has_mention and bot_username:
                text = text.replace(f'@{bot_username}', '').replace(f'@{bot_username.lower()}', '').strip()
                # Clean up extra spaces
                text = ' '.join(text.split())

        # Check for /login redirect
        if text.startswith("http"):
            if _pending_oauth.get("code_verifier"):
                # Exchange code for tokens
                try:
                    from prometheus.codex_auth import exchange_code_for_tokens, save_auth
                    code_verifier = _pending_oauth.pop("code_verifier")
                    expected_state = _pending_oauth.pop("state", None)
                    tokens = exchange_code_for_tokens(text, code_verifier)
                    save_auth(tokens)
                    send_with_budget(chat_id, "✅ Authenticated successfully!")
                except Exception as e:
                    send_with_budget(chat_id, f"❌ Auth failed: {e}")
                continue

        # Dedup: skip messages we've already processed (Telegram sometimes re-delivers)
        import hashlib as _hl
        _msg_hash = _hl.md5((text or caption or "")[:200].encode()).hexdigest()[:12]
        if not hasattr(handle_one_update, '_seen_msgs'):
            handle_one_update._seen_msgs = {}
        _now = time.time()
        # Clean old entries (>30s)
        handle_one_update._seen_msgs = {
            k: v for k, v in handle_one_update._seen_msgs.items()
            if _now - v < 30
        }
        if _msg_hash in handle_one_update._seen_msgs:
            log.debug("Dedup: skipping re-delivered message hash=%s", _msg_hash)
            continue
        handle_one_update._seen_msgs[_msg_hash] = _now

        # Log chat (incoming message from owner)
        try:
            msg_text = text or caption or "(image)"
            log_chat("in", chat_id, user_id, msg_text)
        except Exception:
            log.exception("log_chat failed")

        # Supervisor commands
        cmd_result = _handle_supervisor_command(text, chat_id, offset)
        if cmd_result is True:
            continue
        dual_path_note = cmd_result  # "" or str

        # Update timestamp
        _last_message_ts = time.time()

        # Enqueue chat task in a thread so main loop can drain events
        threading.Thread(
            target=handle_chat_direct,
            args=(chat_id, text, image_data),
            daemon=True,
        ).start()

    return offset


# 8) Run the main loop
# ----------------------------
st = load_state()
offset = int(st.get("tg_offset", 0))
log.info("Starting main loop at offset %s", offset)

# Background event drain thread — sends worker progress messages to Telegram
# in near-real-time instead of waiting for the 60s long-poll to return.
_drain_stop = threading.Event()


def _event_drain_loop():
    """Drain worker event queue every 1.5s, independent of Telegram polling."""
    while not _drain_stop.is_set():
        try:
            event_q = get_event_q()
            while True:
                try:
                    evt = event_q.get_nowait()
                except _queue_mod.Empty:
                    break
                try:
                    dispatch_event(evt, _event_ctx)
                except Exception:
                    log.debug("Event drain dispatch error", exc_info=True)
            # Also drain pending events from handlers
            while _event_ctx.pending_events:
                try:
                    evt = _event_ctx.pending_events.pop(0)
                    dispatch_event(evt, _event_ctx)
                except Exception:
                    log.debug("Pending event drain error", exc_info=True)
        except Exception:
            log.debug("Event drain loop error", exc_info=True)
        _drain_stop.wait(1.5)


_drain_thread = threading.Thread(
    target=_event_drain_loop, daemon=True, name="event-drain"
)
_drain_thread.start()
log.info("Event drain thread started (1.5s interval)")

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

        # 1.6) drain pending events generated by event handlers (e.g., restart_request)
        try:
            while _event_ctx.pending_events:
                evt = _event_ctx.pending_events.pop(0)
                dispatch_event(evt, _event_ctx)
        except Exception:
            log.exception("Error draining pending events")

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
        except Exception:
            log.exception("Check for scheduled tasks (no-op)")

        # 6) drain worker event queue after task tick (in case tasks added events)
        try:
            event_q = get_event_q()
            while True:
                try:
                    evt = event_q.get_nowait()
                except _queue_mod.Empty:
                    break
                dispatch_event(evt, _event_ctx)
        except Exception:
            log.exception("Error draining event queue after task tick")

        # 6.5) drain pending events again after task tick
        try:
            while _event_ctx.pending_events:
                evt = _event_ctx.pending_events.pop(0)
                dispatch_event(evt, _event_ctx)
        except Exception:
            log.exception("Error draining pending events after task tick")

        # 7) enqueue evolution task if needed
        try:
            enqueue_evolution_task_if_needed()
        except Exception:
            log.exception("Error in enqueue_evolution_task_if_needed")

        # 8) persist queue snapshot
        try:
            persist_queue_snapshot(reason="heartbeat")
        except Exception:
            log.exception("Error persisting queue snapshot")

        # 9) ensure workers healthy
        try:
            ensure_workers_healthy()
        except Exception:
            log.exception("Error in ensure_workers_healthy")

    except KeyboardInterrupt:
        log.info("Keyboard interrupt — exiting")
        break
    except SystemExit as e:
        log.info("SystemExit — exiting: %s", e)
        break
    except Exception:
        log.exception("Fatal error in main loop")
        # Don't exit on unexpected error — try to keep running
        time.sleep(5)

log.info("Main loop exited — goodbye.")


if __name__ == "__main__":
    main()