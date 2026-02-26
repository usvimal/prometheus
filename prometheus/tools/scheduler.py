"""
Scheduler: scheduled and recurring tasks for Prometheus.

Provides tools to schedule tasks that run:
- Once at a specific time
- Recurring at intervals (hourly, daily, etc.)

Scheduler runs as a background thread, checking for due tasks every minute.
Scheduled tasks are persisted to {drive_root}/memory/scheduled.json.

Tools:
- schedule_task_at: Schedule a one-time task
- schedule_task_recurring: Schedule a recurring task  
- schedule_list: List all scheduled tasks
- schedule_cancel: Cancel a scheduled task
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# Module-level scheduler state
_scheduler_thread: Optional[threading.Thread] = None
_shutdown_event = threading.Event()
_scheduled_tasks: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# Default check interval
CHECK_INTERVAL_SEC = 60


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _get_scheduled_file(ctx: ToolContext) -> Path:
    """Get path to scheduled tasks file."""
    sched_dir = ctx.drive_path("memory")
    sched_dir.mkdir(parents=True, exist_ok=True)
    return sched_dir / "scheduled.json"


def _load_scheduled(ctx: ToolContext) -> Dict[str, Dict[str, Any]]:
    """Load scheduled tasks from disk."""
    global _scheduled_tasks
    path = _get_scheduled_file(ctx)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _scheduled_tasks = data
                return _scheduled_tasks
        except Exception as e:
            log.warning("Failed to load scheduled tasks: %s", e)
    return {}


def _save_scheduled(ctx: ToolContext) -> None:
    """Save scheduled tasks to disk."""
    path = _get_scheduled_file(ctx)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_scheduled_tasks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Failed to save scheduled tasks: %s", e)


# ---------------------------------------------------------------------------
# Scheduler thread
# ---------------------------------------------------------------------------

def _get_task_text(task: Dict[str, Any]) -> str:
    """Build task text from scheduled task config."""
    task_type = task.get("task_type", "task")
    
    if task_type == "evolution":
        from supervisor.queue import build_evolution_task_text
        cycle = task.get("cycle", 1)
        return build_evolution_task_text(cycle)
    elif task_type == "review":
        from supervisor.queue import build_review_task_text
        reason = task.get("review_reason", "scheduled review")
        return build_review_task_text(reason)
    else:
        return task.get("text", "")


def _enqueue_scheduled_task(task: Dict[str, Any]) -> bool:
    """Enqueue a scheduled task to the supervisor queue."""
    try:
        from supervisor.queue import enqueue_task, persist_queue_snapshot
        from supervisor.state import load_state
        
        st = load_state()
        owner_chat_id = int(st.get("owner_chat_id") or 0)
        
        task_id = task.get("id", uuid.uuid4().hex[:8])
        task_type = task.get("task_type", "task")
        
        enqueue_task({
            "id": task_id,
            "type": task_type,
            "chat_id": owner_chat_id,
            "text": _get_task_text(task),
            "priority": task.get("priority", 0),
            "scheduled_from": task.get("id"),  # Track source
        })
        persist_queue_snapshot(reason=f"scheduled_task_{task_id}")
        
        log.info("Enqueued scheduled task %s (type=%s)", task_id, task_type)
        return True
    except Exception as e:
        log.error("Failed to enqueue scheduled task: %s", e)
        return False


def _check_and_run_due_tasks(ctx: ToolContext):
    """Check for due tasks and enqueue them."""
    now = time.time()
    due_tasks = []
    
    with _lock:
        for task_id, task in list(_scheduled_tasks.items()):
            # Skip disabled tasks
            if not task.get("enabled", True):
                continue
            
            # Check if it's a one-time or recurring task
            if task.get("recurring"):
                # For recurring, check interval
                interval_sec = task.get("interval_sec", 3600)
                last_run = task.get("last_run_at", 0)
                next_run = last_run + interval_sec
                
                if now >= next_run:
                    due_tasks.append((task_id, task))
            else:
                # One-time task
                run_at = task.get("run_at", 0)
                if now >= run_at and not task.get("executed"):
                    due_tasks.append((task_id, task))
    
    # Enqueue due tasks
    for task_id, task in due_tasks:
        if _enqueue_scheduled_task(task):
            with _lock:
                if task.get("recurring"):
                    # Update last_run for recurring tasks
                    task["last_run_at"] = now
                    # Calculate next run
                    task["next_run_at"] = now + task.get("interval_sec", 3600)
                else:
                    # Mark one-time task as executed
                    task["executed"] = True
                    task["executed_at"] = now
            _save_scheduled(ctx)


def _scheduler_loop(ctx: ToolContext):
    """Main scheduler loop - runs in background thread."""
    log.info("Scheduler thread started")
    
    while not _shutdown_event.is_set():
        try:
            # Reload tasks from disk (in case of restart)
            _load_scheduled(ctx)
            
            # Check for due tasks
            _check_and_run_due_tasks(ctx)
            
            # Save any changes
            _save_scheduled(ctx)
            
        except Exception as e:
            log.error("Scheduler loop error: %s", e)
        
        # Wait for next check interval
        _shutdown_event.wait(CHECK_INTERVAL_SEC)
    
    log.info("Scheduler thread stopped")


def start_scheduler(ctx: ToolContext):
    """Start the scheduler background thread."""
    global _scheduler_thread
    
    if _scheduler_thread and _scheduler_thread.is_alive():
        return "Scheduler already running"
    
    _shutdown_event.clear()
    _load_scheduled(ctx)
    
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(ctx,),
        daemon=True,
        name="scheduler-thread"
    )
    _scheduler_thread.start()
    
    return "Scheduler started"


def stop_scheduler():
    """Stop the scheduler background thread."""
    global _scheduler_thread
    
    _shutdown_event.set()
    
    if _scheduler_thread and _scheduler_thread.is_alive():
        _scheduler_thread.join(timeout=5)
    
    _scheduler_thread = None
    return "Scheduler stopped"


def is_scheduler_running() -> bool:
    """Check if scheduler is running."""
    return _scheduler_thread is not None and _scheduler_thread.is_alive()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _schedule_task_at(ctx: ToolContext, text: str, run_at: str, 
                      task_type: str = "task", priority: int = 0) -> str:
    """Schedule a one-time task to run at a specific time.
    
    Args:
        text: Task text/content
        run_at: ISO timestamp or relative time (e.g., "2026-02-26T12:00:00", "in 30 minutes", "every hour")
        task_type: Type of task (task, evolution, review)
        priority: Task priority
    """
    if not text or not text.strip():
        return "Error: text is required"
    if not run_at or not run_at.strip():
        return "Error: run_at is required"
    
    # Parse run_at time
    run_at_ts = _parse_time_string(run_at.strip())
    if run_at_ts is None:
        return f"Error: Invalid time format. Use ISO (2026-02-26T12:00:00Z) or relative (in 30 minutes, every hour)"
    
    # Check if time is in the past
    now = time.time()
    if run_at_ts < now:
        return f"Error: run_at time is in the past ({datetime.datetime.fromtimestamp(run_at_ts).isoformat()})"
    
    task_id = uuid.uuid4().hex[:8]
    
    task = {
        "id": task_id,
        "text": text.strip(),
        "run_at": run_at_ts,
        "task_type": task_type,
        "priority": priority,
        "recurring": False,
        "enabled": True,
        "created_at": now,
    }
    
    with _lock:
        _scheduled_tasks[task_id] = task
    
    _save_scheduled(ctx)
    
    run_at_dt = datetime.datetime.fromtimestamp(run_at_ts)
    return f"Scheduled one-time task {task_id} to run at {run_at_dt.isoformat()} (type={task_type})"


def _schedule_task_recurring(ctx: ToolContext, text: str, interval: str,
                              task_type: str = "task", priority: int = 0) -> str:
    """Schedule a recurring task.
    
    Args:
        text: Task text/content
        interval: Interval string (e.g., "hourly", "daily", "every 30 minutes")
        task_type: Type of task (task, evolution, review)
        priority: Task priority
    """
    if not text or not text.strip():
        return "Error: text is required"
    if not interval or not interval.strip():
        return "Error: interval is required"
    
    # Parse interval
    interval_sec = _parse_interval(interval.strip())
    if interval_sec is None:
        return f"Error: Invalid interval. Use: hourly, daily, every hour, every 30 minutes, etc."
    
    now = time.time()
    task_id = uuid.uuid4().hex[:8]
    
    task = {
        "id": task_id,
        "text": text.strip(),
        "interval_sec": interval_sec,
        "interval": interval.strip(),
        "task_type": task_type,
        "priority": priority,
        "recurring": True,
        "enabled": True,
        "created_at": now,
        "last_run_at": 0,
        "next_run_at": now + interval_sec,
    }
    
    with _lock:
        _scheduled_tasks[task_id] = task
    
    _save_scheduled(ctx)
    
    next_run_dt = datetime.datetime.fromtimestamp(now + interval_sec)
    return f"Scheduled recurring task {task_id} to run every {interval.strip()} (type={task_type}, first run: {next_run_dt.isoformat()})"


def _schedule_list(ctx: ToolContext) -> str:
    """List all scheduled tasks."""
    _load_scheduled(ctx)
    
    if not _scheduled_tasks:
        return "No scheduled tasks. Use schedule_task_at or schedule_task_recurring to create one."
    
    now = time.time()
    lines = [f"Scheduled tasks ({len(_scheduled_tasks)} total):\n"]
    
    for task_id, task in _scheduled_tasks.items():
        enabled = "✅" if task.get("enabled", True) else "❌"
        recurring = "🔄" if task.get("recurring") else "⏰"
        
        if task.get("recurring"):
            interval = task.get("interval", "unknown")
            next_run = task.get("next_run_at", 0)
            next_str = datetime.datetime.fromtimestamp(next_run).isoformat() if next_run > 0 else "N/A"
            last_run = task.get("last_run_at", 0)
            last_str = datetime.datetime.fromtimestamp(last_run).isoformat() if last_run > 0 else "never"
            lines.append(
                f"{enabled} {recurring} {task_id}: {task.get('task_type', 'task')} every {interval}\n"
                f"   next: {next_str}, last: {last_str}"
            )
        else:
            run_at = task.get("run_at", 0)
            run_at_str = datetime.datetime.fromtimestamp(run_at).isoformat() if run_at > 0 else "N/A"
            executed = " (EXECUTED)" if task.get("executed") else ""
            lines.append(
                f"{enabled} {recurring} {task_id}: {task.get('task_type', 'task')} at {run_at_str}{executed}"
            )
        
        lines.append(f"   text: {task.get('text', '')[:80]}...")
    
    return "\n".join(lines)


def _schedule_cancel(ctx: ToolContext, task_id: str) -> str:
    """Cancel a scheduled task."""
    _load_scheduled(ctx)
    
    if task_id not in _scheduled_tasks:
        return f"Task {task_id} not found"
    
    with _lock:
        del _scheduled_tasks[task_id]
    
    _save_scheduled(ctx)
    
    return f"Cancelled scheduled task {task_id}"


def _schedule_enable(ctx: ToolContext, task_id: str, enabled: bool = True) -> str:
    """Enable or disable a scheduled task."""
    _load_scheduled(ctx)
    
    if task_id not in _scheduled_tasks:
        return f"Task {task_id} not found"
    
    with _lock:
        _scheduled_tasks[task_id]["enabled"] = enabled
    
    _save_scheduled(ctx)
    
    state = "enabled" if enabled else "disabled"
    return f"Task {task_id} {state}"


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------

def _parse_time_string(time_str: str) -> Optional[float]:
    """Parse time string to Unix timestamp.
    
    Supports:
    - ISO 8601: "2026-02-26T12:00:00Z", "2026-02-26T12:00:00+00:00"
    - Relative: "in 30 minutes", "in 2 hours", "in 1 day"
    """
    time_str = time_str.strip().lower()
    now = time.time()
    
    # Try ISO format first
    try:
        # Handle Z suffix
        ts = time_str.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(ts)
        return dt.timestamp()
    except Exception:
        pass
    
    # Try relative time
    import re
    
    # Match "in X minutes/hours/days"
    match = re.match(r"in\s+(\d+)\s*(minute|minutes|hour|hours|day|days|second|seconds)", time_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        
        if "minute" in unit:
            return now + (value * 60)
        elif "hour" in unit:
            return now + (value * 3600)
        elif "day" in unit:
            return now + (value * 86400)
        elif "second" in unit:
            return now + value
    
    return None


def _parse_interval(interval_str: str) -> Optional[int]:
    """Parse interval string to seconds.
    
    Supports:
    - "hourly", "daily", "weekly"
    - "every X minutes/hours/days"
    - "X minutes", "X hours", "X days"
    """
    interval_str = interval_str.strip().lower()
    
    # Direct keywords
    keyword_map = {
        "hourly": 3600,
        "every hour": 3600,
        "every hour": 3600,
        "daily": 86400,
        "every day": 86400,
        "weekly": 604800,
        "every week": 604800,
        "minutely": 60,
        "every minute": 60,
    }
    
    if interval_str in keyword_map:
        return keyword_map[interval_str]
    
    # Try numeric pattern
    import re
    match = re.match(r"every\s+(\d+)\s*(minute|minutes|hour|hours|day|days|second|seconds)", interval_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        
        if "minute" in unit:
            return value * 60
        elif "hour" in unit:
            return value * 3600
        elif "day" in unit:
            return value * 86400
        elif "second" in unit:
            return value
    
    # Try just number + unit
    match = re.match(r"(\d+)\s*(minute|minutes|hour|hours|day|days|second|seconds)", interval_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        
        if "minute" in unit:
            return value * 60
        elif "hour" in unit:
            return value * 3600
        elif "day" in unit:
            return value * 86400
        elif "second" in unit:
            return value
    
    return None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("schedule_task_at", {
            "name": "schedule_task_at",
            "description": (
                "Schedule a one-time task to run at a specific time. "
                "Supports ISO timestamps (2026-02-26T12:00:00Z) or relative time (in 30 minutes). "
                "Task will be enqueued when the time is reached."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Task text/content to execute",
                    },
                    "run_at": {
                        "type": "string",
                        "description": "When to run - ISO timestamp or relative (in 30 minutes, in 2 hours, tomorrow)",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["task", "evolution", "review"],
                        "default": "task",
                        "description": "Type of task",
                    },
                    "priority": {
                        "type": "integer",
                        "default": 0,
                        "description": "Task priority (0=highest)",
                    },
                },
                "required": ["text", "run_at"],
            },
        }, _schedule_task_at),
        ToolEntry("schedule_task_recurring", {
            "name": "schedule_task_recurring",
            "description": (
                "Schedule a recurring task that runs at regular intervals. "
                "Supports: hourly, daily, every 30 minutes, every 2 hours, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Task text/content to execute",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Interval - hourly, daily, every 30 minutes, every 2 hours, etc.",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["task", "evolution", "review"],
                        "default": "task",
                        "description": "Type of task",
                    },
                    "priority": {
                        "type": "integer",
                        "default": 0,
                        "description": "Task priority (0=highest)",
                    },
                },
                "required": ["text", "interval"],
            },
        }, _schedule_task_recurring),
        ToolEntry("schedule_list", {
            "name": "schedule_list",
            "description": "List all scheduled tasks with their status, next run time, and history.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }, _schedule_list),
        ToolEntry("schedule_cancel", {
            "name": "schedule_cancel",
            "description": "Cancel a scheduled task by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the scheduled task to cancel",
                    },
                },
                "required": ["task_id"],
            },
        }, _schedule_cancel),
        ToolEntry("schedule_enable", {
            "name": "schedule_enable",
            "description": "Enable or disable a scheduled task without deleting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the scheduled task",
                    },
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "True to enable, False to disable",
                    },
                },
                "required": ["task_id"],
            },
        }, _schedule_enable),
    ]
