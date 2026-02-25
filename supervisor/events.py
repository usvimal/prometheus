"""Event handlers for supervisor."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict

from .state import StateManager

logger = logging.getLogger(__name__)

# Global event bus - handlers registered at module load
_EVENT_HANDLERS: Dict[str, callable] = {}


def on_event(event_type: str):
    """Decorator to register an event handler."""

    def decorator(func: callable):
        _EVENT_HANDLERS[event_type] = func
        return func

    return decorator


def dispatch(evt: Dict[str, Any], ctx: Any) -> None:
    """Dispatch an event to its handler."""
    event_type = evt.get("type")
    handler = _EVENT_HANDLERS.get(event_type)
    if handler:
        try:
            handler(evt, ctx)
        except Exception as e:
            logger.exception(f"Error handling event {event_type}: {e}")
    else:
        logger.debug(f"No handler for event type: {event_type}")


# ============================================================================
# Task Events
# ============================================================================


@on_event("task_received")
def _handle_task_received(evt: Dict[str, Any], ctx: Any) -> None:
    """Log task receipt."""
    task = evt.get("task", {})
    task_type = task.get("type", "unknown")
    task_id = task.get("id", "?")
    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "task_received",
            "task_id": task_id,
            "task_type": task_type,
        },
    )


@on_event("task_done")
def _handle_task_done(evt: Dict[str, Any], ctx: Any) -> None:
    task_id = evt.get("task_id")
    task_type = str(evt.get("task_type") or "")
    wid = evt.get("worker_id")

    # Track evolution task success/failure for circuit breaker
    if task_type == "evolution":
        st = ctx.load_state()
        
        # Get the actual metrics from the event
        cost = float(evt.get("cost_usd") or 0)
        rounds = int(evt.get("total_rounds") or 0)
        
        logger.info(f"Evolution task {task_id} completed: rounds={rounds}, cost=${cost}")

        # FIXED: More robust success detection
        # A successful evolution should have:
        # - At least 1 round of work (LLM actually ran)
        # - Not a zero-round "immediate failure"
        # Even if cost is $0 (MiniMax free tier), the rounds indicate real work
        is_success = rounds > 0

        if is_success:
            # Success: reset failure counter
            st["evolution_consecutive_failures"] = 0
            logger.info(f"Evolution task {task_id} succeeded (rounds={rounds}), resetting failure counter")
            ctx.save_state(st)
        else:
            # Likely failure (no rounds = immediate failure)
            failures = int(st.get("evolution_consecutive_failures") or 0) + 1
            st["evolution_consecutive_failures"] = failures
            ctx.save_state(st)
            logger.warning(f"Evolution task {task_id} failed (rounds={rounds}), consecutive failures: {failures}")
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "evolution_task_failure_tracked",
                    "task_id": task_id,
                    "consecutive_failures": failures,
                    "cost_usd": cost,
                    "rounds": rounds,
                },
            )


@on_event("worker_boot")
def _handle_worker_boot(evt: Dict[str, Any], ctx: Any) -> None:
    """Log worker boot."""
    pid = evt.get("pid")
    git_sha = evt.get("git_sha", "?")[:8]
    git_branch = evt.get("git_branch", "?")

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "worker_boot",
            "pid": pid,
            "git_sha": git_sha,
            "git_branch": git_branch,
        },
    )


@on_event("worker_crash")
def _handle_worker_crash(evt: Dict[str, Any], ctx: Any) -> None:
    """Log worker crash."""
    worker_id = evt.get("worker_id")
    task_id = evt.get("task_id")
    exit_code = evt.get("exit_code")

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "worker_crash",
            "worker_id": worker_id,
            "task_id": task_id,
            "exit_code": exit_code,
        },
    )


@on_event("worker_timeout")
def _handle_worker_timeout(evt: Dict[str, Any], ctx: Any) -> None:
    """Log worker timeout."""
    worker_id = evt.get("worker_id")
    task_id = evt.get("task_id")
    elapsed = evt.get("elapsed_sec")

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "worker_timeout",
            "worker_id": worker_id,
            "task_id": task_id,
            "elapsed_sec": elapsed,
        },
    )


@on_event("worker_terminate")
def _handle_worker_terminate(evt: Dict[str, Any], ctx: Any) -> None:
    """Log worker termination."""
    worker_id = evt.get("worker_id")
    task_id = evt.get("task_id")
    reason = evt.get("reason")

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "worker_terminate",
            "worker_id": worker_id,
            "task_id": task_id,
            "reason": reason,
        },
    )


@on_event("owner_message_injected")
def _handle_owner_message_injected(evt: Dict[str, Any], ctx: Any) -> None:
    """Log owner message injection into a running task."""
    task_id = evt.get("task_id")
    message_preview = evt.get("message_preview", "")[:50]

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "owner_message_injected",
            "task_id": task_id,
            "message_preview": message_preview,
        },
    )


# ============================================================================
# Startup / Health Events
# ============================================================================


@on_event("startup_verification")
def _handle_startup_verification(evt: Dict[str, Any], ctx: Any) -> None:
    """Log startup verification results."""
    checks = evt.get("checks", {})
    issues_count = evt.get("issues_count", 0)
    git_sha = evt.get("git_sha", "?")[:8]

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "startup_verification",
            "git_sha": git_sha,
            "issues_count": issues_count,
            "checks": checks,
        },
    )


# ============================================================================
# Budget Events
# ============================================================================


@on_event("budget_threshold_reached")
def _handle_budget_threshold(evt: Dict[str, Any], ctx: Any) -> None:
    """Log budget threshold warnings."""
    spent = evt.get("spent_usd", 0)
    threshold = evt.get("threshold", 0)
    pct = (spent / threshold * 100) if threshold > 0 else 0

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "budget_threshold_reached",
            "spent_usd": spent,
            "threshold": threshold,
            "pct": pct,
        },
    )


@on_event("budget_exhausted")
def _handle_budget_exhausted(evt: Dict[str, Any], ctx: Any) -> None:
    """Log budget exhaustion."""
    spent = evt.get("spent_usd", 0)

    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "budget_exhausted",
            "spent_usd": spent,
        },
    )
