"""Event handlers for supervisor."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict


from supervisor.queue import enqueue_task


logger = logging.getLogger(__name__)

# Global event bus - handlers registered at module load
_EVENT_HANDLERS: Dict[str, callable] = {}


def on_event(event_type: str):
    """Decorator to register an event handler."""

    def decorator(func: callable):
        _EVENT_HANDLERS[event_type] = func
        return func

    return decorator


def dispatch_event(evt: Dict[str, Any], ctx: Any) -> None:
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


@on_event("send_message")
def _handle_send_message(evt, ctx):
    """Send a message to Telegram (from worker process)."""
    try:
        log_text = evt.get("log_text")
        fmt = str(evt.get("format") or "")
        is_progress = bool(evt.get("is_progress"))
        reply_to = evt.get("reply_to_message_id")
        thread_id = evt.get("message_thread_id")
        ctx.send_with_budget(
            int(evt["chat_id"]),
            str(evt.get("text") or ""),
            log_text=(str(log_text) if isinstance(log_text, str) else None),
            fmt=fmt,
            is_progress=is_progress,
            reply_to_message_id=int(reply_to) if reply_to else None,
            message_thread_id=int(thread_id) if thread_id else None,
        )
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "send_message_event_error", "error": repr(e),
            },
        )


@on_event("task_received")
def _handle_task_received(evt: Dict[str, Any], ctx: Any) -> None:
    """Log task receipt. For evolution tasks, save git HEAD for restart decision."""
    task = evt.get("task", {})
    task_type = task.get("type", "unknown")
    task_id = task.get("id", "?")

    # Save git HEAD for evolution tasks so we can detect if commits were made
    if task_type == "evolution":
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True,
                cwd=str(ctx.DRIVE_ROOT / ".." / "repo"),
                timeout=5,
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
                st = ctx.load_state()
                st["_evolution_start_sha"] = sha
                ctx.save_state(st)
                logger.debug("Saved evolution start SHA: %s", sha[:8])
        except Exception:
            logger.debug("Failed to save evolution start SHA", exc_info=True)

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

    # --- Clear RUNNING dict and free the worker ---
    # Without this, completed tasks stay in RUNNING forever and the
    # timeout enforcer kills them at 30 min (false hard-timeout).
    if task_id and hasattr(ctx, "RUNNING") and task_id in ctx.RUNNING:
        ctx.RUNNING.pop(task_id, None)
        logger.info("Cleared RUNNING for completed task %s", task_id)
    if wid is not None and hasattr(ctx, "WORKERS"):
        worker = ctx.WORKERS.get(wid)
        if worker is not None and getattr(worker, "busy_task_id", None) == task_id:
            worker.busy_task_id = None
            logger.info("Freed worker %s (was busy with %s)", wid, task_id)
    if task_id and hasattr(ctx, "persist_queue_snapshot"):
        try:
            ctx.persist_queue_snapshot(reason="task_done")
        except Exception:
            logger.debug("Failed to persist queue snapshot after task_done", exc_info=True)

    # Track evolution task success/failure for circuit breaker
    # Also trigger auto-restart after successful evolution
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
            
            # Only restart if evolution actually committed code changes
            # Compare current git HEAD with SHA saved at task start
            needs_restart = False
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True,
                    cwd=str(ctx.DRIVE_ROOT / ".." / "repo"),
                    timeout=5,
                )
                current_sha = result.stdout.strip() if result.returncode == 0 else ""
                start_sha = st.get("_evolution_start_sha", "")
                if current_sha and start_sha and current_sha != start_sha:
                    needs_restart = True
                    logger.info(f"Evolution made commits: {start_sha[:8]} -> {current_sha[:8]}")
                elif not start_sha:
                    needs_restart = True
                    logger.warning("No evolution start SHA found, restarting to be safe")
                else:
                    logger.info(f"Evolution made no commits (HEAD still {current_sha[:8]}), skipping restart")
            except Exception:
                needs_restart = True
                logger.debug("Failed to check git HEAD for restart decision", exc_info=True)

            # Clean up saved SHA
            st.pop("_evolution_start_sha", None)
            ctx.save_state(st)

            if needs_restart:
                logger.info(f"Evolution task {task_id} succeeded with commits - requesting restart")
                ctx.pending_events.append({
                    "type": "restart_request",
                    "reason": f"Auto-restart after successful evolution (task {task_id})",
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
                if st.get("owner_chat_id"):
                    try:
                        ctx.send_with_budget(
                            int(st["owner_chat_id"]),
                            "🧬 Evolution completed with code changes! Restarting to apply...",
                        )
                    except Exception as e:
                        logger.debug(f"Failed to send evolution notification: {e}")
            else:
                if st.get("owner_chat_id"):
                    try:
                        ctx.send_with_budget(
                            int(st["owner_chat_id"]),
                            "🧬 Evolution completed (analysis only, no code changes). No restart needed.",
                        )
                    except Exception as e:
                        logger.debug(f"Failed to send evolution notification: {e}")
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


# ============================================================================
# Control Events
# ============================================================================


@on_event("schedule_task")
def _handle_schedule_task(evt: Dict[str, Any], ctx: Any) -> None:
    """Handle schedule_task events - actually enqueue the task."""
    task_id = evt.get("task_id")
    description = evt.get("description")
    context = evt.get("context", "")
    parent_task_id = evt.get("parent_task_id")
    
    if not task_id or not description:
        logger.error(f"Invalid schedule_task event: {evt}")
        return
    
    logger.info(f"Processing schedule_task event: {task_id}")
    
    # Build task dict and enqueue
    task = {
        "id": task_id,
        "type": "task",
        "description": description,
        "context": context,
        "parent_task_id": parent_task_id,
    }
    enqueue_task(task)
    
    logger.info(f"Task {task_id} enqueued successfully")


@on_event("restart_request")
def _handle_restart_request(evt: Dict[str, Any], ctx: Any) -> None:
    """Handle restart request events - trigger supervisor restart."""
    reason = evt.get("reason", "Unknown reason")
    logger.info(f"Processing restart request: {reason}")
    # Import here to avoid circular imports
    from supervisor.git_ops import safe_restart
    safe_restart(reason)
