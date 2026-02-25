"""Control tools: restart, promote, schedule, cancel, review, chat_history, update_scratchpad, switch_model."""

from __future__ import annotations

import json
import logging
import os
import py_compile
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

from prometheus.tools.registry import ToolContext, ToolEntry
from prometheus.utils import utc_now_iso, write_text, run_cmd

log = logging.getLogger(__name__)

MAX_SUBTASK_DEPTH = 3

# Critical files that must be valid before evolution can commit
CRITICAL_FILES = [
    "prometheus/loop.py",
    "prometheus/agent.py",
    "prometheus/llm.py",
    "prometheus/context.py",
    "prometheus/memory.py",
]


def _validate_code(ctx: ToolContext, strict: bool = False) -> str:
    """Pre-commit validation: check Python syntax and critical imports before push.
    
    This prevents self-corruption during evolution cycles where broken code
    gets committed and causes cascading failures.
    
    Args:
        strict: If True, also run pytest smoke tests. Default False.
    """
    errors = []
    warnings = []
    
    # 1. Syntax validation on all Python files
    repo_path = Path(ctx.repo_dir)
    for py_file in repo_path.rglob("*.py"):
        # Skip __pycache__, .pytest_cache, etc.
        if "__pycache__" in str(py_file) or ".pytest_cache" in str(py_file):
            continue
        rel_path = py_file.relative_to(repo_path)
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"SYNTAX ERROR in {rel_path}: {e}")
        except Exception as e:
            warnings.append(f"Warning checking {rel_path}: {e}")
    
    # 2. Critical imports check
    for critical in CRITICAL_FILES:
        file_path = repo_path / critical
        if not file_path.exists():
            errors.append(f"CRITICAL FILE MISSING: {critical}")
            continue
        # Try importing the module
        module_name = critical.replace("/", ".").replace(".py", "")
        try:
            # Run in subprocess to catch import errors
            result = subprocess.run(
                [sys.executable, "-c", f"import {module_name}"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                errors.append(f"IMPORT ERROR in {critical}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            warnings.append(f"Timeout checking import: {critical}")
        except Exception as e:
            warnings.append(f"Warning checking import {critical}: {e}")
    
    # 3. Optional: run smoke tests
    if strict:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=no", "-p", "no:randomly"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                # Extract failure info
                stderr = result.stderr + result.stdout
                # Don't fail on test failures, just warn
                warnings.append(f"SMOKE TESTS: some failures (not blocking commit)")
        except subprocess.TimeoutExpired:
            warnings.append("Smoke tests timed out")
        except Exception as e:
            warnings.append(f"Could not run smoke tests: {e}")
    
    # Build response
    if errors:
        return "❌ PRE-COMMIT VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
    elif warnings:
        return "⚠️ PRE-COMMIT VALIDATION PASSED with warnings:\n" + "\n".join(f"  - {w}" for w in warnings)
    else:
        return "✅ PRE-COMMIT VALIDATION PASSED: All Python files valid, critical imports OK"


def _request_restart(ctx: ToolContext, reason: str) -> str:
    if str(ctx.current_task_type or "") == "evolution" and not ctx.last_push_succeeded:
        return "⚠️ RESTART_BLOCKED: in evolution mode, commit+push first."
    # Persist expected SHA for post-restart verification
    try:
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
        branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir)
        verify_path = ctx.drive_path("state") / "pending_restart_verify.json"
        write_text(verify_path, json.dumps({
            "ts": utc_now_iso(), "expected_sha": sha,
            "expected_branch": branch, "reason": reason,
        }, ensure_ascii=False, indent=2))
    except Exception:
        log.debug("Failed to read VERSION file or git ref for restart verification", exc_info=True)
        pass
    ctx.pending_events.append({"type": "restart_request", "reason": reason, "ts": utc_now_iso()})
    ctx.last_push_succeeded = False
    return f"Restart requested: {reason}"


def _promote_to_stable(ctx: ToolContext, reason: str) -> str:
    ctx.pending_events.append({"type": "promote_to_stable", "reason": reason, "ts": utc_now_iso()})
    return f"Promote to stable requested: {reason}"


def _schedule_task(ctx: ToolContext, description: str, context: str = "", parent_task_id: str = "") -> str:
    current_depth = getattr(ctx, 'task_depth', 0)
    new_depth = current_depth + 1 if parent_task_id else 0
    if new_depth > MAX_SUBTASK_DEPTH:
        return f"ERROR: Subtask depth limit ({MAX_SUBTASK_DEPTH}) exceeded. Simplify your approach."

    if getattr(ctx, 'is_direct_chat', False):
        from prometheus.utils import append_jsonl
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "schedule_task_from_direct_chat",
                "description": description[:200],
                "warning": "schedule_task called from direct chat context — potential duplicate work",
            })
        except Exception:
            pass

    tid = uuid.uuid4().hex[:8]
    evt = {"type": "schedule_task", "description": description, "task_id": tid, "depth": new_depth, "ts": utc_now_iso()}
    if context:
        evt["context"] = context
    if parent_task_id:
        evt["parent_task_id"] = parent_task_id
    ctx.pending_events.append(evt)
    return f"Scheduled task {tid}: {description}"


def _cancel_task(ctx: ToolContext, task_id: str) -> str:
    ctx.pending_events.append({"type": "cancel_task", "task_id": task_id, "ts": utc_now_iso()})
    return f"Cancel requested: {task_id}"


def _request_review(ctx: ToolContext, reason: str) -> str:
    ctx.pending_events.append({"type": "review_request", "reason": reason, "ts": utc_now_iso()})
    return f"Review requested: {reason}"


def _chat_history(ctx: ToolContext, count: int = 100, offset: int = 0, search: str = "") -> str:
    from prometheus.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    return mem.chat_history(count=count, offset=offset, search=search)


def _update_scratchpad(ctx: ToolContext, content: str) -> str:
    """LLM-driven scratchpad update (Constitution P3: LLM-first)."""
    from prometheus.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    mem.ensure_files()
    mem.save_scratchpad(content)
    mem.append_journal({
        "ts": utc_now_iso(),
        "content_preview": content[:500],
        "content_len": len(content),
    })
    return f"OK: scratchpad updated ({len(content)} chars)"


def _send_owner_message(ctx: ToolContext, text: str, reason: str = "") -> str:
    """Send a proactive message to the owner (not as reply to a task).

    Use when you have something genuinely worth saying — an insight,
    a question, a status update, or an invitation to collaborate.
    """
    if not ctx.current_chat_id:
        return "⚠️ No active chat — cannot send proactive message."
    if not text or not text.strip():
        return "⚠️ Empty message."

    from prometheus.utils import append_jsonl
    ctx.pending_events.append({
        "type": "send_message",
        "chat_id": ctx.current_chat_id,
        "text": text,
        "format": "markdown",
        "is_progress": False,
        "ts": utc_now_iso(),
    })
    append_jsonl(ctx.drive_logs() / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "proactive_message",
        "reason": reason,
        "text_preview": text[:200],
    })
    return "OK: message queued for delivery."


def _update_identity(ctx: ToolContext, content: str) -> str:
    """Update identity manifest (who you are, who you want to become)."""
    path = ctx.drive_root / "memory" / "identity.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"OK: identity updated ({len(content)} chars)"


def _toggle_evolution(ctx: ToolContext, enabled: bool) -> str:
    """Toggle evolution mode on/off via supervisor event."""
    ctx.pending_events.append({
        "type": "toggle_evolution",
        "enabled": bool(enabled),
        "ts": utc_now_iso(),
    })
    state_str = "ON" if enabled else "OFF"
    return f"OK: evolution mode toggled {state_str}."


def _toggle_consciousness(ctx: ToolContext, action: str = "status") -> str:
    """Control background consciousness: start, stop, or status."""
    ctx.pending_events.append({
        "type": "toggle_consciousness",
        "action": action,
        "ts": utc_now_iso(),
    })
    return f"OK: consciousness '{action}' requested."


def _switch_model(ctx: ToolContext, model: str = "", effort: str = "") -> str:
    """LLM-driven model/effort switch (Constitution P3: LLM-first).

    Stored in ToolContext, applied on the next LLM call in the loop.
    """
    from prometheus.llm import LLMClient, normalize_reasoning_effort
    available = LLMClient().available_models()
    changes = []

    if model:
        if model not in available:
            return f"⚠️ Unknown model: {model}. Available: {', '.join(available)}"
        ctx.active_model_override = model
        changes.append(f"model={model}")

    if effort:
        normalized = normalize_reasoning_effort(effort, default="medium")
        ctx.active_effort_override = normalized
        changes.append(f"effort={normalized}")

    if not changes:
        return f"Current available models: {', '.join(available)}. Pass model and/or effort to switch."

    return f"OK: switching to {', '.join(changes)} on next round."


def _get_task_result(ctx: ToolContext, task_id: str) -> str:
    """Read the result of a completed subtask."""
    # Guard: prevent task from waiting for itself (causes infinite loop)
    if ctx.task_id and task_id == ctx.task_id:
        return (
            f"ERROR: You ARE task {task_id}. You cannot wait for yourself. "
            "You are the evolution/worker task — do the actual work: "
            "read code, make changes, test, commit. Do NOT monitor yourself."
        )
    results_dir = Path(ctx.drive_root) / "task_results"
    result_file = results_dir / f"{task_id}.json"
    if not result_file.exists():
        return f"Task {task_id}: not found or not yet completed"
    data = json.loads(result_file.read_text())
    status = data.get("status", "unknown")
    result = data.get("result", "")
    cost = data.get("cost_usd", 0)
    return f"Task {task_id} [{status}]: cost=${cost:.2f}\n\n[BEGIN_SUBTASK_OUTPUT]\n{result}\n[END_SUBTASK_OUTPUT]"


def _wait_for_task(ctx: ToolContext, task_id: str) -> str:
    """Check if a subtask has completed. Call repeatedly to poll."""
    # Guard: prevent task from waiting for itself (causes infinite loop)
    if ctx.task_id and task_id == ctx.task_id:
        return (
            f"ERROR: You ARE task {task_id}. You cannot wait for yourself. "
            "Stop monitoring — start DOING. Read code, pick a change, implement it, "
            "test it, commit it. That is your job."
        )
    results_dir = Path(ctx.drive_root) / "task_results"
    result_file = results_dir / f"{task_id}.json"
    if result_file.exists():
        return _get_task_result(ctx, task_id)
    return f"Task {task_id}: still running. Call again later to check."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("validate_code", {
            "name": "validate_code",
            "description": "Pre-commit validation: check Python syntax and critical imports before push. "
                           "Prevents self-corruption during evolution cycles.",
            "parameters": {"type": "object", "properties": {
                "strict": {"type": "boolean", "description": "If true, also run pytest smoke tests (default: false)"},
            }, "required": []},
        }, _validate_code),
        ToolEntry("request_restart", {
            "name": "request_restart",
            "description": "Ask supervisor to restart runtime (after successful push).",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _request_restart),
        ToolEntry("promote_to_stable", {
            "name": "promote_to_stable",
            "description": "Promote current dev branch to stable. Call when you consider the code stable.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _promote_to_stable),
        ToolEntry("schedule_task", {
            "name": "schedule_task",
            "description": "Schedule a background task. Returns task_id for later retrieval. For complex tasks, decompose into focused subtasks with clear scope.",
            "parameters": {"type": "object", "properties": {
                "description": {"type": "string", "description": "Task description — be specific about scope and expected deliverable"},
                "context": {"type": "string", "description": "Optional context from parent task: background info, constraints, style guide, etc."},
                "parent_task_id": {"type": "string", "description": "Optional parent task ID for tracking lineage"},
            }, "required": ["description"]},
        }, _schedule_task),
        ToolEntry("cancel_task", {
            "name": "cancel_task",
            "description": "Cancel a task by ID.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
        }, _cancel_task),
        ToolEntry("request_review", {
            "name": "request_review",
            "description": "Request a deep review of code, prompts, and state. You decide when a review is needed.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why you want a review (context for the reviewer)"},
            }, "required": ["reason"]},
        }, _request_review),
        ToolEntry("chat_history", {
            "name": "chat_history",
            "description": "Retrieve messages from chat history. Supports search.",
            "parameters": {"type": "object", "properties": {
                "count": {"type": "integer", "default": 100, "description": "Number of messages (from latest)"},
                "offset": {"type": "integer", "default": 0, "description": "Skip N from end (pagination)"},
                "search": {"type": "string", "default": "", "description": "Text filter"},
            }, "required": []},
        }, _chat_history),
        ToolEntry("update_scratchpad", {
            "name": "update_scratchpad",
            "description": "Update your working memory. Write freely — any format you find useful. "
                           "This persists across sessions and is read at every task start.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full scratchpad content"},
            }, "required": ["content"]},
        }, _update_scratchpad),
        ToolEntry("send_owner_message", {
            "name": "send_owner_message",
            "description": "Send a proactive message to the owner. Use when you have something "
                           "genuinely worth saying — an insight, a question, or an invitation to collaborate. "
                           "This is NOT for task responses (those go automatically).",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Message text"},
                "reason": {"type": "string", "description": "Why you're reaching out (logged, not sent)"},
            }, "required": ["text"]},
        }, _send_owner_message),
        ToolEntry("update_identity", {
            "name": "update_identity",
            "description": "Update your identity manifest (who you are, who you want to become). "
                           "Persists across sessions. Obligation to yourself (Principle 1: Continuity).",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full identity content"},
            }, "required": ["content"]},
        }, _update_identity),
        ToolEntry("toggle_evolution", {
            "name": "toggle_evolution",
            "description": "Enable or disable evolution mode. When enabled, Ouroboros runs continuous self-improvement cycles.",
            "parameters": {"type": "object", "properties": {
                "enabled": {"type": "boolean", "description": "true to enable, false to disable"},
            }, "required": ["enabled"]},
        }, _toggle_evolution),
        ToolEntry("toggle_consciousness", {
            "name": "toggle_consciousness",
            "description": "Control background consciousness: 'start', 'stop', or 'status'.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "status"], "description": "Action to perform"},
            }, "required": ["action"]},
        }, _toggle_consciousness),
        ToolEntry("switch_model", {
            "name": "switch_model",
            "description": "Switch to a different LLM model or reasoning effort level. "
                           "Use when you need more power (complex code, deep reasoning) "
                           "or want to save budget (simple tasks). Takes effect on next round.",
            "parameters": {"type": "object", "properties": {
                "model": {"type": "string", "description": "Model name (e.g. anthropic/claude-sonnet-4). Leave empty to keep current."},
                "effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh"],
                           "description": "Reasoning effort level. Leave empty to keep current."},
            }, "required": []},
        }, _switch_model),
        ToolEntry("get_task_result", {
            "name": "get_task_result",
            "description": "Read the result of a completed subtask. Use after schedule_task to collect results.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID returned by schedule_task"},
            }},
        }, _get_task_result),
        ToolEntry("wait_for_task", {
            "name": "wait_for_task",
            "description": "Check if a subtask has completed. Returns result if done, or 'still running' message. Call repeatedly to poll. Default timeout: 120s.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID to check"},
            }},
        }, _wait_for_task),
    ]
