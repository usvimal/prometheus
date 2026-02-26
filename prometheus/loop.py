"""
Ouroboros — LLM tool loop.

Core loop: send messages to LLM, execute tool calls, repeat until final response.
Extracted from agent.py to keep the agent thin.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from prometheus.llm import LLMClient, normalize_reasoning_effort, add_usage
from prometheus.tools.registry import ToolRegistry
from prometheus.context import compact_tool_history, compact_tool_history_llm
from prometheus.utils import utc_now_iso, append_jsonl, truncate_for_log, sanitize_tool_args_for_log, sanitize_tool_result_for_log, estimate_tokens

log = logging.getLogger(__name__)

# Pricing from OpenRouter API (2026-02-17). Update periodically via /api/v1/models.
_MODEL_PRICING_STATIC = {
    "anthropic/claude-opus-4.6": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4": (15.0, 1.5, 75.0),
    "anthropic/claude-sonnet-4": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4.6": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4.5": (3.0, 0.30, 15.0),
    "openai/o3": (2.0, 0.50, 8.0),
    "openai/o3-pro": (20.0, 1.0, 80.0),
    "openai/o4-mini": (1.10, 0.275, 4.40),
    "openai/gpt-4.1": (2.0, 0.50, 8.0),
    "openai/gpt-5.2": (1.75, 0.175, 14.0),
    "openai/gpt-5.2-codex": (1.75, 0.175, 14.0),
    "google/gemini-2.5-pro-preview": (1.25, 0.125, 10.0),
    "google/gemini-3-pro-preview": (2.0, 0.20, 12.0),
    "x-ai/grok-3-mini": (0.30, 0.03, 0.50),
    "qwen/qwen3.5-plus-02-15": (0.40, 0.04, 2.40),
}

_pricing_fetched = False
_cached_pricing = None
_pricing_lock = threading.Lock()

def _get_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Lazy-load pricing. On first call, attempts to fetch from OpenRouter API.
    Falls back to static pricing if fetch fails.
    Thread-safe via module-level lock.
    """
    global _pricing_fetched, _cached_pricing

    # Fast path: already fetched (read without lock for performance)
    if _pricing_fetched:
        return _cached_pricing or _MODEL_PRICING_STATIC

    # Slow path: fetch pricing (lock required)
    with _pricing_lock:
        # Double-check after acquiring lock (another thread may have fetched)
        if _pricing_fetched:
            return _cached_pricing or _MODEL_PRICING_STATIC

        _pricing_fetched = True
        _cached_pricing = dict(_MODEL_PRICING_STATIC)

        try:
            from prometheus.llm import fetch_openrouter_pricing
            _live = fetch_openrouter_pricing()
            if _live and len(_live) > 5:
                _cached_pricing.update(_live)
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("Failed to sync pricing from OpenRouter: %s", e)
            # Reset flag so we retry next time
            _pricing_fetched = False

        return _cached_pricing

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int,
                   cached_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Estimate cost from token counts using known pricing. Returns 0 if model unknown."""
    model_pricing = _get_pricing()
    # Try exact match first
    pricing = model_pricing.get(model)
    if not pricing:
        # Try longest prefix match
        best_match = None
        best_length = 0
        for key, val in model_pricing.items():
            if model and model.startswith(key):
                if len(key) > best_length:
                    best_match = val
                    best_length = len(key)
        pricing = best_match
    if not pricing:
        return 0.0
    input_price, cached_price, output_price = pricing
    # Non-cached input tokens = prompt_tokens - cached_tokens
    regular_input = max(0, prompt_tokens - cached_tokens)
    cost = (
        regular_input * input_price / 1_000_000
        + cached_tokens * cached_price / 1_000_000
        + completion_tokens * output_price / 1_000_000
    )
    return round(cost, 6)

READ_ONLY_PARALLEL_TOOLS = frozenset({
    "repo_read", "repo_list",
    "drive_read", "drive_list",
    "web_search", "codebase_digest", "chat_history",
})

# Stateful browser tools require thread-affinity (Playwright sync uses greenlet)
STATEFUL_BROWSER_TOOLS = frozenset({"browse_page", "browser_action"})


def _truncate_tool_result(result: Any) -> str:
    """
    Hard-cap tool result string to 15000 characters.
    If truncated, append a note with the original length.
    """
    result_str = str(result)
    if len(result_str) <= 15000:
        return result_str
    original_len = len(result_str)
    return result_str[:15000] + f"\n... (truncated from {original_len} chars)"


def _execute_single_tool(
    tools: ToolRegistry,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    task_id: str = "",
) -> Dict[str, Any]:
    """
    Execute a single tool call and return all needed info.

    Returns dict with: tool_call_id, fn_name, result, is_error, args_for_log, is_code_tool
    """
    fn_name = tc["function"]["name"]
    tool_call_id = tc["id"]
    is_code_tool = fn_name in tools.CODE_TOOLS

    # Parse arguments
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, ValueError) as e:
        result = f"⚠️ TOOL_ARG_ERROR: Could not parse arguments for '{fn_name}': {e}"
        return {
            "tool_call_id": tool_call_id,
            "fn_name": fn_name,
            "result": result,
            "is_error": True,
            "args_for_log": {},
            "is_code_tool": is_code_tool,
        }

    args_for_log = sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {})

    # Execute tool
    tool_ok = True
    try:
        result = tools.execute(fn_name, args)
    except Exception as e:
        tool_ok = False
        result = f"⚠️ TOOL_ERROR ({fn_name}): {type(e).__name__}: {e}"
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "tool_error", "task_id": task_id,
            "tool": fn_name, "args": args_for_log, "error": repr(e),
        })

    # Log tool execution (sanitize secrets from result before persisting)
    append_jsonl(drive_logs / "tools.jsonl", {
        "ts": utc_now_iso(), "tool": fn_name, "task_id": task_id,
        "args": args_for_log,
        "result_preview": sanitize_tool_result_for_log(truncate_for_log(result, 2000)),
    })

    is_error = (not tool_ok) or str(result).startswith("⚠️")

    return {
        "tool_call_id": tool_call_id,
        "fn_name": fn_name,
        "result": result,
        "is_error": is_error,
        "args_for_log": args_for_log,
        "is_code_tool": is_code_tool,
    }


class _StatefulToolExecutor:
    """
    Thread-sticky executor for stateful tools (browser, etc).

    Playwright sync API uses greenlet internally which has strict thread-affinity:
    once a greenlet starts in a thread, all subsequent calls must happen in the same thread.
    This executor ensures browse_page/browser_action always run in the same thread.

    On timeout: we shutdown the executor and create a fresh one to reset state.
    """
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None

    def submit(self, fn, *args, **kwargs):
        """Submit work to the sticky thread. Creates executor on first call."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stateful_tool")
        return self._executor.submit(fn, *args, **kwargs)

    def reset(self):
        """Shutdown current executor and create a fresh one. Used after timeout/error."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def shutdown(self, wait=True, cancel_futures=False):
        """Final cleanup."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None


def _make_timeout_result(
    fn_name: str,
    tool_call_id: str,
    is_code_tool: bool,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    timeout_sec: int,
    task_id: str = "",
    reset_msg: str = "",
) -> Dict[str, Any]:
    """
    Create a timeout error result dictionary and log the timeout event.

    Args:
        reset_msg: Optional additional message (e.g., "Browser state has been reset. ")

    Returns: Dict with tool_call_id, fn_name, result, is_error, args_for_log, is_code_tool
    """
    args_for_log = {}
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
        args_for_log = sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {})
    except Exception:
        pass

    result = (
        f"⚠️ TOOL_TIMEOUT ({fn_name}): exceeded {timeout_sec}s limit. "
        f"The tool is still running in background but control is returned to you. "
        f"{reset_msg}Try a different approach or inform the owner{' about the issue' if not reset_msg else ''}."
    )

    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "tool_timeout",
        "tool": fn_name, "args": args_for_log,
        "timeout_sec": timeout_sec,
    })
    append_jsonl(drive_logs / "tools.jsonl", {
        "ts": utc_now_iso(), "tool": fn_name,
        "args": args_for_log, "result_preview": result,
    })

    return {
        "tool_call_id": tool_call_id,
        "fn_name": fn_name,
        "result": result,
        "is_error": True,
        "args_for_log": args_for_log,
        "is_code_tool": is_code_tool,
    }


def _execute_with_timeout(
    tools: ToolRegistry,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    timeout_sec: int,
    task_id: str = "",
    stateful_executor: Optional[_StatefulToolExecutor] = None,
) -> Dict[str, Any]:
    """
    Execute a tool call with a hard timeout.

    On timeout: returns TOOL_TIMEOUT error so the LLM regains control.
    For stateful tools (browser): resets the sticky executor to recover state.
    For regular tools: the hung worker thread leaks as daemon — watchdog handles recovery.
    """
    fn_name = tc["function"]["name"]
    tool_call_id = tc["id"]
    is_code_tool = fn_name in tools.CODE_TOOLS
    use_stateful = stateful_executor and fn_name in STATEFUL_BROWSER_TOOLS

    # Two distinct paths: stateful (thread-sticky) vs regular (per-call)
    if use_stateful:
        # Stateful executor: submit + wait, reset on timeout
        future = stateful_executor.submit(_execute_single_tool, tools, tc, drive_logs, task_id)
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            stateful_executor.reset()
            reset_msg = "Browser state has been reset. "
            return _make_timeout_result(
                fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                timeout_sec, task_id, reset_msg
            )
    else:
        # Regular executor: explicit lifecycle to avoid shutdown(wait=True) deadlock
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_execute_single_tool, tools, tc, drive_logs, task_id)
            try:
                return future.result(timeout=timeout_sec)
            except TimeoutError:
                return _make_timeout_result(
                    fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                    timeout_sec, task_id, reset_msg=""
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def _handle_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: ToolRegistry,
    drive_logs: pathlib.Path,
    task_id: str,
    stateful_executor: _StatefulToolExecutor,
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """
    Execute tool calls and append results to messages.

    Returns: Number of errors encountered
    """
    # Parallelize only for a strict read-only whitelist; all calls wrapped with timeout.
    can_parallel = (
        len(tool_calls) > 1 and
        all(
            tc.get("function", {}).get("name") in READ_ONLY_PARALLEL_TOOLS
            for tc in tool_calls
        )
    )

    if not can_parallel:
        results = [
            _execute_with_timeout(tools, tc, drive_logs,
                                  tools.get_timeout(tc["function"]["name"]), task_id,
                                  stateful_executor)
            for tc in tool_calls
        ]
    else:
        max_workers = min(len(tool_calls), 8)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_index = {
                executor.submit(
                    _execute_with_timeout, tools, tc, drive_logs,
                    tools.get_timeout(tc["function"]["name"]), task_id,
                    stateful_executor,
                ): idx
                for idx, tc in enumerate(tool_calls)
            }
            results = [None] * len(tool_calls)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # Process results in original order
    return _process_tool_results(results, messages, llm_trace, emit_progress)


def _handle_text_response(
    content: Optional[str],
    llm_trace: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Handle LLM response without tool calls (final response).

    Returns: (final_text, accumulated_usage, llm_trace)
    """
    if content and content.strip():
        llm_trace["assistant_notes"].append(content.strip()[:320])
    return (content or ""), accumulated_usage, llm_trace


def _check_budget_limits(
    budget_remaining_usd: Optional[float],
    accumulated_usage: Dict[str, Any],
    round_idx: int,
    messages: List[Dict[str, Any]],
    llm: LLMClient,
    active_model: str,
    active_effort: str,
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    event_queue: Optional[queue.Queue],
    llm_trace: Dict[str, Any],
    task_type: str = "task",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Check budget limits and handle budget overrun.

    Returns:
        None if budget is OK (continue loop)
        (final_text, accumulated_usage, llm_trace) if budget exceeded (stop loop)
    """
    if budget_remaining_usd is None:
        return None

    task_cost = accumulated_usage.get("cost", 0)
    budget_pct = task_cost / budget_remaining_usd if budget_remaining_usd > 0 else 1.0

    if budget_pct > 0.5:
        # Hard stop — protect the budget
        finish_reason = f"Task spent ${task_cost:.2f} ({budget_pct*100:.0f}% of budget). Stopping to protect remaining funds."
        llm_trace["finish_reason"] = finish_reason
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "budget_exceeded",
            "task_id": task_id, "task_type": task_type,
            "task_cost": task_cost, "budget_remaining": budget_remaining_usd,
            "budget_pct": budget_pct,
        })
        return (finish_reason, accumulated_usage, llm_trace)

    if budget_pct > 0.25:
        # Soft warning at 25%
        warning = f"⚠️ Task has spent ${task_cost:.2f} ({budget_pct*100:.0f}% of budget)"
        log.warning(warning)
        if event_queue:
            event_queue.put(("progress", warning))
        llm_trace["budget_warning"] = warning

    return None


def _check_empty_response(
    response: Optional[Dict[str, Any]],
    retry_count: int,
    max_retries: int,
    active_model: str,
    fallback_models: List[str],
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    drive_logs: pathlib.Path,
    task_id: str,
    emit_progress: Callable[[str], None],
) -> Tuple[Optional[Dict[str, Any]], int, str]:
    """
    Check for empty model response and handle fallback/retry.

    Returns: (response, retry_count, active_model)
    """
    if response and response.get("content"):
        return response, retry_count, active_model

    # Empty response detected
    retry_count += 1
    log.warning(f"Empty response from {active_model} (retry {retry_count}/{max_retries})")
    emit_progress(f"⚠️ Empty response from {active_model}, retry {retry_count}/{max_retries}")

    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "empty_response",
        "task_id": task_id, "model": active_model,
        "retry": retry_count,
    })

    # Retry with same model
    if retry_count < max_retries:
        return None, retry_count, active_model

    # Try fallback models
    for fallback in fallback_models:
        if fallback == active_model:
            continue
        log.warning(f"Empty response from {active_model}, trying fallback {fallback}")
        emit_progress(f"⚠️ {active_model} failed, trying {fallback}...")

        try:
            response = llm.chat(
                model=fallback,
                messages=messages,
                tools=None,
            )
            if response and response.get("content"):
                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "fallback_success",
                    "task_id": task_id, "original_model": active_model,
                    "fallback_model": fallback,
                })
                return response, 0, fallback
        except Exception as e:
            log.warning(f"Fallback {fallback} also failed: {e}")

    # All failed
    return None, retry_count, active_model


def _process_tool_results(
    results: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """
    Process tool results and append to messages.

    Returns: Number of errors encountered
    """
    error_count = 0
    for r in results:
        tc_result = {
            "role": "tool",
            "tool_call_id": r["tool_call_id"],
            "content": _truncate_tool_result(r["result"]),
        }
        messages.append(tc_result)

        if r["is_error"]:
            error_count += 1
            err_msg = f"Tool error: {r['fn_name']}: {r['result'][:200]}"
            log.warning(err_msg)
            emit_progress(err_msg)
            llm_trace["tool_errors"].append({
                "tool": r["fn_name"],
                "error": r["result"][:200],
            })
        else:
            llm_trace["tool_results"].append({
                "tool": r["fn_name"],
                "result_preview": r["result"][:200],
            })

    return error_count



def _run_loop_iteration(
    llm: LLMClient,
    tools: ToolRegistry,
    messages: List[Dict[str, Any]],
    task_id: str,
    task_type: str,
    budget_remaining_usd: Optional[float],
    max_rounds: int,
    max_retries: int,
    emit_progress: Callable[[str], None],
    drive_logs: pathlib.Path,
    event_queue: Optional[queue.Queue],
    active_model: str,
    active_effort: str,
    fallback_models: List[str],
    round_idx: int,
    retry_count: int,
    accumulated_usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    stateful_executor: _StatefulToolExecutor,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], int, int, str, str]:
    """
    Execute one iteration of the LLM tool loop.
    
    Returns:
        (final_text, usage, trace, new_round_idx, new_retry_count, next_model, finish_reason)
        - If final_text is not None: task is complete (return this result)
        - If final_text is None: continue to next round (use returned state)
    """
    round_idx += 1
    llm_trace["rounds"].append(round_idx)

    # Soft self-check at 50/100/150
    if round_idx in (50, 100, 150):
        log.info(f"Self-check at round {round_idx}")

    # Check budget
    budget_check = _check_budget_limits(
        budget_remaining_usd, accumulated_usage, round_idx,
        messages, llm, active_model, active_effort,
        max_retries, drive_logs, task_id, event_queue, llm_trace, task_type,
    )
    if budget_check:
        return budget_check[0], budget_check[1], budget_check[2], round_idx, retry_count, active_model, "budget"

    # Emit progress
    round_msg = f"[{task_type[:3].upper()}] {task_id[:8]} round {round_idx}/{max_rounds}"
    if round_idx % 10 == 0:
        emit_progress(round_msg)
    log.info(round_msg)

    # Compact old tool history when needed
    pending_compaction = getattr(tools._ctx, '_pending_compaction', None)
    if pending_compaction is not None:
        messages = compact_tool_history_llm(messages, keep_recent=pending_compaction)
        tools._ctx._pending_compaction = None
    elif round_idx > 6:
        messages = compact_tool_history(messages, keep_recent=8)
    elif round_idx > 2:
        if len(messages) > 40:
            messages = compact_tool_history(messages, keep_recent=8)

    # Call LLM
    try:
        response = llm.chat(
            model=active_model,
            messages=messages,
            tools=tools.get_schemas(),
            reasoning_effort=active_effort,
        )
    except Exception as e:
        err_msg = f"LLM error: {type(e).__name__}: {e}"
        log.error(err_msg)
        emit_progress(f"⚠️ {err_msg}")
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "llm_error",
            "task_id": task_id, "error": repr(e),
        })
        llm_trace["llm_errors"] = llm_trace.get("llm_errors", []) + [repr(e)]

        retry_count += 1
        if retry_count <= max_retries:
            return None, None, None, round_idx, retry_count, active_model, "retry"
        else:
            final_text = f"⚠️ LLM error after {max_retries} retries: {e}"
            llm_trace["finish_reason"] = "llm_error"
            return final_text, accumulated_usage, llm_trace, round_idx, retry_count, active_model, "llm_error"

    # Check for empty response
    response, retry_count, active_model = _check_empty_response(
        response, retry_count, max_retries, active_model,
        fallback_models, llm, messages, accumulated_usage,
        drive_logs, task_id, emit_progress,
    )

    if response is None:
        final_text = "⚠️ Failed to get a response from the model after 3 attempts. Fallback models also returned no response."
        llm_trace["finish_reason"] = "empty_response"
        emit_progress(final_text)
        return final_text, accumulated_usage, llm_trace, round_idx, retry_count, active_model, "empty_response"

    # Extract response
    response = llm.normalize_response(response)
    content = llm.extract_content(response)
    tool_calls = llm.extract_tool_calls(response)

    # Track usage
    usage = llm.extract_usage(response)
    if usage:
        add_usage(accumulated_usage, usage)
        usage_cost = _estimate_cost(
            active_model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("cached_tokens", 0),
        )
        accumulated_usage["cost"] += usage_cost

        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "llm_usage",
            "task_id": task_id, "round": round_idx,
            "model": active_model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost": usage_cost,
        })

    # Handle response
    if tool_calls:
        error_count = _handle_tool_calls(
            tool_calls, tools, drive_logs, task_id,
            stateful_executor, messages, llm_trace, emit_progress,
        )

        if error_count:
            llm_trace["round_errors"] = llm_trace.get("round_errors", 0) + error_count
            log.warning(f"Round {round_idx}: {error_count} tool errors")

        # Check budget after tools too
        budget_check = _check_budget_limits(
            budget_remaining_usd, accumulated_usage, round_idx,
            messages, llm, active_model, active_effort,
            max_retries, drive_logs, task_id, event_queue, llm_trace, task_type,
        )
        if budget_check:
            return budget_check[0], budget_check[1], budget_check[2], round_idx, retry_count, active_model, "budget"

        return None, None, None, round_idx, retry_count, active_model, "tools"

    # Final response (no tools)
    final_text, accumulated_usage, llm_trace = _handle_text_response(
        content, llm_trace, accumulated_usage,
    )
    llm_trace["finish_reason"] = "stop"
    return final_text, accumulated_usage, llm_trace, round_idx, retry_count, active_model, "stop"


def run_loop(
    llm: LLMClient,
    tools: ToolRegistry,
    messages: List[Dict[str, Any]],
    task_id: str,
    task_type: str = "task",
    budget_remaining_usd: Optional[float] = None,
    max_rounds: int = 200,
    max_retries: int = 3,
    emit_progress: Optional[Callable[[str], None]] = None,
    drive_logs: Optional[pathlib.Path] = None,
    event_queue: Optional[queue.Queue] = None,
    active_model: Optional[str] = None,
    active_effort: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Core LLM tool loop.

    Args:
        llm: LLM client instance
        tools: Tool registry
        messages: Conversation messages (modified in place)
        task_id: Task identifier
        task_type: Type of task (task/evolution/direct)
        budget_remaining_usd: Budget cap in USD (None = unlimited)
        max_rounds: Maximum LLM rounds
        max_retries: Max retries on empty response
        emit_progress: Callback for progress messages
        drive_logs: Path to logs directory
        event_queue: Queue for supervisor events
        active_model: Override model (None = use llm.model)
        active_effort: Reasoning effort (None = default)

    Returns:
        (final_text, usage_dict, llm_trace)
    """
    if emit_progress is None:
        emit_progress = lambda x: None
    if drive_logs is None:
        drive_logs = pathlib.Path("/home/vimal2/prometheus/data/logs")

    fallback_models = os.environ.get("OUROBOROS_MODEL_FALLBACK_LIST", "").split(",")
    fallback_models = [m.strip() for m in fallback_models if m.strip()]

    # State
    round_idx = 0
    retry_count = 0
    if active_model is None:
        active_model = llm.default_model()
    active_effort = normalize_reasoning_effort(active_effort)

    # Trace for observability
    llm_trace: Dict[str, Any] = {
        "task_id": task_id,
        "task_type": task_type,
        "model": active_model,
        "rounds": [],
        "tool_results": [],
        "tool_errors": [],
        "assistant_notes": [],
        "finish_reason": None,
    }

    accumulated_usage: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
    }

    # Thread-sticky executor for stateful tools
    stateful_executor = _StatefulToolExecutor()

    try:
        while round_idx < max_rounds:
            final_text, usage, trace, round_idx, retry_count, active_model, finish_reason = _run_loop_iteration(
                llm, tools, messages, task_id, task_type,
                budget_remaining_usd, max_rounds, max_retries,
                emit_progress, drive_logs, event_queue,
                active_model, active_effort, fallback_models,
                round_idx, retry_count, accumulated_usage, llm_trace,
                stateful_executor,
            )
            
            # Check if task completed
            if final_text is not None:
                return final_text, usage or accumulated_usage, trace or llm_trace
            
            # Handle retry state
            if finish_reason == "retry":
                continue
        
        # Max rounds reached
        final_text = f"⚠️ Max rounds ({max_rounds}) reached. Task stopped."
        llm_trace["finish_reason"] = "max_rounds"
        emit_progress(final_text)
        return final_text, accumulated_usage, llm_trace

    finally:
        stateful_executor.shutdown(wait=False, cancel_futures=True)


