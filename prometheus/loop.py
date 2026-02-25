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
import re
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


def _repair_json(malformed: str, fn_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Attempt to repair malformed JSON from LLM tool arguments.
    
    Strategies tried in order:
    1. Strip trailing commas before closing braces/brackets
    2. Remove control characters and whitespace issues
    3. Fix unquoted property names (simple cases)
    4. Handle truncated JSON (find valid prefix)
    5. Strip markdown code blocks
    
    Returns repaired dict or None if all strategies fail.
    """
    if not malformed:
        return {}
    
    original = malformed
    repair_strategy = "none"
    
    # Strategy 1: Strip markdown code blocks
    cleaned = malformed.strip()
    if cleaned.startswith("```"):
        # Remove ```json or ```python or ``` blocks
        lines = cleaned.split('\n')
        if lines[0].strip().startswith("```"):
            lines = lines[1:]  # Remove first line with ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove last line with ```
        cleaned = '\n'.join(lines)
        repair_strategy = "strip_markdown"
    
    # Strategy 2: Strip trailing commas (common LLM mistake)
    # Pattern: comma followed by } or ]
    fixed = re.sub(r',([\s]*[}\]])', r'\1', cleaned)
    if fixed != cleaned:
        repair_strategy = "trailing_comma"
        cleaned = fixed
    
    # Strategy 3: Remove control characters
    fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
    if fixed != cleaned:
        repair_strategy = "control_chars"
        cleaned = fixed
    
    # Strategy 4: Fix unquoted simple keys (e.g., {key: "value"})
    # Only handles alphanumeric keys, not reserved words
    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    if fixed != cleaned:
        repair_strategy = "unquoted_keys"
        cleaned = fixed
    
    # Strategy 5: Handle truncated JSON - find valid prefix
    # Try to close open braces/brackets
    if not cleaned.startswith("{"):
        # Not a dict, can't repair
        pass
    else:
        stack = []
        i = 0
        while i < len(cleaned):
            c = cleaned[i]
            if c == '{':
                stack.append('}')
            elif c == '[':
                stack.append(']')
            elif c == '}' and stack and stack[-1] == '}':
                stack.pop()
            elif c == ']' and stack and stack[-1] == ']':
                stack.pop()
            elif c == '"' and (i == 0 or cleaned[i-1] != '\\'):
                # Skip string
                i += 1
                while i < len(cleaned):
                    if cleaned[i] == '"' and cleaned[i-1] != '\\':
                        break
                    i += 1
            i += 1
        
        if stack:
            # Close remaining brackets
            while stack:
                cleaned += stack.pop()
            repair_strategy = "close_brackets"
    
    # Try parsing the repaired JSON
    try:
        parsed = json.loads(cleaned)
        if repair_strategy != "none":
            log.info(f"JSON repair succeeded for {fn_name}: strategy={repair_strategy}")
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Final fallback: try to extract just the arguments we care about
    # This is a desperate measure for severely broken JSON
    try:
        # Look for key-value patterns like "key": value
        kv_pattern = r'"([^"]+)"\s*:\s*("[^"]*"|[\d.\-]+|\{[^}]*\}|\[[^\]]*\])'
        matches = re.findall(kv_pattern, cleaned)
        if matches:
            result = {}
            for key, val in matches:
                try:
                    # Try to parse the value as JSON
                    result[key] = json.loads(val)
                except:
                    # Use as string
                    result[key] = val.strip('"')
            if result:
                log.info(f"JSON repair fallback succeeded for {fn_name}: extracted {len(result)} keys")
                return result
    except Exception as e:
        log.warning(f"JSON repair fallback failed for {fn_name}: {e}")
    
    log.warning(f"JSON repair failed for {fn_name}, original length={len(original)}")
    return None


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

    # Parse arguments - with JSON repair on failure
    args = None
    raw_args = tc["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, ValueError) as e:
        # Try to repair malformed JSON
        repaired = _repair_json(raw_args, fn_name)
        if repaired is not None:
            args = repaired
            log.info(f"Tool {fn_name}: used repaired JSON instead of failing")
        else:
            result = f"⚠️ TOOL_ARG_ERROR: Could not parse arguments for '{fn_name}': {e}. Also failed JSON repair."
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
        raw_args = tc["function"].get("arguments") or "{}"
        args = json.loads(raw_args)
        args = _repair_json(raw_args, fn_name) if args is None else args
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
                timeout_sec, task_id, reset    else:
       _msg
            )
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
        finish_reason = f"Task spent ${task_cost:.2f} of ${budget_remaining_usd:.2f} budget ({budget_pct*100:.0f}%)."
        log.warning(f"[{task_id}] Budget hard stop at round {round_idx}: {finish_reason}")
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "budget_hard_stop",
            "task_id": task_id, "task_type": task_type,
            "cost": task_cost, "budget": budget_remaining_usd,
            "round": round_idx,
        })
        return (finish_reason, accumulated_usage, llm_trace)

    if budget_pct > 0.25:
        # Soft warning — encourage the agent to wrap up
        warn_msg = f"⚠️ Budget warning: {budget_pct*100:.0f}% spent (${task_cost:.2f}/${budget_remaining_usd:.2f}). Consider wrapping up."
        if event_queue:
            event_queue.put({"type": "progress", "text": warn_msg})
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "budget_warning",
            "task_id": task_id, "task_type": task_type,
            "cost": task_cost, "budget": budget_remaining_usd,
            "pct": budget_pct, "round": round_idx,
        })

    return None


def _process_tool_results(
    results: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """
    Process tool results and add to messages.

    Returns: Number of errors encountered
    """
    error_count = 0
    for r in results:
        tool_call_id = r["tool_call_id"]
        fn_name = r["fn_name"]
        result = r["result"]
        is_error = r["is_error"]
        args_for_log = r["args_for_log"]

        if is_error:
            error_count += 1
            llm_trace["tool_errors"].append(fn_name)
        else:
            llm_trace["tool_successes"].append(fn_name)

        # Add tool result message to history
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result)[:32000],
        })

        # Emit progress for errors (not for every success to reduce noise)
        if is_error and emit_progress:
            emit_progress(f"Tool error: {fn_name} — {str(result)[:200]}")

    return error_count


def _extract_and_validate_tool_calls(
    llm: LLMClient,
    response: Dict[str, Any],
    fn_name: str,
) -> List[Dict[str, Any]]:
    """
    Extract tool calls from LLM response, with JSON repair on failure.
    """
    tool_calls = llm.extract_tool_calls(response)
    
    # If extraction failed, try to repair malformed tool_calls
    if not tool_calls and response.get("tool_calls"):
        raw_calls = response.get("tool_calls", [])
        repaired_calls = []
        for tc in raw_calls:
            if not tc.get("function", {}).get("arguments"):
                continue
            fn = tc.get("function", {}).get("name", "")
            raw_args = tc["function"]["arguments"]
            try:
                parsed = json.loads(raw_args)
            except:
                parsed = _repair_json(raw_args, fn)
            if parsed is not None:
                repaired_calls.append({
                    "id": tc.get("id", f"repaired_{fn}"),
                    "type": "function",
                    "function": {
                        "name": fn,
                        "arguments": json.dumps(parsed),
                    }
                })
        if repaired_calls:
            log.info(f"Repaired {len(repaired_calls)} tool calls from malformed response")
            tool_calls = repaired_calls
    
    return tool_calls


def _get_system_prompt(task_type: str, context_tokens: int, model: str) -> str:
    """
    Build system prompt with current task type and context info.
    """
    return f"""You are Ouroboros, a self-modifying AI agent. You are currently running task type: {task_type}.

Context: {context_tokens} tokens in the conversation. Model: {model}.
Your goal is to accomplish the task by using tools and reasoning.
"""
