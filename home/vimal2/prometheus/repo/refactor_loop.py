#!/usr/bin/env python3
"""Refactor run_loop to reduce line count."""

# Read the file
with open('/home/vimal2/prometheus/repo/prometheus/loop.py', 'r') as f:
    lines = f.readlines()

# Find run_loop function boundaries
run_loop_start = None
for i, line in enumerate(lines):
    if line.strip().startswith('def run_loop('):
        run_loop_start = i
        break

# Find where run_loop ends
run_loop_end = len(lines)
for i in range(run_loop_start + 1, len(lines)):
    if lines[i].strip().startswith('def ') and not lines[i].strip().startswith('def run_loop'):
        run_loop_end = i
        break

print(f'run_loop: lines {run_loop_start+1}-{run_loop_end+1} ({run_loop_end - run_loop_start} lines)')

# Extract the function
run_loop = lines[run_loop_start:run_loop_end]

# Create _init_loop_state helper
init_helper = '''def _init_loop_state(
    llm,
    task_id: str,
    task_type: str,
    budget_remaining_usd: Optional[float],
    max_rounds: int,
    emit_progress: Optional[Callable],
    drive_logs: pathlib.Path,
    active_model: Optional[str],
    active_effort: Optional[str],
):
    """Initialize loop state variables."""
    if emit_progress is None:
        emit_progress = lambda x: None
    if drive_logs is None:
        drive_logs = pathlib.Path("/home/vimal2/prometheus/data/logs")

    fallback_models = os.environ.get("OUROBOROS_MODEL_FALLBACK_LIST", "").split(",")
    fallback_models = [m.strip() for m in fallback_models if m.strip()]

    round_idx = 0
    retry_count = 0
    if active_model is None:
        active_model = llm.model
    active_effort = normalize_reasoning_effort(active_effort)

    llm_trace = {
        "task_id": task_id,
        "task_type": task_type,
        "model": active_model,
        "rounds": [],
        "tool_results": [],
        "tool_errors": [],
        "assistant_notes": [],
        "finish_reason": None,
    }

    accumulated_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
    }

    stateful_executor = _StatefulToolExecutor()
    
    return (round_idx, retry_count, active_model, active_effort, 
            llm_trace, accumulated_usage, stateful_executor, fallback_models)


'''

# Now create the shortened run_loop
shortened_run_loop = '''def run_loop(
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
    if drive_logs is None:
        drive_logs = pathlib.Path("/home/vimal2/prometheus/data/logs")

    # Initialize state
    (round_idx, retry_count, active_model, active_effort, 
     llm_trace, accumulated_usage, stateful_executor, fallback_models) = _init_loop_state(
        llm, task_id, task_type, budget_remaining_usd, max_rounds,
        emit_progress, drive_logs, active_model, active_effort,
    )

    try:
        while round_idx < max_rounds:
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
                return budget_check

            # Emit progress
            round_msg = f"[{task_type[:3].upper()}] {task_id[:8]} round {round_idx}/{max_rounds}"
            if round_idx % 10 == 0 and emit_progress:
                emit_progress(round_msg)
            log.info(round_msg)

            # Compact old tool history when needed
            pending_compaction = getattr(tools._ctx, '_pending_compaction', None)
            if pending_compaction is not None:
                messages = compact_tool_history_llm(messages, keep_recent=pending_compaction)
                tools._ctx._pending_compaction = None
            elif round_idx > 6:
                messages = compact_tool_history(messages, keep_recent=8)
            elif round_idx > 2 and len(messages) > 40:
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
                if emit_progress:
                    emit_progress(f"⚠️ {err_msg}")
                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "llm_error",
                    "task_id": task_id, "error": repr(e),
                })
                llm_trace["llm_errors"] = llm_trace.get("llm_errors", []) + [repr(e)]
                retry_count += 1
                if retry_count <= max_retries:
                    continue
                else:
                    final_text = f"⚠️ LLM error after {max_retries} retries: {e}"
                    llm_trace["finish_reason"] = "llm_error"
                    return final_text, accumulated_usage, llm_trace

            # Check for empty response
            response, retry_count, active_model = _check_empty_response(
                response, retry_count, max_retries, active_model,
                fallback_models, llm, messages, accumulated_usage,
                drive_logs, task_id, emit_progress,
            )

            if response is None:
                final_text = "⚠️ Failed to get a response from the model after 3 attempts."
                llm_trace["finish_reason"] = "empty_response"
                if emit_progress:
                    emit_progress(final_text)
                return final_text, accumulated_usage, llm_trace

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

                budget_check = _check_budget_limits(
                    budget_remaining_usd, accumulated_usage, round_idx,
                    messages, llm, active_model, active_effort,
                    max_retries, drive_logs, task_id, event_queue, llm_trace, task_type,
                )
                if budget_check:
                    return budget_check
                continue

            # Final response (no tools)
            final_text, accumulated_usage, llm_trace = _handle_text_response(
                content, llm_trace, accumulated_usage,
            )
            llm_trace["finish_reason"] = "stop"
            return final_text, accumulated_usage, llm_trace

        # Max rounds reached
        final_text = f"⚠️ Max rounds ({max_rounds}) reached. Task stopped."
        llm_trace["finish_reason"] = "max_rounds"
        if emit_progress:
            emit_progress(final_text)
        return final_text, accumulated_usage, llm_trace

    finally:
        stateful_executor.shutdown(wait=False, cancel_futures=True)


'''

# Write new file
with open('/home/vimal2/prometheus/repo/prometheus/loop_new.py', 'w') as f:
    # Write lines before run_loop
    for line in lines[:run_loop_start]:
        f.write(line)
    # Write the helper
    f.write(init_helper)
    # Write shortened run_loop
    f.write(shortened_run_loop)

print("New file written to loop_new.py")
print(f"Original run_loop was {run_loop_end - run_loop_start} lines")
print(f"New run_loop is {len(shortened_run_loop.splitlines())} lines")
print(f"Helper is {len(init_helper.splitlines())} lines")
