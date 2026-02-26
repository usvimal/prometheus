"""
Ouroboros context builder.

Assembles LLM context from prompts, memory, logs, and runtime state.
Extracted from agent.py to keep the agent thin and focused.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

from prometheus.utils import (
    utc_now_iso, read_text, clip_text, estimate_tokens, get_git_info,
)
from prometheus.memory import Memory

log = logging.getLogger(__name__)


def _build_user_content(task: Dict[str, Any]) -> Any:
    """Build user message content. Supports text + optional image."""
    text = task.get("text", "")
    image_b64 = task.get("image_base64")
    image_mime = task.get("image_mime", "image/jpeg")
    image_caption = task.get("image_caption", "")

    if not image_b64:
        # Return fallback text if both text and image are empty
        if not text:
            return "(empty message)"
        return text

    # Multipart content with text + image
    parts = []
    # Combine caption and text for the text part
    combined_text = ""
    if image_caption:
        combined_text = image_caption
    if text and text != image_caption:
        combined_text = (combined_text + "\n" + text).strip() if combined_text else text

    # Always include a text part when there's an image
    if not combined_text:
        combined_text = "Analyze the screenshot"

    parts.append({"type": "text", "text": combined_text})
    parts.append({
        "type": "image_url",
        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}
    })
    return parts


def _build_runtime_section(env: Any, task: Dict[str, Any]) -> str:
    """Build the runtime context section (utc_now, repo_dir, drive_root, git_head, git_branch, task info, budget info)."""
    # --- Git context ---
    try:
        git_branch, git_sha = get_git_info(env.repo_dir)
    except Exception:
        log.debug("Failed to get git info for context", exc_info=True)
        git_branch, git_sha = "unknown", "unknown"

    # --- Budget calculation ---
    budget_info = None
    try:
        state_json = _safe_read(env.drive_path("state/state.json"), fallback="{}")
        state_data = json.loads(state_json)
        spent_usd = float(state_data.get("spent_usd", 0))
        total_usd = float(os.environ.get("TOTAL_BUDGET", "0"))
        if total_usd <= 0:
            # Subscription mode — show actual quota from MiniMax API
            from prometheus.llm import fetch_minimax_quota
            quota = fetch_minimax_quota()
            if quota:
                # Find the primary model's quota
                primary_model = os.environ.get("OUROBOROS_MODEL", "MiniMax-M2.5")
                model_quota = quota.get(primary_model) or next(iter(quota.values()), None)
                if model_quota:
                    budget_info = {
                        "mode": "subscription",
                        "calls_remaining": model_quota["remaining"],
                        "calls_total": model_quota["total"],
                        "calls_used": model_quota["used"],
                        "window_resets_in_sec": model_quota["window_remaining_sec"],
                    }
                else:
                    budget_info = {"mode": "subscription", "limit": "unlimited"}
            else:
                budget_info = {"mode": "subscription", "limit": "unlimited"}
        else:
            remaining_usd = total_usd - spent_usd
            budget_info = {"total_usd": total_usd, "spent_usd": spent_usd, "remaining_usd": remaining_usd}
    except Exception:
        log.debug("Failed to calculate budget info for context", exc_info=True)
        pass

    # --- Runtime context JSON ---
    runtime_data = {
        "utc_now": utc_now_iso(),
        "repo_dir": str(env.repo_dir),
        "drive_root": str(env.drive_root),
        "git_head": git_sha,
        "git_branch": git_branch,
        "task": {"id": task.get("id"), "type": task.get("type")},
    }
    if budget_info:
        runtime_data["budget"] = budget_info
    runtime_ctx = json.dumps(runtime_data, ensure_ascii=False, indent=2)
    return "## Runtime context\n\n" + runtime_ctx


def _build_memory_sections(memory: Memory) -> List[str]:
    """Build scratchpad, identity, dialogue summary sections."""
    sections = []

    scratchpad_raw = memory.load_scratchpad()
    sections.append("## Scratchpad\n\n" + clip_text(scratchpad_raw, 90000))

    identity_raw = memory.load_identity()
    sections.append("## Identity\n\n" + clip_text(identity_raw, 80000))

    # Dialogue summary (key moments from chat history)
    summary_path = memory.drive_root / "memory" / "dialogue_summary.md"
    if summary_path.exists():
        summary_text = read_text(summary_path)
        if summary_text.strip():
            sections.append("## Dialogue Summary\n\n" + clip_text(summary_text, 20000))

    return sections


def _build_recent_sections(memory: Memory, env: Any, task_id: str = "") -> List[str]:
    """Build recent chat, recent progress, recent tools, recent events sections."""
    sections = []

    chat_summary = memory.summarize_chat(
        memory.read_jsonl_tail("chat.jsonl", 200))
    if chat_summary:
        sections.append("## Recent chat\n\n" + chat_summary)

    progress_entries = memory.read_jsonl_tail("progress.jsonl", 200)
    if task_id:
        progress_entries = [e for e in progress_entries if e.get("task_id") == task_id]
    progress_summary = memory.summarize_progress(progress_entries, limit=15)
    if progress_summary:
        sections.append("## Recent progress\n\n" + progress_summary)

    tools_entries = memory.read_jsonl_tail("tools.jsonl", 200)
    if task_id:
        tools_entries = [e for e in tools_entries if e.get("task_id") == task_id]
    tools_summary = memory.summarize_tools(tools_entries)
    if tools_summary:
        sections.append("## Recent tools\n\n" + tools_summary)

    events_entries = memory.read_jsonl_tail("events.jsonl", 200)
    if task_id:
        events_entries = [e for e in events_entries if e.get("task_id") == task_id]
    events_summary = memory.summarize_events(events_entries)
    if events_summary:
        sections.append("## Recent events\n\n" + events_summary)

    supervisor_summary = memory.summarize_supervisor(
        memory.read_jsonl_tail("supervisor.jsonl", 200))
    if supervisor_summary:
        sections.append("## Supervisor\n\n" + supervisor_summary)

    return sections


def _build_health_invariants(env: Any) -> str:
    """Build health invariants section for LLM-first self-detection.

    Surfaces anomalies as informational text. The LLM (not code) decides
    what action to take based on what it reads here. (Bible P0+P3)
    """
    checks = []

    # 1. Version sync: VERSION file vs pyproject.toml vs README.md
    try:
        ver_file = read_text(env.repo_path("VERSION")).strip()
        pyproject = read_text(env.repo_path("pyproject.toml"))
        pyproject_ver = ""
        for line in pyproject.splitlines():
            if line.strip().startswith("version"):
                pyproject_ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        
        # Also check README.md for version (format: **Version:** X.Y.Z)
        readme = read_text(env.repo_path("README.md"))
        readme_ver = ""
        match = re.search(r'\*\*Version:\*\*\s*(\d+\.\d+\.\d+)', readme)
        if match:
            readme_ver = match.group(1)
        
        # Build version status
        versions = {"VERSION": ver_file}
        if pyproject_ver:
            versions["pyproject.toml"] = pyproject_ver
        if readme_ver:
            versions["README.md"] = readme_ver
        
        unique_versions = set(v for v in versions.values() if v)
        if len(unique_versions) > 1:
            version_details = ", ".join(f"{k}={v}" for k, v in versions.items() if v)
            checks.append(f"CRITICAL: VERSION DESYNC — {version_details}")
        elif ver_file:
            checks.append(f"OK: version sync ({ver_file})")
    except Exception:
        pass

    # 2. Budget drift (only relevant for pay-per-token mode)
    try:
        total_budget = float(os.environ.get("TOTAL_BUDGET", "0"))
        if total_budget > 0:
            state_json = read_text(env.drive_path("state/state.json"))
            state_data = json.loads(state_json)
            if state_data.get("budget_drift_alert"):
                drift_pct = state_data.get("budget_drift_pct", 0)
                our = state_data.get("spent_usd", 0)
                theirs = state_data.get("openrouter_total_usd", 0)
                checks.append(f"WARNING: BUDGET DRIFT {drift_pct:.1f}% — tracked=${our:.2f} vs OpenRouter=${theirs:.2f}")
            else:
                checks.append("OK: budget drift within tolerance")
        else:
            checks.append("OK: subscription mode (no budget limit)")
    except Exception:
        pass

    # 3. Per-task cost anomalies
    try:
        from supervisor.state import per_task_cost_summary
        costly = [t for t in per_task_cost_summary(5) if t["cost"] > 5.0]
        for t in costly:
            checks.append(
                f"WARNING: HIGH-COST TASK — task_id={t['task_id']} "
                f"cost=${t['cost']:.2f} rounds={t['rounds']}"
            )
        if not costly:
            checks.append("OK: no high-cost tasks (>$5)")
    except Exception:
        pass

    # 4. Stale identity.md
    try:
        import time as _time
        identity_path = env.drive_path("memory/identity.md")
        if identity_path.exists():
            age_hours = (_time.time() - identity_path.stat().st_mtime) / 3600
            if age_hours > 8:
                checks.append(f"WARNING: STALE IDENTITY — identity.md last updated {age_hours:.0f}h ago")
            else:
                checks.append("OK: identity.md recent")
    except Exception:
        pass

    # 5. Duplicate processing detection: same owner message text appearing in multiple tasks
    try:
        import hashlib
        msg_hash_to_tasks: Dict[str, set] = {}
        tail_bytes = 256_000

        def _scan_file_for_injected(path, type_field="type", type_value="owner_message_injected"):
            if not path.exists():
                return
            file_size = path.stat().st_size
            with path.open("r", encoding="utf-8") as f:
                if file_size > tail_bytes:
                    f.seek(file_size - tail_bytes)
                    f.readline()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        if ev.get(type_field) != type_value:
                            continue
                        text = ev.get("text", "")
                        if not text and "event_repr" in ev:
                            # Historical entries in supervisor.jsonl lack "text";
                            # try to extract task_id at least for presence detection
                            text = ev.get("event_repr", "")[:200]
                        if not text:
                            continue
                        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
                        tid = ev.get("task_id") or "unknown"
                        if text_hash not in msg_hash_to_tasks:
                            msg_hash_to_tasks[text_hash] = set()
                        msg_hash_to_tasks[text_hash].add(tid)
                    except (json.JSONDecodeError, ValueError):
                        continue

        _scan_file_for_injected(env.drive_path("logs/events.jsonl"))
        # Also check supervisor.jsonl for historically unhandled events
        _scan_file_for_injected(
            env.drive_path("logs/supervisor.jsonl"),
            type_field="event_type",
            type_value="owner_message_injected",
        )

        dupes = {h: tids for h, tids in msg_hash_to_tasks.items() if len(tids) > 1}
        if dupes:
            checks.append(
                f"CRITICAL: DUPLICATE PROCESSING — {len(dupes)} message(s) "
                f"appeared in multiple tasks: {', '.join(str(sorted(tids)) for tids in dupes.values())}"
            )
        else:
            checks.append("OK: no duplicate message processing detected")
    except Exception:
        pass

    if not checks:
        return ""
    return "## Health Invariants\n\n" + "\n".join(f"- {c}" for c in checks)


def build_llm_messages(
    env: Any,
    memory: Memory,
    task: Dict[str, Any],
    review_context_builder: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build the full LLM message context for a task.

    Args:
        env: Env instance with repo_path/drive_path helpers
        memory: Memory instance for scratchpad/identity/logs
        task: Task dict with id, type, text, etc.
        review_context_builder: Optional callable for review tasks (signature: () -> str)

    Returns:
        (messages, cap_info) tuple:
            - messages: List of message dicts ready for LLM
            - cap_info: Dict with token trimming metadata
    """
    # --- Extract task type for adaptive context ---
    task_type = str(task.get("type") or "user")

    # --- Read base prompts and state ---
    base_prompt = _safe_read(
        env.repo_path("prompts/SYSTEM.md"),
        fallback="You are Prometheus. Your base prompt could not be loaded."
    )

    # Conditional prompt sections — saves ~950 tokens on non-evolution/consciousness tasks
    if task_type in ("evolution", "review"):
        evo_prompt = _safe_read(env.repo_path("prompts/EVOLUTION.md"), fallback="")
        if evo_prompt:
            base_prompt += "\n\n" + evo_prompt
    if task_type == "consciousness":
        bg_prompt = _safe_read(env.repo_path("prompts/CONSCIOUSNESS.md"), fallback="")
        if bg_prompt:
            base_prompt += "\n\n" + bg_prompt
    bible_md = _safe_read(env.repo_path("BIBLE.md"))
    readme_md = _safe_read(env.repo_path("README.md"))
    state_json = _safe_read(env.drive_path("state/state.json"), fallback="{}")

    # --- Load memory ---
    memory.ensure_files()

    # --- Assemble messages with 3-block prompt caching ---
    # Block 1: Static content (SYSTEM.md + BIBLE.md + README) — cached
    # Block 2: Semi-stable content (identity + scratchpad + knowledge) — cached
    # Block 3: Dynamic content (state + runtime + recent logs) — uncached

    # BIBLE.md always included (Constitution requires it for every decision)
    # README.md only for evolution/review (architecture context)
    needs_full_context = task_type in ("evolution", "review", "scheduled")
    static_text = (
        base_prompt + "\n\n"
        + "## BIBLE.md\n\n" + clip_text(bible_md, 180000)
    )
    if needs_full_context:
        static_text += "\n\n## README.md\n\n" + clip_text(readme_md, 180000)

    # Semi-stable
    semi_text = "\n\n".join(_build_memory_sections(memory))

    # Dynamic
    runtime_section = _build_runtime_section(env, task)
    recent_sections = _build_recent_sections(memory, env, task.get("id", ""))
    dynamic_text = runtime_section + "\n\n" + "\n\n".join(recent_sections)

    # Add health invariants to dynamic (Bible P0: System Invariants)
    health_invariants = _build_health_invariants(env)
    if health_invariants:
        dynamic_text += "\n\n" + health_invariants

    # --- Build messages ---
    messages = []

    # System message (cached blocks 1 + 2)
    system_content = static_text
    if semi_text:
        system_content += "\n\n" + semi_text
    messages.append({
        "role": "system",
        "content": system_content,
    })

    # Assistant messages from prior rounds (context window + cap estimation)
    prior_rounds = task.get("prior_rounds", [])
    for rnd in prior_rounds:
        if rnd.get("role") == "assistant":
            messages.append({"role": "assistant", "content": rnd.get("content", "")})
        elif rnd.get("role") == "user":
            messages.append({"role": "user", "content": rnd.get("content", "")})
        # Skip 'tool' role - handled separately in loop.py

    # Task user message (always last, uncached)
    user_content = _build_user_content(task)
    messages.append({"role": "user", "content": user_content})

    # Tool results from prior rounds
    for rnd in prior_rounds:
        if rnd.get("role") == "tool":
            tool_name = rnd.get("name", "unknown")
            tool_result = rnd.get("content", "")
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": tool_result,
            })

    # Review context (if provided)
    if review_context_builder:
        review_text = review_context_builder()
        if review_text:
            messages.append({
                "role": "system",
                "name": "review_context",
                "content": "\n## Review Context\n\n" + review_text,
            })

    # --- Token estimation for cap decisions ---
    total_chars = sum(estimate_tokens(m["content"]) * 4 for m in messages)
    cap_info = {"total_chars": total_chars, "msg_count": len(messages)}

    return messages, cap_info


def _safe_read(path: pathlib.Path, fallback: str = "") -> str:
    """Read file, return fallback on error."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        log.debug("Failed to read %s", path, exc_info=True)
    return fallback


def summarize_chat(messages: List[Dict[str, Any]]) -> str:
    """Summarize chat messages for context window."""
    if not messages:
        return ""
    lines = []
    for m in messages[-20:]:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            # Skip image parts in summary
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " [multimedia] ".join(text_parts)
        lines.append(f"**{role}**: {clip_text(content, 300)}")
    return "\n".join(lines)
