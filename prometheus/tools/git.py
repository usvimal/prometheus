"""Git tools: repo_write_commit, repo_commit_push, git_status, git_diff."""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry
from prometheus.utils import utc_now_iso, write_text, safe_relpath, run_cmd

log = logging.getLogger(__name__)

# --- Evolution guardrails ---
# These files are critical supervisor infrastructure. Evolution tasks must NOT
# modify them because MiniMax can't safely refactor multi-file dependencies.
# If evolution breaks these, the bot can't process messages, drain events, or restart.
PROTECTED_PATHS = frozenset({
    "launcher.py",
    "supervisor/events.py",
    "supervisor/workers.py",
    "supervisor/queue.py",
    "supervisor/state.py",
    "supervisor/telegram.py",
    "supervisor/git_ops.py",
})

MAX_EVOLUTION_FILES = 4


def _check_evolution_guard(ctx, paths):
    """Block evolution tasks from modifying protected supervisor files."""
    task_type = getattr(ctx, 'current_task_type', None)
    if not task_type or task_type != 'evolution':
        return None

    blocked = []
    for p in paths:
        norm = str(p).replace('\\', '/').lstrip('/')
        if norm in PROTECTED_PATHS:
            blocked.append(norm)
    if blocked:
        return (
            'EVOLUTION GUARD: Cannot modify protected supervisor files: '
            + ', '.join(blocked)
            + '. These files are critical infrastructure. '
            + 'Focus evolution on prometheus/ (agent code, tools, prompts) instead.'
        )
    if len(paths) > MAX_EVOLUTION_FILES:
        return (
            'EVOLUTION GUARD: Too many files (' + str(len(paths)) + ') in one commit. '
            + 'Max ' + str(MAX_EVOLUTION_FILES) + ' during evolution. Break into smaller changes.'
        )
    return None


# Maximum allowed shrinkage ratio before blocking (0.2 = 20%)
_MAX_SHRINKAGE_RATIO = 0.20
# Minimum file size (lines) to trigger truncation check
_MIN_LINES_FOR_CHECK = 50


def _check_truncation_guard(ctx, path: str, new_content: str) -> Optional[str]:
    """Block commits that would significantly shrink an existing file.

    LLMs often truncate large files when rewriting them, silently dropping
    hundreds of lines. This guard catches it before the damage is committed.
    """
    try:
        full_path = ctx.repo_path(path)
        if not full_path.exists():
            return None  # New file, no guard needed

        old_content = full_path.read_text(encoding="utf-8")
        old_lines = len(old_content.splitlines())
        new_lines = len(new_content.splitlines())

        if old_lines < _MIN_LINES_FOR_CHECK:
            return None  # Small file, not worth checking

        if new_lines >= old_lines:
            return None  # File grew or stayed same size

        shrinkage = (old_lines - new_lines) / old_lines
        if shrinkage > _MAX_SHRINKAGE_RATIO:
            lost = old_lines - new_lines
            return (
                f"⚠️ TRUNCATION GUARD: Blocked commit to {path}. "
                f"File would shrink from {old_lines} to {new_lines} lines "
                f"({lost} lines / {shrinkage:.0%} lost). "
                f"This usually means the LLM truncated the file content. "
                f"Read the full file with repo_read first, then write back "
                f"the complete content with only your intended changes."
            )
    except Exception:
        pass  # Don't block on guard errors
    return None




# --- Git lock ---

def _acquire_git_lock(ctx: ToolContext, timeout_sec: int = 120) -> pathlib.Path:
    lock_dir = ctx.drive_path("locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "git.lock"
    stale_sec = 600
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if lock_path.exists():
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_sec:
                    lock_path.unlink()
                    continue
            except (FileNotFoundError, OSError):
                pass
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"locked_at={utc_now_iso()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(0.5)
    raise TimeoutError(f"Git lock not acquired within {timeout_sec}s: {lock_path}")


def _release_git_lock(lock_path: pathlib.Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# --- Pre-push test gate ---

MAX_TEST_OUTPUT = 8000

def _run_pre_push_tests(ctx: ToolContext) -> Optional[str]:
    """Run pre-push tests if enabled. Returns None if tests pass, error string if they fail."""
    # Guard against ctx=None
    if ctx is None:
        log.warning("_run_pre_push_tests called with ctx=None, skipping tests")
        return None

    if os.environ.get("PROMETHEUS_PRE_PUSH_TESTS", "1") != "1":
        return None

    tests_dir = pathlib.Path(ctx.repo_dir) / "tests"
    if not tests_dir.exists():
        return None

    try:
        result = subprocess.run(
            ["pytest", "tests/", "-q", "--tb=line", "--no-header"],
            cwd=ctx.repo_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return None

        # Truncate output if too long
        output = result.stdout + result.stderr
        if len(output) > MAX_TEST_OUTPUT:
            output = output[:MAX_TEST_OUTPUT] + "\n...(truncated)..."
        return output

    except subprocess.TimeoutExpired:
        return "⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after 30 seconds"

    except FileNotFoundError:
        return "⚠️ PRE_PUSH_TEST_ERROR: pytest not installed or not found in PATH"

    except Exception as e:
        log.warning(f"Pre-push tests failed with exception: {e}", exc_info=True)
        return f"⚠️ PRE_PUSH_TEST_ERROR: Unexpected error running tests: {e}"


def _git_push_with_tests(ctx: ToolContext) -> Optional[str]:
    """Run pre-push tests, then pull --rebase and push. Returns None on success, error string on failure."""
    test_error = _run_pre_push_tests(ctx)
    if test_error:
        log.error("Pre-push tests failed, blocking push")
        ctx.last_push_succeeded = False
        return f"⚠️ PRE_PUSH_TESTS_FAILED: Tests failed, push blocked.\n{test_error}\nCommitted locally but NOT pushed. Fix tests and push manually."

    try:
        run_cmd(["git", "pull", "--rebase", "origin", ctx.branch_dev], cwd=ctx.repo_dir)
    except Exception:
        log.debug(f"Failed to pull --rebase before push", exc_info=True)

    try:
        run_cmd(["git", "push", "origin", ctx.branch_dev], cwd=ctx.repo_dir)
    except Exception as e:
        return f"⚠️ GIT_ERROR (push): {e}\nCommitted locally but NOT pushed."

    return None


# --- Tool implementations ---

def _repo_write_commit(ctx: ToolContext, path: str, content: str, commit_message: str) -> str:
    ctx.last_push_succeeded = False
    guard = _check_evolution_guard(ctx, [path])
    if guard:
        return guard
    trunc = _check_truncation_guard(ctx, path, content)
    if trunc:
        return trunc
    if not commit_message.strip():
        return "⚠️ ERROR: commit_message must be non-empty."
    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (checkout): {e}"
        try:
            write_text(ctx.repo_path(path), content)
        except Exception as e:
            return f"⚠️ FILE_WRITE_ERROR: {e}"
        try:
            run_cmd(["git", "add", safe_relpath(path)], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (add): {e}"
        try:
            run_cmd(["git", "commit", "-m", commit_message], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (commit): {e}"

        push_error = _git_push_with_tests(ctx)
        if push_error:
            return push_error
    finally:
        _release_git_lock(lock)
    ctx.last_push_succeeded = True
    # Warn if commit message seems inflated relative to change size
    if len(commit_message) > 200:
        try:
            diff_stat = run_cmd(["git", "diff", "HEAD~1", "--stat", "--stat-count=1"], cwd=ctx.repo_dir)
            if "1 file changed" in diff_stat and len(commit_message) > 300:
                log.warning("Commit message (%d chars) seems inflated for a 1-file change", len(commit_message))
        except Exception:
            pass
    return f"OK: committed and pushed to {ctx.branch_dev}: {commit_message}"


def _repo_commit_push(ctx: ToolContext, commit_message: str, paths: Optional[List[str]] = None) -> str:
    ctx.last_push_succeeded = False
    if paths:
        guard = _check_evolution_guard(ctx, paths)
        if guard:
            return guard
    if not commit_message.strip():
        return "⚠️ ERROR: commit_message must be non-empty."
    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (checkout): {e}"
        if paths:
            try:
                safe_paths = [safe_relpath(p) for p in paths if str(p).strip()]
            except ValueError as e:
                return f"⚠️ PATH_ERROR: {e}"
            add_cmd = ["git", "add"] + safe_paths
        else:
            add_cmd = ["git", "add", "-A"]
        try:
            run_cmd(add_cmd, cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (add): {e}"
        try:
            status = run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (status): {e}"
        if not status.strip():
            return "⚠️ GIT_NO_CHANGES: nothing to commit."

        # Block no-op commits (only marker files, no real code)
        changed_files = [line[3:].strip() for line in status.strip().splitlines() if line.strip()]
        meaningful = [f for f in changed_files if not f.startswith('.') and not f.endswith('.marker')]
        if not meaningful:
            return "⚠️ EVOLUTION_GUARD: Only marker/dot files changed. This is not a meaningful commit."
        try:
            run_cmd(["git", "commit", "-m", commit_message], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (commit): {e}"

        push_error = _git_push_with_tests(ctx)
        if push_error:
            return push_error
    finally:
        _release_git_lock(lock)
    ctx.last_push_succeeded = True
    result = f"OK: committed and pushed to {ctx.branch_dev}: {commit_message}"
    if paths is not None:
        try:
            untracked = run_cmd(["git", "ls-files", "--others", "--exclude-standard"], cwd=ctx.repo_dir)
            if untracked.strip():
                files = ", ".join(untracked.strip().split("\n"))
                result += f"\n⚠️ WARNING: untracked files remain: {files} — they are NOT in git. Use repo_commit_push without paths to add everything."
        except Exception:
            log.debug("Failed to check for untracked files after repo_commit_push", exc_info=True)
            pass
    return result


def _git_status(ctx: ToolContext) -> str:
    try:
        return run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir)
    except Exception as e:
        return f"⚠️ GIT_ERROR: {e}"


def _git_diff(ctx: ToolContext, staged: bool = False) -> str:
    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        return run_cmd(cmd, cwd=ctx.repo_dir)
    except Exception as e:
        return f"⚠️ GIT_ERROR: {e}"


def _repo_search_replace(ctx: ToolContext, path: str, search: str, replace: str, commit_message: str) -> str:
    """Search-and-replace in an existing file, then commit + push.

    Safer than repo_write_commit for editing existing files because only
    the changed portion is in the tool output (no truncation risk).
    """
    ctx.last_push_succeeded = False
    guard = _check_evolution_guard(ctx, [path])
    if guard:
        return guard
    if not commit_message.strip():
        return "\u26a0\ufe0f ERROR: commit_message must be non-empty."

    full_path = ctx.repo_path(path)
    if not full_path.exists():
        return f"\u26a0\ufe0f FILE_NOT_FOUND: {path} does not exist. Use repo_write_commit for new files."

    try:
        content = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"\u26a0\ufe0f FILE_READ_ERROR: {e}"

    count = content.count(search)
    if count == 0:
        # Show nearby content to help the agent find the right text
        lines = content.splitlines()
        snippet = "\n".join(lines[:30]) if len(lines) > 30 else content
        return (
            f"\u26a0\ufe0f SEARCH_NOT_FOUND: The search text was not found in {path}.\n"
            f"First 30 lines of file:\n{snippet}"
        )
    if count > 1:
        return (
            f"\u26a0\ufe0f SEARCH_NOT_UNIQUE: Found {count} occurrences of the search text in {path}. "
            f"Provide more context to make the match unique."
        )

    new_content = content.replace(search, replace, 1)

    # Run truncation guard on result
    trunc = _check_truncation_guard(ctx, path, new_content)
    if trunc:
        return trunc

    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
        except Exception as e:
            return f"\u26a0\ufe0f GIT_ERROR (checkout): {e}"
        try:
            write_text(full_path, new_content)
        except Exception as e:
            return f"\u26a0\ufe0f FILE_WRITE_ERROR: {e}"
        try:
            run_cmd(["git", "add", safe_relpath(path)], cwd=ctx.repo_dir)
        except Exception as e:
            return f"\u26a0\ufe0f GIT_ERROR (add): {e}"
        try:
            run_cmd(["git", "commit", "-m", commit_message], cwd=ctx.repo_dir)
        except Exception as e:
            return f"\u26a0\ufe0f GIT_ERROR (commit): {e}"

        push_error = _git_push_with_tests(ctx)
        if push_error:
            return push_error
    finally:
        _release_git_lock(lock)
    ctx.last_push_succeeded = True
    return f"OK: search-replace in {path}, committed and pushed: {commit_message}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("repo_write_commit", {
            "name": "repo_write_commit",
            "description": "Write one file + commit + push to dev branch. For small deterministic edits.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "commit_message": {"type": "string"},
            }, "required": ["path", "content", "commit_message"]},
        }, _repo_write_commit, is_code_tool=True),
        ToolEntry("repo_commit_push", {
            "name": "repo_commit_push",
            "description": "Commit + push already-changed files. Does pull --rebase before push.",
            "parameters": {"type": "object", "properties": {
                "commit_message": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Files to add (empty = git add -A)"},
            }, "required": ["commit_message"]},
        }, _repo_commit_push, is_code_tool=True),
        ToolEntry("git_status", {
            "name": "git_status",
            "description": "git status --porcelain",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _git_status, is_code_tool=True),
        ToolEntry("git_diff", {
            "name": "git_diff",
            "description": "git diff (use staged=true to see staged changes after git add)",
            "parameters": {"type": "object", "properties": {
                "staged": {"type": "boolean", "default": False, "description": "If true, show staged changes (--staged)"},
            }, "required": []},
        }, _git_diff, is_code_tool=True),
        ToolEntry("repo_search_replace", {
            "name": "repo_search_replace",
            "description": "Search-and-replace in an existing file, then commit + push. "
                           "PREFERRED over repo_write_commit for editing existing files — "
                           "only outputs the changed portion (no truncation risk). "
                           "Search text must match exactly and be unique in the file.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "search": {"type": "string", "description": "Exact text to find (must be unique in file)"},
                "replace": {"type": "string", "description": "Text to replace it with"},
                "commit_message": {"type": "string"},
            }, "required": ["path", "search", "replace", "commit_message"]},
        }, _repo_search_replace, is_code_tool=True),
    ]
