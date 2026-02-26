"""File watcher tool — monitor files and directories for changes.

Provides:
- watch_start: Begin watching a path (file or directory)
- watch_check: Check for changes since last check
- watch_stop: Stop watching a path
"""

import os
import time
import pathlib
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolEntry, ToolContext

# In-memory watch state (per-process)
_watches: Dict[str, Dict[str, Any]] = {}


def _get_stat_info(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Get file stat info for change detection."""
    try:
        stat = path.stat()
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "exists": True,
        }
    except FileNotFoundError:
        return {"exists": False}
    except Exception:
        return None


def _watch_start(ctx: ToolContext, path: str, recursive: bool = True) -> str:
    """Start watching a file or directory for changes."""
    global _watches
    
    abs_path = ctx.repo_path(path).resolve()
    
    if not abs_path.exists():
        return f"⚠️ Path does not exist: {path}"
    
    watch_id = str(abs_path)
    
    # Get initial state
    files_state = {}
    if abs_path.is_dir():
        if recursive:
            for f in abs_path.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(abs_path)
                    files_state[str(rel)] = _get_stat_info(f)
        else:
            for f in abs_path.iterdir():
                if f.is_file():
                    rel = f.relative_to(abs_path)
                    files_state[str(rel)] = _get_stat_info(f)
    else:
        files_state[abs_path.name] = _get_stat_info(abs_path)
    
    _watches[watch_id] = {
        "path": str(abs_path),
        "recursive": recursive,
        "files": files_state,
        "started_at": time.time(),
    }
    
    action = "recursively" if recursive else "non-recursively"
    return f"✅ Started watching `{path}` ({action}). Watch ID: `{watch_id}`"


def _watch_check(ctx: ToolContext, path: str) -> str:
    """Check for changes since last check."""
    global _watches
    
    abs_path = ctx.repo_path(path).resolve()
    watch_id = str(abs_path)
    
    if watch_id not in _watches:
        return f"⚠️ No active watch for: {path}. Use watch_start first."
    
    watch = _watches[watch_id]
    old_files = watch["files"]
    changes = []
    
    base_path = pathlib.Path(watch["path"])
    
    # Check current state
    current_files = {}
    if base_path.is_dir():
        if watch.get("recursive", True):
            for f in base_path.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(base_path)
                    current_files[str(rel)] = _get_stat_info(f)
        else:
            for f in base_path.iterdir():
                if f.is_file():
                    rel = f.relative_to(base_path)
                    current_files[str(rel)] = _get_stat_info(f)
    else:
        current_files[base_path.name] = _get_stat_info(base_path)
    
    # Detect changes
    all_files = set(old_files.keys()) | set(current_files.keys())
    
    for f in all_files:
        old_state = old_files.get(f)
        new_state = current_files.get(f)
        
        if old_state is None and new_state is not None:
            changes.append(f"➕ Added: {f}")
        elif old_state is not None and new_state is None:
            changes.append(f"➖ Removed: {f}")
        elif old_state and new_state:
            if old_state.get("mtime") != new_state.get("mtime"):
                changes.append(f"✏️ Modified: {f}")
            elif old_state.get("size") != new_state.get("size"):
                changes.append(f"📏 Size changed: {f}")
    
    # Update state
    watch["files"] = current_files
    
    if not changes:
        return f"ℹ️ No changes detected in `{path}` since last check."
    
    result = f"🔔 Changes detected in `{path}`:\n" + "\n".join(f"  {c}" for c in changes)
    return result


def _watch_stop(ctx: ToolContext, path: str) -> str:
    """Stop watching a file or directory."""
    global _watches
    
    abs_path = ctx.repo_path(path).resolve()
    watch_id = str(abs_path)
    
    if watch_id not in _watches:
        return f"⚠️ No active watch for: {path}"
    
    del _watches[watch_id]
    return f"✅ Stopped watching: {path}"


def _watch_list(ctx: ToolContext) -> str:
    """List all active watches."""
    global _watches
    
    if not _watches:
        return "ℹ️ No active file watches."
    
    lines = ["📁 Active file watches:"]
    for watch_id, watch in _watches.items():
        path = watch["path"]
        started = time.strftime("%H:%M:%S", time.localtime(watch["started_at"]))
        recursive = "recursive" if watch.get("recursive", True) else "non-recursive"
        lines.append(f"  - {path} ({recursive}) since {started}")
    
    return "\n".join(lines)


def get_tools() -> List[ToolEntry]:
    """Return file watcher tools."""
    return [
        ToolEntry(
            name="watch_start",
            schema={
                "name": "watch_start",
                "description": "Start watching a file or directory for changes. Use watch_check to detect changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File or directory path to watch (relative to repo)"},
                        "recursive": {"type": "boolean", "description": "Watch subdirectories recursively (default: true)", "default": True},
                    },
                    "required": ["path"],
                },
            },
            handler=lambda ctx, path, recursive=True: _watch_start(ctx, path, recursive),
            timeout_sec=30,
        ),
        ToolEntry(
            name="watch_check",
            schema={
                "name": "watch_check",
                "description": "Check for file changes since watch_start. Returns list of added, removed, or modified files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File or directory path being watched"},
                    },
                    "required": ["path"],
                },
            },
            handler=lambda ctx, path: _watch_check(ctx, path),
            timeout_sec=30,
        ),
        ToolEntry(
            name="watch_stop",
            schema={
                "name": "watch_stop",
                "description": "Stop watching a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File or directory path to stop watching"},
                    },
                    "required": ["path"],
                },
            },
            handler=lambda ctx, path: _watch_stop(ctx, path),
            timeout_sec=30,
        ),
        ToolEntry(
            name="watch_list",
            schema={
                "name": "watch_list",
                "description": "List all active file watches.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            handler=lambda ctx: _watch_list(ctx),
            timeout_sec=30,
        ),
    ]
