"""Python execution tool - direct Python code execution with result capture."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict
from datetime import datetime

from prometheus.tools.registry import ToolContext, ToolEntry
from prometheus.utils import utc_now_iso

log = logging.getLogger(__name__)


def _python_exec(ctx: ToolContext, code: str, timeout_sec: int = 60) -> str:
    """Execute Python code directly and return structured results.
    
    Args:
        code: Python code to execute
        timeout_sec: Maximum execution time in seconds (default 60)
    
    Returns:
        JSON string with execution results including stdout, stderr, 
        return value, and any exceptions.
    """
    if not code or not code.strip():
        return json.dumps({
            "success": False,
            "error": "No code provided",
        }, ensure_ascii=False, indent=2)
    
    # Create a sandboxed-like environment
    sandbox_globals = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        # Common imports available
        "json": json,
        "datetime": datetime,
        "os": os,
        "sys": sys,
    }
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    result = None
    error = None
    traceback_str = None
    
    try:
        # Redirect stdout/stderr
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            try:
                # Try to compile first to catch syntax errors early
                compiled = compile(code, "<exec>", "exec")
                # Execute with captured output
                exec(compiled, sandbox_globals)
                result = sandbox_globals.get("_result")
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                traceback_str = traceback.format_exc()
    except Exception as e:
        error = f"Execution error: {type(e).__name__}: {e}"
        traceback_str = traceback.format_exc()
    
    # Build result
    stdout = stdout_capture.getvalue()
    stderr = stderr_capture.getvalue()
    
    response: Dict[str, Any] = {
        "success": error is None,
        "stdout": stdout,
        "stderr": stderr,
    }
    
    if result is not None:
        # Try to serialize the result
        try:
            response["result"] = json.loads(json.dumps(result))
        except (TypeError, ValueError):
            response["result"] = str(result)
    
    if error:
        response["error"] = error
        if traceback_str:
            response["traceback"] = traceback_str
    
    # Truncate long outputs
    max_len = 25000
    if len(response.get("stdout", "")) > max_len:
        response["stdout"] = response["stdout"][:max_len] + "\n...(truncated)..."
    if len(response.get("stderr", "")) > max_len:
        response["stderr"] = response["stderr"][:max_len] + "\n...(truncated)..."
    
    return json.dumps(response, ensure_ascii=False, indent=2)


def get_tools() -> list[ToolEntry]:
    return [
        ToolEntry("python_exec", {
            "name": "python_exec",
            "description": "Execute Python code directly and return structured results. Use for Python calculations, data processing, testing code snippets, and quick scripts. Returns stdout, stderr, and any return value as JSON.",
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout_sec": {"type": "integer", "default": 60, "description": "Maximum execution time in seconds"},
            }, "required": ["code"]},
        }, _python_exec, is_code_tool=True, timeout_sec=300),
    ]
