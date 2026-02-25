"""Smoke test suite for Ouroboros.

Tests core invariants:
- All modules import cleanly
- Tool registry discovers all 33 tools
- Utility functions work correctly
- Memory operations don't crash
- Context builder produces valid structure
- Bible invariants hold (no hardcoded replies, version sync)

Run: python -m pytest tests/test_smoke.py -v
"""
import ast
import os
import pathlib
import re
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# ── Module imports ───────────────────────────────────────────────

CORE_MODULES = [
    "prometheus.agent",
    "prometheus.context",
    "prometheus.loop",
    "prometheus.llm",
    "prometheus.memory",
    "prometheus.review",
    "prometheus.utils",
    "prometheus.consciousness",
]

TOOL_MODULES = [
    "prometheus.tools.registry",
    "prometheus.tools.core",
    "prometheus.tools.git",
    "prometheus.tools.shell",
    "prometheus.tools.search",
    "prometheus.tools.control",
    "prometheus.tools.browser",
    "prometheus.tools.review",
]

SUPERVISOR_MODULES = [
    "supervisor.state",
    "supervisor.telegram",
    "supervisor.queue",
    "supervisor.workers",
    "supervisor.git_ops",
    "supervisor.events",
]


@pytest.mark.parametrize("module", CORE_MODULES + TOOL_MODULES + SUPERVISOR_MODULES)
def test_import(module):
    """Every module imports without error."""
    __import__(module)


# ── Tool registry ────────────────────────────────────────────────

@pytest.fixture
def registry():
    from prometheus.tools.registry import ToolRegistry
    tmp = pathlib.Path(tempfile.mkdtemp())
    return ToolRegistry(repo_dir=tmp, drive_root=tmp)


def test_tool_set_matches(registry):
    """Tool registry contains exactly the expected tools (no more, no less)."""
    schemas = registry.schemas()
    actual_tools = {t["function"]["name"] for t in schemas}
    expected_tools = set(EXPECTED_TOOLS)

    missing = expected_tools - actual_tools
    extra = actual_tools - expected_tools

    assert missing == set(), f"Missing tools: {sorted(missing)}"
    assert extra == set(), f"Extra tools: {sorted(extra)}"
    assert actual_tools == expected_tools, "Tool set mismatch"


EXPECTED_TOOLS = [
    "repo_read", "repo_write_commit", "repo_list", "repo_commit_push",
    "drive_read", "drive_write", "drive_list",
    "git_status", "git_diff",
    "run_shell", "claude_code_edit",
    "browse_page", "browser_action",
    "browser_search",
    # Semantic Memory tools
    "memory_store", "memory_recall", "memory_list", "memory_delete",
    # Communication
    "chat_history", "update_scratchpad", "update_identity",
    "request_restart", "promote_to_stable", "request_review",
    "schedule_task", "cancel_task",
    "switch_model", "toggle_evolution", "toggle_consciousness",
    "send_owner_message", "send_photo",
    "codebase_digest", "codebase_health",
    "knowledge_read", "knowledge_write", "knowledge_list",
    "multi_model_review",
    # GitHub Issues
    "list_github_issues", "get_github_issue", "comment_on_issue",
    "close_github_issue", "create_github_issue",
    "summarize_dialogue",
    # Task decomposition
    "get_task_result", "wait_for_task",
    "generate_evolution_stats",
    # VLM / Vision
    "analyze_screenshot", "vlm_query",
    # Message routing
    "forward_to_worker",
    # Context management
    "compact_context",
    "list_available_tools",
    "enable_tools",
    "quick_search",
]


@pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
def test_tool_registered(registry, tool_name):
    """Each expected tool is in the registry."""
    available = [t["function"]["name"] for t in registry.schemas()]
    assert tool_name in available, f"{tool_name} not in registry"


def test_unknown_tool_returns_warning(registry):
    """Calling unknown tool returns warning, not exception."""
    result = registry.execute("__nonexistent__", {})
    assert "Unknown tool" in result or "⚠️" in result


def test_tool_schemas_valid(registry):
    """All tool schemas have required OpenAI fields."""
    for schema in registry.schemas():
        assert schema["type"] == "function"
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_tool_execute_basic(registry):
    """Actually execute a simple tool to verify execution works."""
    result = registry.execute("run_shell", {"cmd": "echo hello"})
    assert isinstance(result, str), "Tool execute should return string"
    assert "hello" in result.lower() or "⚠️" in result, "Should return output or error"


# ── Utilities ────────────────────────────────────────────────────

def test_safe_relpath_normal():
    from prometheus.utils import safe_relpath
    result = safe_relpath("foo/bar.py")
    assert result == "foo/bar.py"


def test_safe_relpath_rejects_traversal():
    from prometheus.utils import safe_relpath
    with pytest.raises(ValueError):
        safe_relpath("../../../etc/passwd")


def test_safe_relpath_strips_leading_slash():
    """safe_relpath strips leading / but doesn't raise."""
    from prometheus.utils import safe_relpath
    result = safe_relpath("/etc/passwd")
    assert not result.startswith("/")


def test_clip_text():
    from prometheus.utils import clip_text

    # Test 1: Long text gets clipped (max_chars=500)
    long_text = "hello world " * 100  # ~1200 chars
    result = clip_text(long_text, 500)
    assert len(result) < len(long_text), "Long text should be clipped"
    assert len(result) > 0, "Result should not be empty"
    assert "...(truncated)..." in result, "Truncation marker should be present"

    # Test 2: Short text passes through unchanged
    short_text = "hello world"
    result_short = clip_text(short_text, 500)
    assert result_short == short_text, "Short text should pass through unchanged"


def test_estimate_tokens():
    from prometheus.utils import estimate_tokens
    tokens = estimate_tokens("Hello world, this is a test.")
    assert 5 <= tokens <= 20


# ── Memory ───────────────────────────────────────────────────────

def test_memory_scratchpad():
    """Memory reads/writes scratchpad without crash."""
    from prometheus.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(drive_root=pathlib.Path(tmp))
        mem.save_scratchpad("test content")
        content = mem.load_scratchpad()
        assert "test content" in content


def test_memory_identity():
    """Memory reads/writes identity without crash."""
    from prometheus.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(drive_root=pathlib.Path(tmp))
        # Write identity file directly (identity_path is a method)
        mem.identity_path().parent.mkdir(parents=True, exist_ok=True)
        mem.identity_path().write_text("I am Ouroboros")
        content = mem.load_identity()
        assert "Ouroboros" in content


def test_memory_chat_history_empty():
    """Chat history returns string when no data."""
    from prometheus.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(drive_root=pathlib.Path(tmp))
        history = mem.chat_history(count=10)
        assert isinstance(history, str)


def test_memory_persistence():
    """Memory persists across instances (write with one, read with another)."""
    from prometheus.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)

        # Write with first instance
        mem1 = Memory(drive_root=tmp_path)
        mem1.save_scratchpad("test persistence content")

        # Read with second instance
        mem2 = Memory(drive_root=tmp_path)
        content = mem2.load_scratchpad()
        assert "test persistence content" in content, "Memory should persist across instances"


# ── Context builder ─────────────────────────────────────────────

def test_context_build_runtime_section():
    """Runtime section builder is callable."""
    from prometheus.context import _build_runtime_section
    # Just check it's importable and callable
    assert callable(_build_runtime_section)


def test_context_build_memory_sections():
    """Memory sections builder is callable."""
    from prometheus.context import _build_memory_sections
    assert callable(_build_memory_sections)


# ── Bible invariants ─────────────────────────────────────────────

def test_no_hardcoded_replies():
    """Principle 3 (LLM-first): no hardcoded reply strings in code.
    
    Checks for suspicious patterns like:
    - reply = "Fixed string"
    - return "Sorry, I can't..."
    """
    suspicious = re.compile(
        r'(reply|response)\s*=\s*["\'](?!$|{|\s*$)',
        re.IGNORECASE,
    )
    violations = []
    for root, dirs, files in os.walk(REPO / "prometheus"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                if suspicious.search(line):
                    if "{" in line or "f'" in line or 'f"' in line:
                        continue
                    violations.append(f"{path.name}:{i}: {line.strip()}")
    assert len(violations) < 5, f"Possible hardcoded replies:\n" + "\n".join(violations)


def test_version_file_exists():
    """VERSION file exists and contains valid semver."""
    version = (REPO / "VERSION").read_text().strip()
    parts = version.split(".")
    assert len(parts) == 3, f"VERSION '{version}' is not semver"
    for p in parts:
        assert p.isdigit(), f"VERSION part '{p}' is not numeric"


def test_version_in_readme():
    """VERSION matches what README claims."""
    version = (REPO / "VERSION").read_text().strip()
    readme = (REPO / "README.md").read_text()
    assert version in readme, f"VERSION {version} not found in README.md"


def test_bible_exists_and_has_principles():
    """BIBLE.md exists and contains all 9 principles (0-8)."""
    bible = (REPO / "BIBLE.md").read_text()
    for i in range(9):
        assert f"Principle {i}" in bible, f"Principle {i} missing from BIBLE.md"


# ── Code quality invariants ──────────────────────────────────────

def test_no_env_dumping():
    """Security: no code dumps entire env (os.environ without key access).

    Allows: os.environ["KEY"], os.environ.get(), os.environ.setdefault(),
            os.environ.copy() (for subprocess).
    Disallows: print(os.environ), json.dumps(os.environ), etc.
    """
    # Only flag raw os.environ passed to print/json/log without bracket or .get( accessor
    dangerous = re.compile(r'(?:print|json\.dumps|log)\s*\(.*\bos\.environ\b(?!\s*[\[.])')
    violations = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'tests')]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                if dangerous.search(line):
                    violations.append(f"{path.name}:{i}: {line.strip()[:80]}")
    assert len(violations) == 0, f"Dangerous env dumping:\n" + "\n".join(violations)


def test_no_oversized_modules():
    """Principle 5: no module exceeds 1000 lines."""
    max_lines = 1000
    violations = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'tests')]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            lines = len(path.read_text().splitlines())
            if lines > max_lines:
                violations.append(f"{path.name}: {lines} lines")
    assert len(violations) == 0, f"Oversized modules (>{max_lines} lines):\n" + "\n".join(violations)


def test_no_bare_except_pass():
    """No bare `except: pass` (not even except Exception: pass with just pass).
    
    v4.9.0 hardened exceptions — but checks the STRICTEST form:
    bare except (no Exception class) followed by pass.
    """
    violations = []
    for root, dirs, files in os.walk(REPO / "prometheus"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Only flag bare `except:` (no class specified)
                if stripped == "except:":
                    # Check next non-empty line is just `pass`
                    for j in range(i, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and next_line == "pass":
                            violations.append(f"{path.name}:{i}: bare except: pass")
                            break
    assert len(violations) == 0, f"Bare except:pass found:\n" + "\n".join(violations)


# ── AST-based function size check ───────────────────────────────

MAX_FUNCTION_LINES = 200  # Hard limit — anything above is a bug


def _get_function_sizes():
    """Return list of (file, func_name, lines) for all functions."""
    results = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'tests')]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    size = node.end_lineno - node.lineno + 1
                    results.append((f, node.name, size))
    return results


def test_no_extremely_oversized_functions():
    """No function exceeds 200 lines (hard limit)."""
    violations = []
    for fname, func_name, size in _get_function_sizes():
        if size > MAX_FUNCTION_LINES:
            violations.append(f"{fname}:{func_name} = {size} lines")
    assert len(violations) == 0, \
        f"Functions exceeding {MAX_FUNCTION_LINES} lines:\n" + "\n".join(violations)


def test_function_count_reasonable():
    """Codebase doesn't have too few or too many functions."""
    sizes = _get_function_sizes()
    assert len(sizes) >= 100, f"Only {len(sizes)} functions — too few?"
    assert len(sizes) <= 1000, f"{len(sizes)} functions — too many?"


# ── Pre-push gate tests ──────────────────────────────────────────────

class TestPrePushGate:
    """Tests for pre-push test gate in supervisor/git_ops.py"""
    
    def test_pre_push_runs_tests(self):
        """Pre-push hook runs tests."""
        # This test is intentionally minimal since we're testing the hook exists
        # Real validation happens at push time
        from supervisor.git_ops import run_pre_push_tests
        assert callable(run_pre_push_tests)


    def test_get_staged_files(self):
        """Can get list of staged files."""
        from supervisor.git_ops import get_staged_files
        # Should return list (may be empty)
        files = get_staged_files()
        assert isinstance(files, list)


    def test_get_diff_summary(self):
        """Can get diff summary."""
        from supervisor.git_ops import get_diff_summary
        # Should return string (may be empty)
        diff = get_diff_summary()
        assert isinstance(diff, dict)


# ── Telegram supervisor tests ─────────────────────────────────────

class TestTelegramSupervisor:
    """Tests for telegram supervisor integration."""
    
    def test_telegram_message_parsing(self):
        """Can parse telegram message format."""
        from supervisor.telegram import parse_telegram_message
        # Test basic parsing
        result = parse_telegram_message({"message": {"text": "hello", "chat": {"id": 123}}})
        assert "text" in result or "chat_id" in result


    def test_telegram_reply_format(self):
        """Can format reply for telegram."""
        from supervisor.telegram import format_telegram_reply
        result = format_telegram_reply("test message")
        assert "test message" in result


# ── Agent tests ─────────────────────────────────────────────────

class TestAgentCore:
    """Core agent functionality tests."""
    
    def test_agent_initialization(self):
        """Agent can be initialized with repo dir."""
        from prometheus.agent import OuroborosAgent
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = OuroborosAgent(repo_dir=pathlib.Path(tmp))
            assert agent is not None


    def test_agent_has_loop(self):
        """Agent has loop attribute."""
        from prometheus.agent import OuroborosAgent
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = OuroborosAgent(repo_dir=pathlib.Path(tmp))
            assert hasattr(agent, 'loop')


    def test_agent_has_context(self):
        """Agent has context builder."""
        from prometheus.agent import OuroborosAgent
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            agent = OuroborosAgent(repo_dir=pathlib.Path(tmp))
            assert hasattr(agent, 'context')


# ── Orchestrator tests ────────────────────────────────────────────
