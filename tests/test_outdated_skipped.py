import unittest
import pytest

class TestToolRegistry:
    """Tests for tool registry."""
    
    def test_import(self):
        from prometheus.tools.registry import get_tools
        tools = get_tools()
        assert isinstance(tools, list)
    
    def test_each_tool_has_name(self):
        from prometheus.tools.registry import get_tools
        tools = get_tools()
        for tool in tools:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert isinstance(tool["name"], str)
            assert len(tool["name"]) > 0

@pytest.mark.skip("outdated")
class TestPrePushGate:
    """Tests for pre-push git hooks."""
    pass

@pytest.mark.skip("outdated") 
class TestTelegramSupervisor:
    """Tests for Telegram supervisor."""
    pass

@pytest.mark.skip("outdated")
class TestAgentCore:
    """Tests for agent core modules."""
    pass

@pytest.mark.skip("outdated")
class TestOrchestrator:
    """Tests for orchestrator."""
    pass
