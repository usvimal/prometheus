"""MCP (Model Context Protocol) client tool - connect to MCP servers and use their tools."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from prometheus.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# MCP client instances cache
_mcp_clients: Dict[str, Any] = {}


def _mcp_connect(ctx: ToolContext, server_name: str, transport: str = "stdio", 
                 command: Optional[str] = None, args: Optional[list] = None,
                 url: Optional[str] = None, env: Optional[dict] = None) -> str:
    """Connect to an MCP server and cache the connection.
    
    Args:
        server_name: A friendly name to identify this connection
        transport: "stdio" for local subprocess or "http" for remote server
        command: Command to run for stdio transport (e.g., "npx", "python")
        args: Arguments for the command (e.g., ["-m", "some-mcp-server"])
        url: URL for HTTP transport (e.g., "http://localhost:8000/mcp")
        env: Environment variables for stdio transport
    
    Returns:
        JSON with connection status and available tools/resources
    """
    global _mcp_clients
    
    if server_name in _mcp_clients:
        return json.dumps({
            "success": True,
            "message": f"Already connected to '{server_name}'. Use mcp_list_tools to see available tools.",
            "server_name": server_name,
        }, ensure_ascii=False, indent=2)
    
    try:
        # Import MCP SDK
        from mcp import ClientSession, StdioServerParameters
        import asyncio
        
        async def connect():
            if transport == "stdio":
                if not command:
                    raise ValueError("command is required for stdio transport")
                
                server_params = StdioServerParameters(
                    command=command,
                    args=args or [],
                    env=env or {},
                )
                async with ClientSession(server_params) as session:
                    await session.initialize()
                    # Get capabilities
                    init_result = session._init_result
                    tools = await session.list_tools()
                    resources = await session.list_resources()
                    
                    _mcp_clients[server_name] = {
                        "session": session,
                        "transport": transport,
                        "tools": [t.name for t in tools.tools],
                        "resources": [r.uri for r in resources.resources] if resources.resources else [],
                        "capabilities": init_result.capabilities if hasattr(init_result, 'capabilities') else {},
                    }
                    return {
                        "success": True,
                        "message": f"Connected to MCP server '{server_name}'",
                        "server_name": server_name,
                        "tools": _mcp_clients[server_name]["tools"],
                        "resources": _mcp_clients[server_name]["resources"],
                    }
                    
            elif transport == "http":
                if not url:
                    raise ValueError("url is required for http transport")
                
                # For HTTP transport
                import httpx
                from mcp.client.streamable_http import StreamableHttpClient
                
                async with StreamableHttpClient(url) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    resources = await session.list_resources()
                    
                    _mcp_clients[server_name] = {
                        "session": session,
                        "transport": transport,
                        "url": url,
                        "tools": [t.name for t in tools.tools],
                        "resources": [r.uri for r in resources.resources] if resources.resources else [],
                    }
                    return {
                        "success": True,
                        "message": f"Connected to MCP server '{server_name}' via HTTP",
                        "server_name": server_name,
                        "tools": _mcp_clients[server_name]["tools"],
                        "resources": _mcp_clients[server_name]["resources"],
                    }
            else:
                raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'http'.")
        
        # Run async connection
        result = asyncio.run(connect())
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except ImportError as e:
        return json.dumps({
            "success": False,
            "error": f"MCP SDK not installed: {e}. Run: pip install mcp",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"MCP connection error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Connection failed: {type(e).__name__}: {e}",
        }, ensure_ascii=False, indent=2)


def _mcp_disconnect(ctx: ToolContext, server_name: str) -> str:
    """Disconnect from an MCP server.
    
    Args:
        server_name: Name of the server to disconnect from
    
    Returns:
        JSON with disconnection status
    """
    global _mcp_clients
    
    if server_name not in _mcp_clients:
        return json.dumps({
            "success": False,
            "error": f"Not connected to '{server_name}'. Use mcp_list to see active connections.",
        }, ensure_ascii=False, indent=2)
    
    try:
        # Close session if possible
        if "session" in _mcp_clients[server_name]:
            import asyncio
            session = _mcp_clients[server_name]["session"]
            if hasattr(session, 'close'):
                asyncio.run(session.close())
        
        del _mcp_clients[server_name]
        return json.dumps({
            "success": True,
            "message": f"Disconnected from '{server_name}'",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Disconnect error: {type(e).__name__}: {e}",
        }, ensure_ascii=False, indent=2)


def _mcp_list(ctx: ToolContext) -> str:
    """List all active MCP server connections.
    
    Returns:
        JSON with list of active connections and their tools/resources
    """
    global _mcp_clients
    
    if not _mcp_clients:
        return json.dumps({
            "success": True,
            "connections": [],
            "message": "No active MCP connections. Use mcp_connect to connect to a server.",
        }, ensure_ascii=False, indent=2)
    
    connections = []
    for name, info in _mcp_clients.items():
        connections.append({
            "server_name": name,
            "transport": info.get("transport"),
            "url": info.get("url"),
            "tools": info.get("tools", []),
            "resources": info.get("resources", []),
            "tool_count": len(info.get("tools", [])),
            "resource_count": len(info.get("resources", [])),
        })
    
    return json.dumps({
        "success": True,
        "connections": connections,
        "connection_count": len(connections),
    }, ensure_ascii=False, indent=2)


def _mcp_list_tools(ctx: ToolContext, server_name: str) -> str:
    """List available tools from a connected MCP server.
    
    Args:
        server_name: Name of the MCP server
    
    Returns:
        JSON with list of available tools
    """
    global _mcp_clients
    
    if server_name not in _mcp_clients:
        return json.dumps({
            "success": False,
            "error": f"Not connected to '{server_name}'. Use mcp_connect first.",
        }, ensure_ascii=False, indent=2)
    
    tools = _mcp_clients[server_name].get("tools", [])
    return json.dumps({
        "success": True,
        "server_name": server_name,
        "tools": tools,
        "count": len(tools),
    }, ensure_ascii=False, indent=2)


def _mcp_call(ctx: ToolContext, server_name: str, tool_name: str, 
              arguments: Optional[dict] = None) -> str:
    """Call a tool on a connected MCP server.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
    
    Returns:
        JSON with the tool's response
    """
    global _mcp_clients
    
    if server_name not in _mcp_clients:
        return json.dumps({
            "success": False,
            "error": f"Not connected to '{server_name}'. Use mcp_connect first.",
        }, ensure_ascii=False, indent=2)
    
    if tool_name not in _mcp_clients[server_name].get("tools", []):
        available = _mcp_clients[server_name].get("tools", [])
        return json.dumps({
            "success": False,
            "error": f"Tool '{tool_name}' not found. Available: {available}",
        }, ensure_ascii=False, indent=2)
    
    try:
        import asyncio
        session = _mcp_clients[server_name]["session"]
        
        async def call_tool():
            result = await session.call_tool(tool_name, arguments or {})
            # Convert result to readable format
            if hasattr(result, 'content'):
                return {
                    "success": True,
                    "content": [str(c) for c in result.content] if result.content else [],
                    "isError": getattr(result, 'isError', False),
                }
            return {"success": True, "result": str(result)}
        
        result = asyncio.run(call_tool())
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        log.error(f"MCP tool call error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Tool call failed: {type(e).__name__}: {e}",
        }, ensure_ascii=False, indent=2)


def _mcp_list_resources(ctx: ToolContext, server_name: str) -> str:
    """List available resources from a connected MCP server.
    
    Args:
        server_name: Name of the MCP server
    
    Returns:
        JSON with list of available resources
    """
    global _mcp_clients
    
    if server_name not in _mcp_clients:
        return json.dumps({
            "success": False,
            "error": f"Not connected to '{server_name}'. Use mcp_connect first.",
        }, ensure_ascii=False, indent=2)
    
    resources = _mcp_clients[server_name].get("resources", [])
    return json.dumps({
        "success": True,
        "server_name": server_name,
        "resources": resources,
        "count": len(resources),
    }, ensure_ascii=False, indent=2)


def _mcp_read_resource(ctx: ToolContext, server_name: str, resource_uri: str) -> str:
    """Read a resource from a connected MCP server.
    
    Args:
        server_name: Name of the MCP server
        resource_uri: URI of the resource to read
    
    Returns:
        JSON with the resource content
    """
    global _mcp_clients
    
    if server_name not in _mcp_clients:
        return json.dumps({
            "success": False,
            "error": f"Not connected to '{server_name}'. Use mcp_connect first.",
        }, ensure_ascii=False, indent=2)
    
    if resource_uri not in _mcp_clients[server_name].get("resources", []):
        available = _mcp_clients[server_name].get("resources", [])
        return json.dumps({
            "success": False,
            "error": f"Resource '{resource_uri}' not found. Available: {available}",
        }, ensure_ascii=False, indent=2)
    
    try:
        import asyncio
        session = _mcp_clients[server_name]["session"]
        
        async def read_resource():
            result = await session.read_resource(resource_uri)
            if hasattr(result, 'contents') and result.contents:
                contents = []
                for c in result.contents:
                    contents.append({
                        "uri": str(c.uri),
                        "mimeType": getattr(c, 'mimeType', None),
                        "text": getattr(c, 'text', None),
                        "blob": getattr(c, 'blob', None),
                    })
                return {"success": True, "contents": contents}
            return {"success": True, "result": str(result)}
        
        result = asyncio.run(read_resource())
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        log.error(f"MCP resource read error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Resource read failed: {type(e).__name__}: {e}",
        }, ensure_ascii=False, indent=2)


def get_tools() -> list[ToolEntry]:
    return [
        ToolEntry("mcp_connect", {
            "name": "mcp_connect",
            "description": "Connect to an MCP (Model Context Protocol) server and cache the connection. MCP is a standardized protocol for connecting AI agents to external tools and data sources. Use this to connect to databases, APIs, file systems, or other MCP-enabled services. After connecting, use mcp_list_tools to see available tools.",
            "parameters": {"type": "object", "properties": {
                "server_name": {"type": "string", "description": "A friendly name to identify this connection (e.g., 'filesystem', 'postgres', 'github')"},
                "transport": {"type": "string", "enum": ["stdio", "http"], "default": "stdio", "description": "Transport type: 'stdio' for local subprocess, 'http' for remote server"},
                "command": {"type": "string", "description": "Command to run for stdio transport (e.g., 'npx', 'python', 'node')"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments for the command (e.g., ['-m', 'mcp-server-filesystem', '/path/to/dir'])"},
                "url": {"type": "string", "description": "URL for HTTP transport (e.g., 'http://localhost:8000/mcp')"},
                "env": {"type": "object", "description": "Environment variables for stdio transport"},
            }, "required": ["server_name", "transport"]},
        }, _mcp_connect, is_code_tool=False, timeout_sec=60),
        
        ToolEntry("mcp_disconnect", {
            "name": "mcp_disconnect",
            "description": "Disconnect from an MCP server. Closes the connection and frees resources.",
            "parameters": {"type": "object", "properties": {
                "server_name": {"type": "string", "description": "Name of the server to disconnect from"},
            }, "required": ["server_name"]},
        }, _mcp_disconnect, is_code_tool=False, timeout_sec=30),
        
        ToolEntry("mcp_list", {
            "name": "mcp_list",
            "description": "List all active MCP server connections. Shows which servers are connected and what tools/resources they expose.",
            "parameters": {"type": "object", "properties": {}},
        }, _mcp_list, is_code_tool=False, timeout_sec=10),
        
        ToolEntry("mcp_list_tools", {
            "name": "mcp_list_tools",
            "description": "List available tools from a connected MCP server. Shows what actions you can perform on that server.",
            "parameters": {"type": "object", "properties": {
                "server_name": {"type": "string", "description": "Name of the MCP server"},
            }, "required": ["server_name"]},
        }, _mcp_list_tools, is_code_tool=False, timeout_sec=10),
        
        ToolEntry("mcp_call", {
            "name": "mcp_call",
            "description": "Call a tool on a connected MCP server. Use mcp_list_tools first to see available tools, then call them with appropriate arguments.",
            "parameters": {"type": "object", "properties": {
                "server_name": {"type": "string", "description": "Name of the MCP server"},
                "tool_name": {"type": "string", "description": "Name of the tool to call"},
                "arguments": {"type": "object", "description": "Arguments to pass to the tool"},
            }, "required": ["server_name", "tool_name"]},
        }, _mcp_call, is_code_tool=False, timeout_sec=120),
        
        ToolEntry("mcp_list_resources", {
            "name": "mcp_list_resources",
            "description": "List available resources from a connected MCP server. Resources are data sources like files, database tables, or API endpoints.",
            "parameters": {"type": "object", "properties": {
                "server_name": {"type": "string", "description": "Name of the MCP server"},
            }, "required": ["server_name"]},
        }, _mcp_list_resources, is_code_tool=False, timeout_sec=10),
        
        ToolEntry("mcp_read_resource", {
            "name": "mcp_read_resource",
            "description": "Read a resource from a connected MCP server. Use mcp_list_resources first to see available resources.",
            "parameters": {"type": "object", "properties": {
                "server_name": {"type": "string", "description": "Name of the MCP server"},
                "resource_uri": {"type": "string", "description": "URI of the resource to read"},
            }, "required": ["server_name", "resource_uri"]},
        }, _mcp_read_resource, is_code_tool=False, timeout_sec=30),
    ]
