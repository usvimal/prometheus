"""Web search tools - browser-based implementation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from prometheus.tools.registry import ToolContext, ToolEntry


def _browser_search(ctx: ToolContext, query: str, num_results: int = 10) -> str:
    """Search the web using browser (DuckDuckGo Lite).
    
    This replaces the OpenAI-dependent web_search with a pure browser solution.
    Returns JSON with search results.
    """
    try:
        # Use browse_page to search via DuckDuckGo Lite
        search_url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
        
        # Import here to avoid circular deps - we'll use the function from browser module
        from prometheus.tools.browser import _browse_page
        result = _browse_page(ctx, search_url, output="text", timeout=20000)
        
        if "error" in result.lower() or "failed" in result.lower():
            # Try fallback: Bing
            search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
            result = _browse_page(ctx, search_url, output="text", timeout=20000)
        
        # Parse results from DuckDuckGo Lite format
        # Format: each result is typically a line with URL and description
        results = []
        lines = result.split('\n')
        
        # Look for result patterns in DuckDuckGo Lite
        for line in lines:
            # Skip irrelevant lines
            if not line.strip() or line.startswith('[') or line.startswith('-'):
                continue
            # DuckDuckGo Lite results typically have URLs
            if 'http' in line and len(line) > 20:
                # Extract URL and title/description
                url_match = re.search(r'https?://[^\s]+', line)
                if url_match:
                    url = url_match.group(0)[:200]  # Truncate long URLs
                    # Clean up the line for title
                    title = line.replace(url, '').strip()[:150]
                    if title and len(title) > 10:
                        results.append({
                            "url": url,
                            "title": title
                        })
                        if len(results) >= num_results:
                            break
        
        # If parsing failed, return raw result with query info
        if not results:
            return json.dumps({
                "query": query,
                "results": [],
                "note": "Parsed 0 results from browser search. Raw output truncated.",
                "raw_snippet": result[:1000]
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "query": query,
            "results": results,
            "count": len(results)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "query": query,
            "error": str(e),
            "results": []
        }, ensure_ascii=False)


def _quick_search(ctx: ToolContext, query: str) -> str:
    """Quick search for fast answers - uses textise dot iitty for minimal HTML."""
    try:
        # Use textise dot iitty for ultra-fast text-only search
        search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        from prometheus.tools.browser import _browse_page
        result = _browse_page(ctx, search_url, output="text", timeout=15000)
        
        # Extract result links from DuckDuckGo HTML format
        results = []
        # Pattern: <a class="result__a" href="URL">Title</a>
        links = re.findall(r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>', result, re.IGNORECASE)
        
        for url, title in links[:10]:
            if url.startswith('http'):
                results.append({
                    "url": url[:200],
                    "title": title.strip()[:150]
                })
        
        if not results:
            return json.dumps({
                "query": query,
                "results": [],
                "note": "No results parsed"
            }, ensure_ascii=False)
        
        return json.dumps({
            "query": query,
            "results": results,
            "count": len(results)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "query": query,
            "error": str(e),
            "results": []
        }, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("browser_search", {
            "name": "browser_search",
            "description": "Search the web using browser (DuckDuckGo). Returns JSON with results including URLs and titles. Use for general web searches.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Maximum number of results (default 10)", "default": 10}
            }, "required": ["query"]},
        }, _browser_search),
        ToolEntry("quick_search", {
            "name": "quick_search",
            "description": "Fast text-based web search. Use for quick factual queries when you need speed over completeness.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query"}
            }, "required": ["query"]},
        }, _quick_search),
    ]
