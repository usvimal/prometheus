"""Research tools - web research and synthesis without browser dependencies.

Uses simple HTTP requests to DuckDuckGo HTML and textise dot iitty for search,
and urllib for page fetching. Falls back gracefully if blocked.

Tools:
- research_search: Web search using multiple backends
- research_fetch: Fetch and extract content from a specific URL  
- research_synthesize: Store research findings in knowledge base
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry


def _ddg_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Search using DuckDuckGo HTML interface."""
    results = []
    try:
        import urllib.request
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        
        # Parse results - DuckDuckGo HTML format
        link_pattern = r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]*)</a>'
        
        links = re.findall(link_pattern, html)
        snippets = re.findall(snippet_pattern, html)
        
        for i, (url, title) in enumerate(links[:num_results]):
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            if url.startswith("http"):
                results.append({
                    "url": url[:300],
                    "title": title[:200] if title else "No title",
                    "snippet": snippet[:300] if snippet else ""
                })
    except Exception as e:
        return [{"error": str(e)}]
    return results


def _fetch_url(url: str, output: str = "text") -> str:
    """Fetch a URL using urllib."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        
        if output == "html":
            return html[:50000] + ("... [truncated]" if len(html) > 50000 else "")
        
        # Extract text from HTML
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")
    except Exception as e:
        return f"Error fetching {url}: {e}"


def _research_search(ctx: ToolContext, query: str, num_results: int = 10) -> str:
    """Search the web for information.
    
    Uses DuckDuckGo HTML. Returns JSON with results.
    """
    results = _ddg_search(query, num_results)
    
    if not results:
        return json.dumps({
            "query": query,
            "results": [],
            "error": "No results found"
        }, ensure_ascii=False, indent=2)
    
    return json.dumps({
        "query": query,
        "results": results,
        "count": len(results)
    }, ensure_ascii=False, indent=2)


def _research_fetch(ctx: ToolContext, url: str, output: str = "text") -> str:
    """Fetch content from a specific URL.
    
    Returns text or HTML from the page.
    """
    content = _fetch_url(url, output)
    return content


def _research_synthesize(ctx: ToolContext, topic: str, content: str, 
                         source_urls: str = "") -> str:
    """Store research findings in knowledge base.
    
    Appends research to a topic in the knowledge base.
    """
    try:
        # Sanitize topic name
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', topic.lower())
        sanitized = re.sub(r'-+', '-', sanitized).strip('-')
        if not sanitized:
            sanitized = "research"
        if sanitized.startswith(("system", "identity", "scratch")):
            sanitized = "research-" + sanitized
        
        # Format the content with sources
        from prometheus.utils import utc_now_iso
        timestamp = utc_now_iso()[:10]
        
        research_entry = f"""
## Research: {topic} ({timestamp})

{content}
"""
        if source_urls:
            research_entry += f"\n### Sources\n"
            for url in source_urls.split(","):
                url = url.strip()
                if url.startswith("http"):
                    research_entry += f"- {url}\n"
        
        # Append to knowledge topic
        topic_path = ctx.drive_path(f"memory/knowledge/{sanitized}.md")
        topic_path.parent.mkdir(parents=True, exist_ok=True)
        
        existing = ""
        if topic_path.exists():
            existing = topic_path.read_text(errors="ignore")
        
        # Add header if new
        if not existing:
            existing = f"# {sanitized}\n\n"
        
        full_content = existing + "\n" + research_entry
        topic_path.write_text(full_content, errors="ignore")
        
        return json.dumps({
            "topic": sanitized,
            "status": "saved",
            "path": str(topic_path),
            "size": len(full_content)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "topic": topic,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("research_search", {
            "name": "research_search",
            "description": "Search the web for information. Uses DuckDuckGo HTML. Returns JSON with results including URLs, titles, and snippets.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Maximum number of results (default 10)", "default": 10}
            }, "required": ["query"]},
        }, _research_search),
        ToolEntry("research_fetch", {
            "name": "research_fetch",
            "description": "Fetch content from a specific URL. Returns text or HTML from the page.",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "output": {"type": "string", "description": "Output format: text or html", "default": "text"}
            }, "required": ["url"]},
        }, _research_fetch),
        ToolEntry("research_synthesize", {
            "name": "research_synthesize",
            "description": "Store research findings in knowledge base. Appends research to a topic for future reference.",
            "parameters": {"type": "object", "properties": {
                "topic": {"type": "string", "description": "Topic name for the research"},
                "content": {"type": "string", "description": "Research findings to store"},
                "source_urls": {"type": "string", "description": "Comma-separated source URLs", "default": ""}
            }, "required": ["topic", "content"]},
        }, _research_synthesize),
    ]
