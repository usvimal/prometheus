"""
Browser tools: tiered fetching with Scrapling (fast HTTP) + browser-use (full browser).

browse_page tries Scrapling first (HTTP with TLS fingerprinting, ~5MB RAM, <1s)
and falls back to browser-use (full Chromium, ~200MB RAM, 3-10s) only when needed.

browser_action always uses browser-use (needs a real browser for click/fill/JS).

Tier 1 — Scrapling Fetcher: HTTP + TLS spoofing. Handles 90% of page reads.
Tier 2 — browser-use CDP: Full Chromium with anti-detection extensions. For JS-heavy
         sites, screenshots, wait_for selectors, and interactive actions.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# Module-level browser-use Browser instance (reused across tasks)
_browser_instance = None
_event_loop = None

# Minimum text length to consider Scrapling result "good enough"
_MIN_SCRAPLING_TEXT = 100


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for async browser-use calls."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop


def _run_async(coro):
    """Run an async coroutine synchronously using our persistent loop."""
    loop = _get_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tier 1: Scrapling (fast HTTP with TLS fingerprinting)
# ---------------------------------------------------------------------------

def _scrapling_fetch(url: str, output: str = "text",
                     timeout: int = 15) -> Optional[str]:
    """Fetch a page via Scrapling HTTP (no browser). Returns None on failure."""
    try:
        from scrapling import Fetcher
        fetcher = Fetcher(timeout=timeout)
        page = fetcher.get(url)

        if page.status != 200:
            log.debug("Scrapling got status %d for %s, deferring to browser", page.status, url)
            return None

        if output == "html":
            html = page.html_content or ""
            if not html and page.body:
                html = page.body.decode("utf-8", errors="replace")
            if len(html) < _MIN_SCRAPLING_TEXT:
                return None
            return html[:50000] + ("... [truncated]" if len(html) > 50000 else "")

        elif output == "markdown":
            text = _html_to_markdown(page)
            if len(text) < _MIN_SCRAPLING_TEXT:
                return None
            return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")

        else:  # text
            text = page.get_all_text() or ""
            if len(text.strip()) < _MIN_SCRAPLING_TEXT:
                return None
            return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")

    except ImportError:
        log.debug("Scrapling not installed, skipping fast path")
        return None
    except Exception as e:
        log.debug("Scrapling fetch failed for %s: %s", url, e)
        return None


def _html_to_markdown(page) -> str:
    """Convert Scrapling page to crude markdown using its DOM API."""
    parts = []
    try:
        # Extract headings
        for level in range(1, 7):
            for h in page.find_all(f"h{level}"):
                text = h.get_all_text().strip()
                if text:
                    parts.append(f"{'#' * level} {text}")

        # Extract paragraphs and list items
        for p in page.find_all("p"):
            text = p.get_all_text().strip()
            if text:
                parts.append(text)

        for li in page.find_all("li"):
            text = li.get_all_text().strip()
            if text:
                parts.append(f"- {text}")

        # Extract links
        for a in page.find_all("a"):
            href = a.attrib.get("href", "")
            text = a.get_all_text().strip()
            if text and href and href.startswith("http"):
                parts.append(f"[{text}]({href})")

    except Exception:
        pass

    if not parts:
        # Fallback: just get all text
        return page.get_all_text() or ""

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tier 2: browser-use (full Chromium with CDP + anti-detection)
# ---------------------------------------------------------------------------

def _ensure_browser_installed():
    """Install browser-use chromium if not available."""
    try:
        _run_async(_check_browser())
    except Exception:
        log.info("Installing browser-use browser...")
        subprocess.check_call(
            [sys.executable, "-m", "browser_use", "install"],
            timeout=120,
        )


async def _check_browser():
    """Quick check that browser-use can launch."""
    from browser_use import Browser
    b = Browser(headless=True)
    await b.start()
    await b.stop()


async def _start_browser():
    """Start a browser-use Browser instance with anti-detection defaults."""
    from browser_use import Browser

    browser = Browser(
        headless=True,
        window_size={"width": 1920, "height": 1080},
        enable_default_extensions=True,  # uBlock Origin, cookie handlers, ClearURLs
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    await browser.start()
    return browser


async def _ensure_browser_async(ctx: ToolContext):
    """Create or reuse browser + page for this task."""
    global _browser_instance

    # Check if existing browser is still alive
    if ctx.browser_state.browser is not None:
        try:
            pages = await ctx.browser_state.browser.get_pages()
            if pages:
                return ctx.browser_state.page
        except Exception:
            log.debug("Browser connection check failed", exc_info=True)
            await _cleanup_browser_async(ctx)

    # Start or reuse module-level browser
    if _browser_instance is None:
        _browser_instance = await _start_browser()
        log.info("browser-use Browser started with anti-detection defaults")

    ctx.browser_state.browser = _browser_instance

    # Create a new page
    page = await _browser_instance.new_page("about:blank")
    ctx.browser_state.page = page
    return page


async def _cleanup_browser_async(ctx: ToolContext):
    """Close page for this context. Keep browser alive for reuse."""
    try:
        if ctx.browser_state.page is not None:
            await ctx.browser_state.browser.close_page(ctx.browser_state.page)
    except Exception:
        log.debug("Failed to close page during cleanup", exc_info=True)
    ctx.browser_state.page = None
    ctx.browser_state.browser = None


def cleanup_browser(ctx: ToolContext) -> None:
    """Close browser page. Called by agent.py in finally block."""
    try:
        _run_async(_cleanup_browser_async(ctx))
    except Exception:
        log.debug("Failed to cleanup browser", exc_info=True)
    ctx.browser_state.page = None
    ctx.browser_state.browser = None


_MARKDOWN_JS = """() => {
    const walk = (el) => {
        let out = '';
        for (const child of el.childNodes) {
            if (child.nodeType === 3) {
                const t = child.textContent.trim();
                if (t) out += t + ' ';
            } else if (child.nodeType === 1) {
                const tag = child.tagName;
                if (['SCRIPT','STYLE','NOSCRIPT'].includes(tag)) continue;
                if (['H1','H2','H3','H4','H5','H6'].includes(tag))
                    out += '\\n' + '#'.repeat(parseInt(tag[1])) + ' ';
                if (tag === 'P' || tag === 'DIV' || tag === 'BR') out += '\\n';
                if (tag === 'LI') out += '\\n- ';
                if (tag === 'A') out += '[';
                out += walk(child);
                if (tag === 'A') out += '](' + (child.href||'') + ')';
            }
        }
        return out;
    };
    return walk(document.body);
}"""


async def _extract_page_output(page: Any, output: str, ctx: ToolContext) -> str:
    """Extract page content in the requested format."""
    if output == "screenshot":
        data = await page.screenshot()
        # browser-use may return base64 string or raw bytes depending on version
        if isinstance(data, str):
            b64 = data
        else:
            b64 = base64.b64encode(data).decode()
        ctx.browser_state.last_screenshot_b64 = b64
        return (
            f"Screenshot captured ({len(b64)} bytes base64). "
            f"Call send_photo(image_base64='__last_screenshot__') to deliver it to the owner."
        )
    elif output == "html":
        html = await page.evaluate("() => document.documentElement.outerHTML")
        return html[:50000] + ("... [truncated]" if len(html) > 50000 else "")
    elif output == "markdown":
        text = await page.evaluate(_MARKDOWN_JS)
        return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")
    else:  # text
        text = await page.evaluate("() => document.body.innerText")
        return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")


async def _browse_page_async(ctx: ToolContext, url: str, output: str = "text",
                              wait_for: str = "", timeout: int = 30000) -> str:
    page = await _ensure_browser_async(ctx)
    await page.goto(url)
    # Give page time to load
    await asyncio.sleep(1)
    if wait_for:
        # Use JS to wait for selector
        await page.evaluate(
            f"""() => new Promise((resolve, reject) => {{
                const timeout = setTimeout(() => reject(new Error('Timeout waiting for {wait_for}')), {timeout});
                const check = () => {{
                    if (document.querySelector('{wait_for}')) {{
                        clearTimeout(timeout);
                        resolve(true);
                    }} else {{
                        requestAnimationFrame(check);
                    }}
                }};
                check();
            }})"""
        )
    return await _extract_page_output(page, output, ctx)


async def _browser_action_async(ctx: ToolContext, action: str, selector: str = "",
                                 value: str = "", timeout: int = 5000) -> str:
    page = await _ensure_browser_async(ctx)

    if action == "click":
        if not selector:
            return "Error: selector required for click"
        elements = await page.get_elements_by_css_selector(selector)
        if not elements:
            return f"Error: no elements found matching '{selector}'"
        await elements[0].click()
        await asyncio.sleep(0.5)
        return f"Clicked: {selector}"
    elif action == "fill":
        if not selector:
            return "Error: selector required for fill"
        elements = await page.get_elements_by_css_selector(selector)
        if not elements:
            return f"Error: no elements found matching '{selector}'"
        await elements[0].fill(value)
        return f"Filled {selector} with: {value}"
    elif action == "select":
        if not selector:
            return "Error: selector required for select"
        # Use JS for select since browser-use doesn't have native select_option
        await page.evaluate(
            f"""() => {{
                const el = document.querySelector('{selector}');
                if (el) {{ el.value = '{value}'; el.dispatchEvent(new Event('change')); }}
            }}"""
        )
        return f"Selected {value} in {selector}"
    elif action == "screenshot":
        data = await page.screenshot()
        if isinstance(data, str):
            b64 = data
        else:
            b64 = base64.b64encode(data).decode()
        ctx.browser_state.last_screenshot_b64 = b64
        return (
            f"Screenshot captured ({len(b64)} bytes base64). "
            f"Call send_photo(image_base64='__last_screenshot__') to deliver it to the owner."
        )
    elif action == "evaluate":
        if not value:
            return "Error: value (JS code) required for evaluate"
        # Wrap in arrow function if not already
        js = value if value.strip().startswith("(") or value.strip().startswith("function") else f"() => {{ {value} }}"
        result = await page.evaluate(js)
        out = str(result)
        return out[:20000] + ("... [truncated]" if len(out) > 20000 else "")
    elif action == "scroll":
        direction = value or "down"
        if direction == "down":
            await page.evaluate("() => window.scrollBy(0, 600)")
        elif direction == "up":
            await page.evaluate("() => window.scrollBy(0, -600)")
        elif direction == "top":
            await page.evaluate("() => window.scrollTo(0, 0)")
        elif direction == "bottom":
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        return f"Scrolled {direction}"
    else:
        return f"Unknown action: {action}. Use: click, fill, select, screenshot, evaluate, scroll"


# ---------------------------------------------------------------------------
# browse_page: tiered fetching
# ---------------------------------------------------------------------------

def _needs_browser(output: str, wait_for: str) -> bool:
    """Return True if this request must use the full browser."""
    if output == "screenshot":
        return True
    if wait_for:
        return True
    return False


def _browse_page(ctx: ToolContext, url: str, output: str = "text",
                 wait_for: str = "", timeout: int = 30000) -> str:
    """Tiered page fetching: Scrapling HTTP first, browser-use fallback."""

    # Tier 2 direct: screenshot or wait_for require real browser
    if _needs_browser(output, wait_for):
        return _browse_page_browser(ctx, url, output, wait_for, timeout)

    # Tier 1: try Scrapling fast HTTP fetch
    scrapling_timeout = min(timeout // 1000, 15)  # convert ms to seconds, cap at 15
    result = _scrapling_fetch(url, output, scrapling_timeout)

    if result is not None:
        log.info("browse_page served via Scrapling (fast HTTP) for %s", url)
        return f"[fetched via HTTP]\n{result}"

    # Tier 2 fallback: Scrapling returned too little content (JS-rendered page?)
    log.info("Scrapling returned insufficient content for %s, falling back to browser-use", url)
    return _browse_page_browser(ctx, url, output, wait_for, timeout)


def _browse_page_browser(ctx: ToolContext, url: str, output: str = "text",
                         wait_for: str = "", timeout: int = 30000) -> str:
    """Full browser-use fetch (Tier 2)."""
    try:
        result = _run_async(_browse_page_async(ctx, url, output, wait_for, timeout))
        return f"[fetched via browser]\n{result}"
    except Exception as e:
        log.warning("browse_page browser error: %s", e, exc_info=True)
        # Reset browser state and retry once
        try:
            cleanup_browser(ctx)
            result = _run_async(_browse_page_async(ctx, url, output, wait_for, timeout))
            return f"[fetched via browser]\n{result}"
        except Exception as e2:
            return f"Error: {e2}"


def _browser_action(ctx: ToolContext, action: str, selector: str = "",
                    value: str = "", timeout: int = 5000) -> str:
    try:
        return _run_async(_browser_action_async(ctx, action, selector, value, timeout))
    except Exception as e:
        log.warning("browser_action error: %s", e, exc_info=True)
        try:
            cleanup_browser(ctx)
            return _run_async(_browser_action_async(ctx, action, selector, value, timeout))
        except Exception as e2:
            return f"Error: {e2}"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="browse_page",
            schema={
                "name": "browse_page",
                "description": (
                    "Open a URL and get its content. Uses fast HTTP fetch with TLS spoofing "
                    "for most pages (~1s), auto-falls back to full stealth browser for JS-heavy "
                    "sites or screenshots. Returns text, html, markdown, or screenshot. "
                    "For screenshots: use send_photo tool to deliver the image to owner."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to open"},
                        "output": {
                            "type": "string",
                            "enum": ["text", "html", "markdown", "screenshot"],
                            "description": "Output format (default: text)",
                        },
                        "wait_for": {
                            "type": "string",
                            "description": "CSS selector to wait for before extraction (forces full browser)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Page load timeout in ms (default: 30000)",
                        },
                    },
                    "required": ["url"],
                },
            },
            handler=_browse_page,
            timeout_sec=60,
        ),
        ToolEntry(
            name="browser_action",
            schema={
                "name": "browser_action",
                "description": (
                    "Perform action on current stealth browser page. Actions: "
                    "click (selector), fill (selector + value), select (selector + value), "
                    "screenshot (base64 PNG), evaluate (JS code in value), "
                    "scroll (value: up/down/top/bottom)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["click", "fill", "select", "screenshot", "evaluate", "scroll"],
                            "description": "Action to perform",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for click/fill/select",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value for fill/select, JS for evaluate, direction for scroll",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Action timeout in ms (default: 5000)",
                        },
                    },
                    "required": ["action"],
                },
            },
            handler=_browser_action,
            timeout_sec=60,
        ),
    ]
