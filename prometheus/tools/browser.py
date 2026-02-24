"""
Browser automation tools via browser-use (CDP + anti-detection).

Provides browse_page (open URL, get content/screenshot)
and browser_action (click, fill, evaluate JS on current page).

Uses browser-use's Browser/Page API which provides:
- CDP-based automation (not Playwright)
- Built-in anti-detection (stealth fingerprinting, extension support)
- Default extensions (uBlock Origin, cookie handlers)

Browser state lives in ToolContext (per-task lifecycle).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import sys
from typing import Any, Dict, List

from prometheus.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# Module-level browser-use Browser instance (reused across tasks)
_browser_instance = None
_event_loop = None


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


def _browse_page(ctx: ToolContext, url: str, output: str = "text",
                 wait_for: str = "", timeout: int = 30000) -> str:
    try:
        return _run_async(_browse_page_async(ctx, url, output, wait_for, timeout))
    except Exception as e:
        log.warning(f"browse_page error: {e}", exc_info=True)
        # Reset browser state and retry once
        try:
            cleanup_browser(ctx)
            return _run_async(_browse_page_async(ctx, url, output, wait_for, timeout))
        except Exception as e2:
            return f"Error: {e2}"


def _browser_action(ctx: ToolContext, action: str, selector: str = "",
                    value: str = "", timeout: int = 5000) -> str:
    try:
        return _run_async(_browser_action_async(ctx, action, selector, value, timeout))
    except Exception as e:
        log.warning(f"browser_action error: {e}", exc_info=True)
        try:
            cleanup_browser(ctx)
            return _run_async(_browser_action_async(ctx, action, selector, value, timeout))
        except Exception as e2:
            return f"Error: {e2}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="browse_page",
            schema={
                "name": "browse_page",
                "description": (
                    "Open a URL in stealth headless browser (browser-use with anti-detection). "
                    "Returns page content as text, html, markdown, or screenshot (base64 PNG). "
                    "Browser persists across calls within a task. "
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
                            "description": "CSS selector to wait for before extraction",
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
