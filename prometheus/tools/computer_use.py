"""Computer use tools - control the desktop (mouse, keyboard, screenshot).

This module provides computer use capabilities similar to Claude Computer Use:
- Screen capture
- Mouse control (move, click, drag, scroll)
- Keyboard control (type, press keys, hotkeys)
- Window management

Uses mss for screenshots and pyautogui for input control.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from prometheus.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# Track if we're in a headless environment
_headless = False


def _check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    deps = {}
    try:
        import mss
        deps["mss"] = True
    except ImportError:
        deps["mss"] = False
    
    try:
        import pyautogui
        deps["pyautogui"] = True
    except ImportError:
        deps["pyautogui"] = False
    
    try:
        import Xlib
        deps["python_xlib"] = True
    except ImportError:
        deps["python_xlib"] = False
    
    return deps


def _get_screen_size() -> Dict[str, int]:
    """Get screen resolution."""
    global _headless
    try:
        import pyautogui
        size = pyautogui.size()
        return {"width": size.width, "height": size.height}
    except Exception as e:
        _headless = True
        return {"error": str(e), "headless": True}


def _computer_screenshot(ctx: ToolContext, output: str = "base64", region: str = "") -> str:
    """Capture a screenshot of the screen or region.
    
    Args:
        output: "base64" (default) for analysis, "path" to save to file
        region: "full" (default), or "WIDTHxHEIGHT+X+Y" (e.g., "800x600+100+50")
    
    Returns:
        Base64 PNG or path to saved file
    """
    global _headless
    
    try:
        import mss
        import numpy as np
    except ImportError as e:
        return json.dumps({
            "success": False,
            "error": f"Missing dependency: {e}. Install: pip install mss numpy",
        }, ensure_ascii=False)
    
    try:
        with mss.mss() as sct:
            # Determine region
            if region and region != "full":
                # Parse WIDTHxHEIGHT+X+Y
                parts = region.split("+")
                if len(parts) == 3:
                    size_parts = parts[0].split("x")
                    if len(size_parts) == 2:
                        width, height = int(size_parts[0]), int(size_parts[1])
                        x, y = int(parts[1]), int(parts[2])
                        monitor = {"left": x, "top": y, "width": width, "height": height}
                    else:
                        monitor = sct.monitors[1]
                else:
                    monitor = sct.monitors[1]
            else:
                # Full screen
                monitor = sct.monitors[1]
            
            # Capture
            sct_img = sct.grab(monitor)
            
            # Convert to PNG
            import io
            img_bytes = mss.tools.to_png(sct_img.rgb, sct_img.size)
            
            if output == "path":
                import os
                import uuid
                filename = f"/tmp/screenshot_{uuid.uuid4().hex[:8]}.png"
                with open(filename, "wb") as f:
                    f.write(img_bytes)
                return json.dumps({
                    "success": True,
                    "path": filename,
                    "size": len(img_bytes),
                    "region": region or "full",
                }, ensure_ascii=False)
            else:
                import base64
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                # Store in browser state for analysis
                ctx.browser_state.last_screenshot_b64 = b64
                return json.dumps({
                    "success": True,
                    "base64": b64[:100] + "...[truncated]",
                    "full_base64_length": len(b64),
                    "region": region or "full",
                    "note": "Full base64 stored in context for analyze_screenshot",
                }, ensure_ascii=False)
                
    except Exception as e:
        log.error("Screenshot failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


def _computer_mouse(ctx: ToolContext, action: str, x: int = 0, y: int = 0, 
                    clicks: int = 1, interval: float = 0.0, button: str = "left",
                    direction: str = "", amount: int = 0) -> str:
    """Control the mouse.
    
    Actions:
    - "position": get current position
    - "move": move to x,y
    - "click": click at x,y (or current position)
    - "double_click": double click
    - "right_click": right click
    - "drag": drag from current to x,y
    - "scroll": scroll up/down by amount
    
    Args:
        action: The mouse action to perform
        x, y: Target coordinates (0 = current position)
        clicks: Number of clicks
        interval: Interval between clicks
        button: "left", "right", "middle"
        direction: "up" or "down" for scroll
        amount: Pixels to scroll
    """
    global _headless
    
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
    except ImportError as e:
        return json.dumps({
            "success": False,
            "error": f"Missing dependency: {e}. Install: pip install pyautogui",
        }, ensure_ascii=False)
    except Exception as e:
        if "headless" in str(e).lower():
            _headless = True
            return json.dumps({
                "success": False,
                "error": "Headless environment detected. Computer use not available.",
                "headless": True,
            }, ensure_ascii=False)
    
    try:
        if action == "position":
            pos = pyautogui.position()
            return json.dumps({
                "success": True,
                "x": pos.x,
                "y": pos.y,
            }, ensure_ascii=False)
        
        elif action == "move":
            # x=0 means current position
            target_x = x if x > 0 else pyautogui.position().x
            target_y = y if y > 0 else pyautogui.position().y
            pyautogui.moveTo(target_x, target_y, duration=0.2)
            return json.dumps({
                "success": True,
                "x": target_x,
                "y": target_y,
                "action": "moved",
            }, ensure_ascii=False)
        
        elif action == "click":
            target_x = x if x > 0 else None
            target_y = y if y > 0 else None
            pyautogui.click(x=target_x, y=target_y, clicks=clicks, 
                          interval=interval, button=button)
            return json.dumps({
                "success": True,
                "x": target_x or pyautogui.position().x,
                "y": target_y or pyautogui.position().y,
                "clicks": clicks,
                "button": button,
                "action": "clicked",
            }, ensure_ascii=False)
        
        elif action == "double_click":
            target_x = x if x > 0 else None
            target_y = y if y > 0 else None
            pyautogui.doubleClick(x=target_x, y=target_y, button=button)
            return json.dumps({
                "success": True,
                "action": "double_clicked",
            }, ensure_ascii=False)
        
        elif action == "right_click":
            target_x = x if x > 0 else None
            target_y = y if y > 0 else None
            pyautogui.rightClick(x=target_x, y=target_y)
            return json.dumps({
                "success": True,
                "action": "right_clicked",
            }, ensure_ascii=False)
        
        elif action == "drag":
            # Drag from current position to target
            start_pos = pyautogui.position()
            target_x = x if x > 0 else start_pos.x
            target_y = y if y > 0 else start_pos.y
            pyautogui.dragTo(target_x, target_y, duration=0.5, button=button)
            return json.dumps({
                "success": True,
                "from_x": start_pos.x,
                "from_y": start_pos.y,
                "to_x": target_x,
                "to_y": target_y,
                "action": "dragged",
            }, ensure_ascii=False)
        
        elif action == "scroll":
            if direction == "up":
                pyautogui.scroll(amount)
            elif direction == "down":
                pyautogui.scroll(-amount)
            else:
                pyautogui.scroll(amount)
            return json.dumps({
                "success": True,
                "direction": direction or "up",
                "amount": amount,
                "action": "scrolled",
            }, ensure_ascii=False)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action: {action}. Use: position, move, click, double_click, right_click, drag, scroll",
            }, ensure_ascii=False)
            
    except Exception as e:
        log.error("Mouse action failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


def _computer_keyboard(ctx: ToolContext, action: str, text: str = "", 
                       key: str = "", keys: str = "") -> str:
    """Control the keyboard.
    
    Actions:
    - "type": type text string
    - "press": press a single key
    - "hotkey": press multiple keys simultaneously (e.g., "ctrl+c")
    - "write": type with interval between each key
    
    Args:
        action: The keyboard action
        text: Text to type
        key: Single key to press
        keys: Comma-separated keys for hotkey (e.g., "ctrl,c")
    """
    global _headless
    
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
    except ImportError as e:
        return json.dumps({
            "success": False,
            "error": f"Missing dependency: {e}. Install: pip install pyautogui",
        }, ensure_ascii=False)
    except Exception as e:
        if "headless" in str(e).lower():
            _headless = True
            return json.dumps({
                "success": False,
                "error": "Headless environment detected. Computer use not available.",
                "headless": True,
            }, ensure_ascii=False)
    
    try:
        if action == "type":
            if not text:
                return json.dumps({
                    "success": False,
                    "error": "text parameter required for 'type' action",
                }, ensure_ascii=False)
            pyautogui.write(text, interval=0.05)
            return json.dumps({
                "success": True,
                "text": text,
                "action": "typed",
                "char_count": len(text),
            }, ensure_ascii=False)
        
        elif action == "write":
            # Type with configurable interval
            interval = 0.1
            if text:
                pyautogui.write(text, interval=interval)
            return json.dumps({
                "success": True,
                "action": "written",
            }, ensure_ascii=False)
        
        elif action == "press":
            if not key:
                return json.dumps({
                    "success": False,
                    "error": "key parameter required for 'press' action",
                }, ensure_ascii=False)
            pyautogui.press(key)
            return json.dumps({
                "success": True,
                "key": key,
                "action": "pressed",
            }, ensure_ascii=False)
        
        elif action == "hotkey":
            if not keys:
                return json.dumps({
                    "success": False,
                    "error": "keys parameter required for 'hotkey' action (comma-separated)",
                }, ensure_ascii=False)
            key_list = [k.strip() for k in keys.split(",")]
            pyautogui.hotkey(*key_list)
            return json.dumps({
                "success": True,
                "keys": key_list,
                "action": "hotkey",
            }, ensure_ascii=False)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action: {action}. Use: type, press, hotkey",
            }, ensure_ascii=False)
            
    except Exception as e:
        log.error("Keyboard action failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


def _computer_list_windows(ctx: ToolContext) -> str:
    """List open windows (Linux/Unix only).
    
    Returns:
        JSON with list of window titles and IDs
    """
    try:
        import subprocess
        # Try xdotool on Linux
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", ".*"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            window_ids = result.stdout.strip().split("\n")
            windows = []
            for wid in window_ids:
                if wid.strip():
                    # Get window name
                    name_result = subprocess.run(
                        ["xdotool", "getwindowname", wid.strip()],
                        capture_output=True, text=True, timeout=2
                    )
                    if name_result.returncode == 0:
                        windows.append({
                            "id": wid.strip(),
                            "title": name_result.stdout.strip()
                        })
            return json.dumps({
                "success": True,
                "windows": windows,
                "count": len(windows),
            }, ensure_ascii=False)
    except FileNotFoundError:
        pass
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)
    
    # macOS / Windows fallback (basic)
    try:
        import pyautogui
        # pyautogui doesn't have window listing, but we can get screen size
        size = pyautogui.size()
        return json.dumps({
            "success": True,
            "message": "Window listing not available on this platform",
            "screen_size": {"width": size.width, "height": size.height},
        }, ensure_ascii=False)
    except Exception:
        pass
    
    return json.dumps({
        "success": False,
        "error": "Window listing requires xdotool (Linux) or is limited on other platforms",
    }, ensure_ascii=False)


def _computer_status(ctx: ToolContext) -> str:
    """Get computer use status and capabilities."""
    deps = _check_dependencies()
    screen = _get_screen_size()
    
    return json.dumps({
        "success": True,
        "available": deps.get("pyautogui", False) and deps.get("mss", False),
        "dependencies": deps,
        "screen": screen,
        "headless": _headless,
        "supported_actions": {
            "mouse": ["position", "move", "click", "double_click", "right_click", "drag", "scroll"],
            "keyboard": ["type", "press", "hotkey"],
            "screenshot": ["full", "region"],
        },
    }, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("computer_screenshot", {
            "name": "computer_screenshot",
            "description": "Capture a screenshot of the screen or a region. Returns base64 PNG (stored in context for VLM analysis) or saves to file. Region format: 'WIDTHxHEIGHT+X+Y' (e.g., '800x600+100+50'). Use analyze_screenshot after this to analyze the image.",
            "parameters": {"type": "object", "properties": {
                "output": {"type": "string", "enum": ["base64", "path"], "default": "base64", "description": "Output format: base64 for analysis, path to save file"},
                "region": {"type": "string", "default": "full", "description": "Screen region: 'full' or 'WIDTHxHEIGHT+X+Y'"},
            }, "required": []},
        }, _computer_screenshot, timeout_sec=30),
        
        ToolEntry("computer_mouse", {
            "name": "computer_mouse",
            "description": "Control the mouse: move, click, double_click, right_click, drag, scroll. Use 'position' to get current location. Coordinates: x=0 means keep current position.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "description": "Action: position, move, click, double_click, right_click, drag, scroll"},
                "x": {"type": "integer", "default": 0, "description": "Target X coordinate (0 = current)"},
                "y": {"type": "integer", "default": 0, "description": "Target Y coordinate (0 = current)"},
                "clicks": {"type": "integer", "default": 1, "description": "Number of clicks"},
                "interval": {"type": "number", "default": 0.0, "description": "Interval between clicks"},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left", "description": "Mouse button"},
                "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction"},
                "amount": {"type": "integer", "default": 0, "description": "Scroll amount in pixels"},
            }, "required": ["action"]},
        }, _computer_mouse, timeout_sec=30),
        
        ToolEntry("computer_keyboard", {
            "name": "computer_keyboard",
            "description": "Control the keyboard: type text, press a single key, or send hotkey combinations. Examples: 'type' with text, 'press' with key='enter', 'hotkey' with keys='ctrl,c'",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "description": "Action: type, write, press, hotkey"},
                "text": {"type": "string", "description": "Text to type (for type/write actions)"},
                "key": {"type": "string", "description": "Single key to press (for press action)"},
                "keys": {"type": "string", "description": "Comma-separated keys for hotkey (e.g., 'ctrl,c, v')"},
            }, "required": ["action"]},
        }, _computer_keyboard, timeout_sec=30),
        
        ToolEntry("computer_list_windows", {
            "name": "computer_list_windows",
            "description": "List open windows (requires xdotool on Linux). Returns window IDs and titles. Useful for window management and focus.",
            "parameters": {"type": "object", "properties": {}},
        }, _computer_list_windows, timeout_sec=10),
        
        ToolEntry("computer_status", {
            "name": "computer_status",
            "description": "Get computer use capabilities: dependencies available, screen size, supported actions. Use to check if computer control is available.",
            "parameters": {"type": "object", "properties": {}},
        }, _computer_status, timeout_sec=10),
    ]
