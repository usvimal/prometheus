"""HTTP request tool - make HTTP requests to external APIs."""
import json
import urllib.request
import urllib.error
from typing import Any, List

from prometheus.tools.registry import ToolContext, ToolEntry


def _http_request(ctx: ToolContext, url: str, method: str = "GET", headers: dict = None, body: Any = None, timeout: int = 30) -> str:
    """Make an HTTP request to an external URL."""
    if headers is None:
        headers = {}
    
    # Default headers
    headers = {
        "User-Agent": "Prometheus/6.5",
        **headers
    }
    
    try:
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = headers.get("Content-Type", "application/json")
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            
            if "application/json" in content_type:
                resp_body = json.loads(response.read().decode("utf-8"))
                return json.dumps({"status": status, "body": resp_body}, indent=2)
            else:
                resp_body = response.read().decode("utf-8")
                return f"Status: {status}\n\n{resp_body[:5000]}"
                
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            return f"⚠️ HTTP {e.code} {e.reason}\n\n{json.dumps(error_body, indent=2)}"
        except:
            return f"⚠️ HTTP {e.code} {e.reason}\n\n{e.read().decode('utf-8', errors='replace')[:1000]}"
    except urllib.error.URLError as e:
        return f"⚠️ Connection error: {e.reason}"
    except TimeoutError:
        return f"⚠️ Request timed out after {timeout} seconds"
    except Exception as e:
        return f"⚠️ Error: {type(e).__name__}: {str(e)}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("http_request", {
            "name": "http_request",
            "description": "Make HTTP requests to external APIs. Supports GET, POST, PUT, DELETE methods. Returns JSON response parsed or raw text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to request"},
                    "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE)", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
                    "headers": {"type": "object", "description": "HTTP headers as key-value pairs", "default": {}},
                    "body": {"type": "object", "description": "JSON body for POST/PUT requests", "default": None},
                    "timeout": {"type": "integer", "description": "Request timeout in seconds", "default": 30}
                },
                "required": ["url"]
            }
        }, _http_request),
    ]
