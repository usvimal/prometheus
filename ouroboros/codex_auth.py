"""
Prometheus — ChatGPT OAuth via Codex CLI flow.

Implements PKCE OAuth against auth.openai.com using the Codex CLI's
public client ID. Routes requests to chatgpt.com/backend-api/codex/responses.

Reference: openai/codex (Rust source), opencode-openai-codex-auth (TS)
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

log = logging.getLogger(__name__)

# --- Constants (from Codex CLI source: codex-rs/core/src/auth.rs) ---
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
AUTH_SCOPES = "openid profile email offline_access"
REFRESH_SCOPES = "openid profile email"
CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
ORIGINATOR = "codex_cli_rs"

TOKEN_REFRESH_MINUTES = 8


# ---- PKCE (from codex-rs/login/src/pkce.rs) ----

def generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier_bytes = secrets.token_bytes(64)
    code_verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


# ---- JWT ----

def decode_jwt_claims(jwt_str: str) -> dict:
    """Decode JWT payload without verification (we only need claims)."""
    parts = jwt_str.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return {}


def extract_account_id(access_token: str) -> Optional[str]:
    """Extract ChatGPT account ID from access token JWT claims."""
    claims = decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth", {})
    return auth_claims.get("chatgpt_account_id")


# ---- Auth URL ----

def build_authorize_url(code_challenge: str, state: str) -> str:
    """Build the OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": AUTH_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ---- Callback Server ----

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code: Optional[str] = None
    received_state: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/auth/callback":
            params = parse_qs(parsed.query)
            _OAuthCallbackHandler.auth_code = params.get("code", [None])[0]
            _OAuthCallbackHandler.received_state = params.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Prometheus authenticated!</h1><p>You can close this tab.</p>")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


# ---- Token Exchange ----

def exchange_code_for_tokens(code: str, code_verifier: str) -> dict:
    """Exchange authorization code for tokens."""
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            content=urlencode({
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "code_verifier": code_verifier,
            }),
        )
        resp.raise_for_status()
        return resp.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Refresh the access token."""
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/json"},
            json={
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": REFRESH_SCOPES,
            },
        )
        resp.raise_for_status()
        return resp.json()


# ---- Auth Storage ----

def _default_auth_path() -> Path:
    """Default auth file location: ~/prometheus/data/auth.json"""
    return Path.home() / "prometheus" / "data" / "auth.json"


def save_auth(tokens: dict, auth_path: Optional[Path] = None,
              account_id: Optional[str] = None) -> None:
    """Save auth tokens to disk."""
    path = auth_path or _default_auth_path()
    auth_data = {
        "auth_mode": "chatgpt",
        "tokens": {
            "id_token": tokens.get("id_token", ""),
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "account_id": account_id or extract_account_id(tokens["access_token"]),
        },
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth_data, indent=2))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows doesn't support Unix permissions


def load_auth(auth_path: Optional[Path] = None) -> Optional[dict]:
    """Load auth tokens from disk."""
    path = auth_path or _default_auth_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---- Login Flows ----

def login_interactive(auth_path: Optional[Path] = None) -> dict:
    """Interactive OAuth login (opens browser, waits for callback)."""
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    auth_url = build_authorize_url(code_challenge, state)

    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.received_state = None

    server = HTTPServer(("127.0.0.1", 1455), _OAuthCallbackHandler)
    log.info("Open this URL to authenticate:\n%s", auth_url)
    print(f"\nOpen this URL to authenticate:\n{auth_url}\n")

    while _OAuthCallbackHandler.auth_code is None:
        server.handle_request()
    server.server_close()

    if _OAuthCallbackHandler.received_state != state:
        raise ValueError("OAuth state mismatch — possible CSRF attack")

    tokens = exchange_code_for_tokens(_OAuthCallbackHandler.auth_code, code_verifier)
    save_auth(tokens, auth_path)
    return tokens


def get_login_url() -> Tuple[str, str, str]:
    """Generate login URL for Telegram-based auth (headless VPS).
    Returns (url, code_verifier, state)."""
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    url = build_authorize_url(code_challenge, state)
    return url, code_verifier, state


# ---- Request Building (Responses API format) ----

def build_headers(access_token: str, account_id: str) -> dict:
    """Build headers for Codex backend API."""
    return {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-ID": account_id,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "originator": ORIGINATOR,
        "User-Agent": "prometheus/0.1.0 (Linux; custom agent)",
    }


def user_message(text: str) -> dict:
    return {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}]}


def assistant_message(text: str) -> dict:
    return {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


def function_call_item(name: str, call_id: str, arguments: str) -> dict:
    return {"type": "function_call", "name": name,
            "call_id": call_id, "arguments": arguments}


def function_call_output_item(call_id: str, output: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def build_codex_request_body(
    model: str,
    instructions: str,
    input_messages: List[dict],
    tools: Optional[List[dict]] = None,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
) -> dict:
    """Build Codex Responses API request body."""
    body: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_messages,
        "tools": tools or [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": {"effort": reasoning_effort, "summary": reasoning_summary},
        "store": False,  # REQUIRED by ChatGPT backend
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    return body


# ---- Chat Completions → Responses API conversion ----

def _convert_messages_to_input(messages: List[dict]) -> Tuple[str, List[dict]]:
    """Convert Chat Completions messages to Responses API input format.

    Returns (instructions, input_items).
    Instructions come from the system message(s).
    """
    instructions_parts = []
    input_items = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                instructions_parts.append(content)
            continue

        if role == "user":
            if isinstance(content, str):
                input_items.append(user_message(content))
            elif isinstance(content, list):
                # Multipart content (text + images)
                input_items.append({
                    "type": "message", "role": "user", "content": content,
                })

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(content, str) and content.strip():
                input_items.append(assistant_message(content))
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    input_items.append(function_call_item(
                        name=fn.get("name", ""),
                        call_id=tc.get("id", ""),
                        arguments=fn.get("arguments", "{}"),
                    ))

        elif role == "tool":
            input_items.append(function_call_output_item(
                call_id=msg.get("tool_call_id", ""),
                output=content if isinstance(content, str) else str(content),
            ))

    return "\n\n".join(instructions_parts), input_items


def _convert_tools_to_responses_format(tools: List[dict]) -> List[dict]:
    """Convert Chat Completions tool schemas to Responses API format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            result.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
    return result


# ---- SSE Response Parsing ----

def parse_sse_to_chat_completion(events: List[dict]) -> Tuple[dict, dict]:
    """Parse Codex SSE events into a Chat Completions-style response message + usage.

    Returns (message_dict, usage_dict) compatible with the existing LLMClient.chat() contract.
    """
    content_parts = []
    tool_calls = []
    _tc_buffers: Dict[str, dict] = {}  # call_id -> {name, arguments}
    usage = {}

    for event in events:
        event_type = event.get("type", "")

        if event_type == "response.output_text.delta":
            content_parts.append(event.get("delta", ""))

        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                call_id = item.get("call_id", "")
                _tc_buffers[call_id] = {
                    "name": item.get("name", ""),
                    "arguments": "",
                }

        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id", "")
            if call_id in _tc_buffers:
                _tc_buffers[call_id]["arguments"] += event.get("delta", "")

        elif event_type == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                call_id = item.get("call_id", "")
                buf = _tc_buffers.get(call_id)
                if buf:
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": buf["name"],
                            "arguments": buf["arguments"],
                        },
                    })

        elif event_type == "response.completed":
            resp = event.get("response", {})
            resp_usage = resp.get("usage", {})
            if resp_usage:
                usage = {
                    "prompt_tokens": resp_usage.get("input_tokens", 0),
                    "completion_tokens": resp_usage.get("output_tokens", 0),
                    "total_tokens": resp_usage.get("total_tokens", 0),
                }

    content = "".join(content_parts) if content_parts else None
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return msg, usage


# ---- Codex LLM Client (sync, matches LLMClient contract) ----

class CodexLLMClient:
    """ChatGPT Codex backend client. Matches the LLMClient.chat() contract."""

    def __init__(self, auth_path: Optional[Path] = None):
        self._auth_path = auth_path or _default_auth_path()
        self._access_token = ""
        self._account_id = ""
        self._refresh_token = ""
        self._last_refresh: float = 0.0
        self._load_tokens()

    def _load_tokens(self) -> None:
        auth = load_auth(self._auth_path)
        if not auth or not auth.get("tokens"):
            log.warning("No Codex auth tokens found at %s", self._auth_path)
            return
        tokens = auth["tokens"]
        self._access_token = tokens.get("access_token", "")
        self._account_id = tokens.get("account_id", "") or extract_account_id(self._access_token) or ""
        self._refresh_token = tokens.get("refresh_token", "")

    @property
    def is_authenticated(self) -> bool:
        return bool(self._access_token)

    def _ensure_fresh_token(self) -> None:
        """Refresh token if older than TOKEN_REFRESH_MINUTES."""
        if time.time() - self._last_refresh < TOKEN_REFRESH_MINUTES * 60:
            return
        if not self._refresh_token:
            return
        try:
            tokens = refresh_access_token(self._refresh_token)
            self._access_token = tokens["access_token"]
            self._refresh_token = tokens.get("refresh_token", self._refresh_token)
            self._account_id = extract_account_id(self._access_token) or self._account_id
            self._last_refresh = time.time()
            save_auth(tokens, self._auth_path, self._account_id)
            log.info("Codex token refreshed")
        except Exception as e:
            log.error("Codex token refresh failed: %s", e)
            raise

    def chat(
        self,
        messages: List[dict],
        model: str,
        tools: Optional[List[dict]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[dict, dict]:
        """Single LLM call via Codex backend. Returns (response_message, usage)."""
        if not self.is_authenticated:
            raise RuntimeError("Codex not authenticated. Run setup first.")

        self._ensure_fresh_token()

        # Convert Chat Completions format to Responses API format
        instructions, input_items = _convert_messages_to_input(messages)
        api_tools = _convert_tools_to_responses_format(tools) if tools else []

        body = build_codex_request_body(
            model=model,
            instructions=instructions,
            input_messages=input_items,
            tools=api_tools or None,
            reasoning_effort=reasoning_effort,
        )

        headers = build_headers(self._access_token, self._account_id)
        url = f"{CODEX_API_BASE}/responses"
        events = []

        with httpx.Client(timeout=300) as client:
            with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code == 401:
                    # Force refresh and retry
                    self._last_refresh = 0
                    self._ensure_fresh_token()
                    headers = build_headers(self._access_token, self._account_id)
                    with client.stream("POST", url, headers=headers, json=body) as retry:
                        retry.raise_for_status()
                        for line in retry.iter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    events.append(json.loads(data))
                                except json.JSONDecodeError:
                                    continue
                    return parse_sse_to_chat_completion(events)

                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            events.append(json.loads(data))
                        except json.JSONDecodeError:
                            continue

        return parse_sse_to_chat_completion(events)
