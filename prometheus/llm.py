"""
Prometheus — LLM client.

Quad-backend: Kimi Code (primary), MiniMax (fallback), Codex (OAuth), OpenRouter (vision).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"
DEFAULT_CODEX_MODEL = "codex-mini"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("prometheus.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


def _is_codex_model(model: str) -> bool:
    """Return True if model should be routed to Codex backend (no provider prefix, not MiniMax)."""
    if "/" in model:
        return False
    if model.lower().startswith("minimax-"):
        return False
    return True


def _is_minimax_model(model: str) -> bool:
    """Return True if model should be routed to MiniMax backend."""
    return model.lower().startswith("minimax-")


def _is_kimi_model(model: str) -> bool:
    """Return True if model should be routed to Kimi Code backend."""
    m = model.lower()
    return m == "kimi-for-coding" or m.startswith("kimi-")


# ---------------------------------------------------------------------------
# MiniMax coding plan quota
# ---------------------------------------------------------------------------
_minimax_quota_cache: Dict[str, Any] = {}
_minimax_quota_ts: float = 0.0
_MINIMAX_QUOTA_TTL: float = 60.0  # Cache for 60 seconds


def fetch_minimax_quota(force: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch remaining quota from MiniMax coding plan API.

    Returns dict like:
      {"MiniMax-M2.5": {"total": 4500, "used": 123, "remaining": 4377, "window_remaining_sec": 14275}}
    or None on error.  Cached for 60s.
    """
    global _minimax_quota_cache, _minimax_quota_ts
    now = time.time()
    if not force and _minimax_quota_cache and (now - _minimax_quota_ts) < _MINIMAX_QUOTA_TTL:
        return _minimax_quota_cache

    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import urllib.request
        import json as _json
        req = urllib.request.Request(
            "https://api.minimax.io/v1/api/openplatform/coding_plan/remains",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        if data.get("base_resp", {}).get("status_code") != 0:
            log.warning("MiniMax quota API error: %s", data.get("base_resp"))
            return None
        result = {}
        for m in data.get("model_remains", []):
            name = m.get("model_name", "")
            result[name] = {
                "total": m.get("current_interval_total_count", 0),
                "used": m.get("current_interval_usage_count", 0),
                "remaining": m.get("current_interval_total_count", 0) - m.get("current_interval_usage_count", 0),
                "window_remaining_sec": int(m.get("remains_time", 0)) // 1000,
            }
        _minimax_quota_cache = result
        _minimax_quota_ts = now
        return result
    except Exception as e:
        log.debug("Failed to fetch MiniMax quota: %s", e)
        return None


# ---------------------------------------------------------------------------
# Kimi Code 5-hour window usage tracking
# ---------------------------------------------------------------------------
_KIMI_WINDOW_SEC: float = 5 * 3600  # 5-hour rolling window
_kimi_window: Dict[str, Any] = {
    "start": 0.0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "calls": 0,
}


def _track_kimi_usage(usage: Dict[str, Any]) -> None:
    """Track Kimi token usage in current 5-hour window."""
    now = time.time()
    if now - _kimi_window["start"] > _KIMI_WINDOW_SEC:
        # Window expired, reset
        _kimi_window["start"] = now
        _kimi_window["input_tokens"] = 0
        _kimi_window["output_tokens"] = 0
        _kimi_window["cache_read_tokens"] = 0
        _kimi_window["cache_write_tokens"] = 0
        _kimi_window["calls"] = 0
    _kimi_window["input_tokens"] += int(usage.get("prompt_tokens") or 0)
    _kimi_window["output_tokens"] += int(usage.get("completion_tokens") or 0)
    _kimi_window["cache_read_tokens"] += int(usage.get("cached_tokens") or 0)
    _kimi_window["cache_write_tokens"] += int(usage.get("cache_write_tokens") or 0)
    _kimi_window["calls"] += 1


def get_kimi_usage() -> Dict[str, Any]:
    """Return current Kimi 5-hour window usage for /status display."""
    now = time.time()
    if now - _kimi_window["start"] > _KIMI_WINDOW_SEC:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "window_remaining_sec": int(_KIMI_WINDOW_SEC),
                "window_elapsed_sec": 0}
    elapsed = int(now - _kimi_window["start"])
    remaining = int(_KIMI_WINDOW_SEC - elapsed)
    return {
        "calls": _kimi_window["calls"],
        "input_tokens": _kimi_window["input_tokens"],
        "output_tokens": _kimi_window["output_tokens"],
        "cache_read_tokens": _kimi_window["cache_read_tokens"],
        "cache_write_tokens": _kimi_window["cache_write_tokens"],
        "window_remaining_sec": max(0, remaining),
        "window_elapsed_sec": elapsed,
    }


class LLMClient:
    """Tri-backend LLM client: Codex + MiniMax + OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None
        self._codex_client = None
        self._codex_init_attempted = False
        self._minimax_client = None
        self._minimax_init_attempted = False
        self._kimi_client = None
        self._kimi_init_attempted = False

    def _get_codex_client(self):
        """Lazy-init Codex client. Returns None if not authenticated."""
        if not self._codex_init_attempted:
            self._codex_init_attempted = True
            try:
                from prometheus.codex_auth import CodexLLMClient
                client = CodexLLMClient()
                if client.is_authenticated:
                    self._codex_client = client
                    log.info("Codex backend available")
                else:
                    log.info("Codex auth not found, using OpenRouter only")
            except Exception as e:
                log.warning("Codex client init failed: %s", e)
        return self._codex_client

    def _get_minimax_client(self):
        """Lazy-init MiniMax client. Returns None if no API key configured."""
        if not self._minimax_init_attempted:
            self._minimax_init_attempted = True
            minimax_key = os.environ.get("MINIMAX_API_KEY", "").strip()
            if minimax_key:
                try:
                    from openai import OpenAI
                    self._minimax_client = OpenAI(
                        base_url="https://api.minimax.io/v1",
                        api_key=minimax_key,
                    )
                    log.info("MiniMax backend available")
                except Exception as e:
                    log.warning("MiniMax client init failed: %s", e)
            else:
                log.info("MiniMax API key not found, skipping")
        return self._minimax_client


    def _get_kimi_client(self):
        """Lazy-init Kimi Code client. Returns API key string or None.

        Kimi Code's OpenAI endpoint is locked to whitelisted agents.
        We use the Anthropic-compatible endpoint (coding/v1/messages) via httpx.
        """
        if not self._kimi_init_attempted:
            self._kimi_init_attempted = True
            kimi_key = os.environ.get("KIMI_API_KEY", "").strip()
            if kimi_key:
                self._kimi_client = kimi_key  # Store key, not client object
                log.info("Kimi Code backend available")
            else:
                log.info("Kimi API key not found, skipping")
        return self._kimi_client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/usvimal/prometheus",
                    "X-Title": "Prometheus",
                },
            )
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Routes to Codex, MiniMax, or OpenRouter based on model name."""
        # Route to Kimi Code for kimi-* models (primary)
        if _is_kimi_model(model):
            kimi_key = self._get_kimi_client()
            if kimi_key:
                try:
                    return self._chat_kimi(kimi_key, messages, model, tools,
                                           reasoning_effort, max_tokens, tool_choice)
                except Exception as e:
                    log.warning("Kimi Code call failed, falling back to MiniMax: %s", e)
                    # Fallback to MiniMax
                    mm = self._get_minimax_client()
                    if mm:
                        return self._chat_minimax(mm, messages, "MiniMax-M2.5", tools,
                                                  reasoning_effort, max_tokens, tool_choice)
                    raise

        # Route to MiniMax for MiniMax-* models
        if _is_minimax_model(model):
            mm_client = self._get_minimax_client()
            if mm_client:
                try:
                    return self._chat_minimax(mm_client, messages, model, tools,
                                              reasoning_effort, max_tokens, tool_choice)
                except Exception as e:
                    log.warning("MiniMax call failed: %s", e)
                    raise  # No fallback — MiniMax is the only LLM

        # Route to Codex for non-prefixed models
        if _is_codex_model(model):
            codex = self._get_codex_client()
            if codex:
                try:
                    return codex.chat(
                        messages=messages, model=model, tools=tools,
                        reasoning_effort=reasoning_effort,
                        max_tokens=max_tokens, tool_choice=tool_choice,
                    )
                except Exception as e:
                    log.warning("Codex call failed, falling back to OpenRouter: %s", e)
                    model = f"openai/{model}"

        # OpenRouter path
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }

        # Pin Anthropic models to Anthropic provider for prompt caching
        if model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if tools:
            # Add cache_control to last tool for Anthropic prompt caching
            # This caches all tool schemas (they never change between calls)
            tools_with_cache = [t for t in tools]  # shallow copy
            if tools_with_cache:
                last_tool = {**tools_with_cache[-1]}  # copy last tool
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def _chat_kimi(
        self,
        api_key,
        messages,
        model,
        tools,
        reasoning_effort,
        max_tokens,
        tool_choice,
    ):
        """Kimi Code chat via Anthropic-compatible endpoint (httpx).

        The OpenAI endpoint is locked to whitelisted agents (Claude Code, Roo Code, etc).
        The Anthropic messages endpoint works for any client.
        """
        import httpx

        # Convert OpenAI message format to Anthropic format
        # Extract system messages into a separate 'system' field
        system_parts = []
        anthropic_msgs = []
        for msg in messages:
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                text = self._flatten_content(raw_content)
                if text.strip():
                    system_parts.append(text.strip())
                continue
            # Convert tool_calls from OpenAI format to Anthropic format
            if role == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                text = self._flatten_content(raw_content)
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in msg["tool_calls"]:
                    import json as _json
                    args = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except _json.JSONDecodeError:
                            args = {"raw": args}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": args,
                    })
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})
                continue
            # Convert tool results from OpenAI format to Anthropic format
            if role == "tool":
                anthropic_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": self._flatten_content(raw_content),
                    }],
                })
                continue
            # Regular user/assistant messages
            text = self._flatten_content(raw_content)
            anthropic_msgs.append({"role": role, "content": text})

        # Merge consecutive same-role messages (Anthropic requires alternating roles)
        merged = []
        for msg in anthropic_msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                # Merge content
                prev = merged[-1]["content"]
                curr = msg["content"]
                if isinstance(prev, str) and isinstance(curr, str):
                    merged[-1]["content"] = prev + chr(10) + curr
                elif isinstance(prev, list) and isinstance(curr, list):
                    merged[-1]["content"] = prev + curr
                elif isinstance(prev, str) and isinstance(curr, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev}] + curr
                elif isinstance(prev, list) and isinstance(curr, str):
                    merged[-1]["content"] = prev + [{"type": "text", "text": curr}]
            else:
                merged.append(msg)
        anthropic_msgs = merged

        # Build Anthropic API request body
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_msgs,
        }
        if system_parts:
            body["system"] = chr(10).join(system_parts)
        if tools:
            # Convert OpenAI tool schema to Anthropic format
            anthropic_tools = []
            for t in tools:
                func = t.get("function", t)
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            body["tools"] = anthropic_tools
            if tool_choice and tool_choice != "auto":
                body["tool_choice"] = {"type": tool_choice}

        resp = httpx.post(
            "https://api.kimi.com/coding/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Convert Anthropic response to OpenAI format (what our loop expects)
        msg_out = {"role": "assistant", "content": None, "tool_calls": None}
        text_parts = []
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                import json as _json
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _json.dumps(block.get("input", {})),
                    },
                })
        if text_parts:
            msg_out["content"] = chr(10).join(text_parts)
        if tool_calls:
            msg_out["tool_calls"] = tool_calls

        # Build usage dict (Anthropic format -> OpenAI format)
        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("input_tokens", 0),
            "completion_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            "cached_tokens": usage_data.get("cache_read_input_tokens", 0),
            "cache_write_tokens": usage_data.get("cache_creation_input_tokens", 0),
        }

        _track_kimi_usage(usage)
        return msg_out, usage

    @staticmethod
    def _flatten_content(content: Any) -> str:
        """Flatten multipart content (list of dicts) to plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return chr(10).join(p for p in parts if p)
        return str(content or "")

    def _prep_minimax_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to MiniMax-compatible format.

        MiniMax rejects role: system ANYWHERE in the message list.
        Strip ALL system messages (not just leading ones) and prepend
        their content to the first user message.
        """
        result: List[Dict[str, Any]] = []
        system_texts: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                # Collect ALL system content from ANY position
                text = self._flatten_content(msg.get("content", ""))
                if text.strip():
                    system_texts.append(text.strip())
                continue

            # Flatten multipart content for non-system messages too
            new_msg = dict(msg)
            if isinstance(new_msg.get("content"), list):
                new_msg["content"] = self._flatten_content(new_msg["content"])
            result.append(new_msg)

        # Prepend all collected system content to first user message
        combined_system = chr(10).join(system_texts)
        if combined_system:
            for i, msg in enumerate(result):
                if msg.get("role") == "user":
                    user_content = self._flatten_content(msg.get("content", ""))
                    result[i] = {
                        **msg,
                        "content": combined_system + chr(10)+chr(10)+"---"+chr(10)+chr(10) + user_content,
                    }
                    break
            else:
                # No user message found — insert as user message at position 0
                result.insert(0, {"role": "user", "content": combined_system})

        return result

    def _chat_minimax(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """MiniMax chat call via OpenAI-compatible API."""
        # Convert system messages — MiniMax rejects role: system
        mm_messages = self._prep_minimax_messages(messages)

        # MiniMax M2.5 recommended parameters (trained at these values)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": mm_messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "extra_body": {
                "reasoning_split": True,
            },
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details (MiniMax auto-caching)
        # Without this, cached_tokens is always 0 even when MiniMax IS caching.
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])
        # Also extract cache_write_tokens if present
        if not usage.get("cache_write_tokens"):
            prompt_details_w = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_w, dict):
                cw = (prompt_details_w.get("cache_write_tokens")
                      or prompt_details_w.get("cache_creation_tokens"))
                if cw:
                    usage["cache_write_tokens"] = int(cw)

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        env_model = os.environ.get("PROMETHEUS_MODEL", "")
        if env_model:
            return env_model
        # Priority: Kimi > Codex > OpenRouter fallback
        if self._get_kimi_client():
            return "kimi-for-coding"
        if self._get_codex_client():
            return DEFAULT_CODEX_MODEL
        return "anthropic/claude-sonnet-4.6"

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = self.default_model()
        code = os.environ.get("PROMETHEUS_MODEL_CODE", "")
        light = os.environ.get("PROMETHEUS_MODEL_LIGHT", "")
        models = [main]
        # Add Codex models if authenticated
        if self._get_codex_client():
            for cm in ["codex-mini", "o4-mini", "gpt-4.1"]:
                if cm not in models:
                    models.append(cm)
        # Add Kimi models if configured
        if self._get_kimi_client():
            if "kimi-for-coding" not in models:
                models.append("kimi-for-coding")
        # Add MiniMax models if configured (fallback)
        if self._get_minimax_client():
            for mm in ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"]:
                if mm not in models:
                    models.append(mm)
        if code and code not in models:
            models.append(code)
        if light and light not in models:
            models.append(light)
        return models
