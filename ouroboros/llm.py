"""
Prometheus — LLM client.

Tri-backend: Codex (ChatGPT OAuth), MiniMax (coding plan), OpenRouter (fallback).
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
    log = logging.getLogger("ouroboros.llm")

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

    def _get_codex_client(self):
        """Lazy-init Codex client. Returns None if not authenticated."""
        if not self._codex_init_attempted:
            self._codex_init_attempted = True
            try:
                from ouroboros.codex_auth import CodexLLMClient
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
        # Route to MiniMax for MiniMax-* models
        if _is_minimax_model(model):
            mm_client = self._get_minimax_client()
            if mm_client:
                try:
                    return self._chat_minimax(mm_client, messages, model, tools,
                                              reasoning_effort, max_tokens, tool_choice)
                except Exception as e:
                    log.warning("MiniMax call failed: %s", e)
                    raise

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
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": {"reasoning_split": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
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
        env_model = os.environ.get("OUROBOROS_MODEL", "")
        if env_model:
            return env_model
        # If Codex is available, default to Codex model
        if self._get_codex_client():
            return DEFAULT_CODEX_MODEL
        return "anthropic/claude-sonnet-4.6"

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = self.default_model()
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        # Add Codex models if authenticated
        if self._get_codex_client():
            for cm in ["codex-mini", "o4-mini", "gpt-4.1"]:
                if cm not in models:
                    models.append(cm)
        # Add MiniMax models if configured
        if self._get_minimax_client():
            for mm in ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"]:
                if mm not in models:
                    models.append(mm)
        if code and code not in models:
            models.append(code)
        if light and light not in models:
            models.append(light)
        return models
