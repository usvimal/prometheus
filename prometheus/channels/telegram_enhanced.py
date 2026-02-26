"""
Enhanced Telegram channel with structured messages, middleware, webhook support,
and conversational state management.

This can eventually replace supervisor/telegram.py
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

from .message_types import Message, TextMessage, CommandMessage, MessageType
from .middleware import (
    MiddlewareRegistry, LoggingMiddleware, RateLimitMiddleware,
    AuthMiddleware, CommandRouterMiddleware, MetricsMiddleware
)
from .state import StateManager, ConversationContext, ConversationState

log = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Configuration for webhook mode."""
    host: str = "0.0.0.0"
    port: int = 8080
    path: str = "/webhook"
    secret_token: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    
    @property
    def url(self) -> str:
        """Full webhook URL."""
        return f"https://{self.host}:{self.port}{self.path}"


@dataclass
class TelegramConfig:
    """Configuration for Telegram channel."""
    bot_token: str
    owner_id: int
    allowed_users: Optional[List[int]] = None
    rate_limit: int = 30  # requests per minute
    webhook: Optional[WebhookConfig] = None
    use_webhook: bool = False
    
    @classmethod
    def from_env(cls) -> TelegramConfig:
        """Create config from environment variables."""
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            owner_id=int(os.getenv("TELEGRAM_OWNER_ID", "0")),
            allowed_users=None,  # Parse from env if needed
            webhook=WebhookConfig() if os.getenv("WEBHOOK_HOST") else None,
            use_webhook=bool(os.getenv("USE_WEBHOOK", "")),
        )


class EnhancedTelegramChannel:
    """
    Enhanced Telegram channel with:
    - Structured message types (replaces raw dicts)
    - Middleware pattern for extensible processing
    - Webhook mode for production deployments
    - Conversational state management
    """
    
    API_BASE = "https://api.telegram.org/bot"
    
    def __init__(self, config: Optional[TelegramConfig] = None):
        self.config = config or TelegramConfig.from_env()
        self.session: Optional[aiohttp.ClientSession] = None
        self._offset: int = 0
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._webhook_app: Optional[web.Application] = None
        self._webhook_runner: Optional[web.AppRunner] = None
        
        # Middleware registry
        self.middleware = MiddlewareRegistry()
        self._setup_default_middleware()
        
        # State management
        self.state_manager = StateManager()
        
        # Command handlers
        self._commands: Dict[str, Callable] = {}
        self._message_handler: Optional[Callable] = None
        self._callback_handler: Optional[Callable] = None
    
    def _setup_default_middleware(self) -> None:
        """Set up default middleware stack."""
        # Logging first to capture everything
        self.middleware.use(LoggingMiddleware())
        
        # Rate limiting
        self.middleware.use(RateLimitMiddleware(
            max_requests=self.config.rate_limit,
            window_seconds=60
        ))
        
        # Auth check
        if self.config.allowed_users:
            self.middleware.use(AuthMiddleware(
                allowed_user_ids=self.config.allowed_users
            ))
        
        # Command routing
        self.middleware.use(CommandRouterMiddleware())
        
        # Metrics
        self.middleware.use(MetricsMiddleware())
        
        # Set final handler
        self.middleware.set_final_handler(self._handle_message)
    
    async def _handle_message(self, message: Message, context: Any) -> Any:
        """Final handler after all middleware."""
        # Handle commands
        if isinstance(message, CommandMessage):
            handler = self._commands.get(message.command)
            if handler:
                return await handler(message)
            
            # Default command handler
            if message.command == "start":
                return await self.send_message(
                    message.chat.id,
                    "👋 Hello! I'm Prometheus, your self-improving AI agent.\n\n"
                    "Use /help to see available commands."
                )
            elif message.command == "help":
                commands = "\n".join([f"/{cmd}" for cmd in self._commands.keys()])
                return await self.send_message(
                    message.chat.id,
                    f"📋 Available commands:\n{commands}"
                )
        
        # Handle callbacks
        if message.message_type == MessageType.CALLBACK:
            if self._callback_handler:
                return await self._callback_handler(message)
        
        # Handle regular messages
        if self._message_handler:
            return await self._message_handler(message)
        
        return None
    
    # ============ Public API ============
    
    def command(self, name: str):
        """Decorator to register a command handler."""
        def decorator(func: Callable):
            self._commands[name.lower()] = func
            # Also register with command router middleware
            router = self.middleware.get_middleware("command_router")
            if router:
                router.register(name.lower(), func)
            return func
        return decorator
    
    def on_message(self, func: Callable):
        """Decorator to register message handler."""
        self._message_handler = func
        return func
    
    def on_callback(self, func: Callable):
        """Decorator to register callback handler."""
        self._callback_handler = func
        return func
    
    def use(self, middleware: Any) -> EnhancedTelegramChannel:
        """Add custom middleware."""
        self.middleware.use(middleware)
        return self
    
    # ============ Telegram API Methods ============
    
    async def _api_request(self, method: str, **params) -> Dict[str, Any]:
        """Make request to Telegram Bot API."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.API_BASE}{self.config.bot_token}/{method}"
        
        async with self.session.post(url, json=params) as resp:
            data = await resp.json()
            if not data.get("ok"):
                log.error("Telegram API error: %s", data.get("description"))
                raise Exception(f"Telegram API error: {data.get('description')}")
            return data.get("result", {})
    
    async def send_message(self, chat_id: int, text: str,
                          parse_mode: str = "Markdown",
                          reply_markup: Optional[Dict] = None,
                          reply_to_message_id: Optional[int] = None) -> Dict[str, Any]:
        """Send text message."""
        params = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup:
            params["reply_markup"] = reply_markup
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        
        return await self._api_request("sendMessage", **params)
    
    async def send_photo(self, chat_id: int, photo: str,
                        caption: Optional[str] = None,
                        parse_mode: str = "Markdown") -> Dict[str, Any]:
        """Send photo."""
        params = {
            "chat_id": chat_id,
            "photo": photo,
            "parse_mode": parse_mode,
        }
        if caption:
            params["caption"] = caption
        
        return await self._api_request("sendPhoto", **params)
    
    async def edit_message_text(self, chat_id: int, message_id: int,
                               text: str, parse_mode: str = "Markdown") -> Dict[str, Any]:
        """Edit message text."""
        return await self._api_request(
            "editMessageText",
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=parse_mode
        )
    
    async def answer_callback(self, callback_id: str, text: Optional[str] = None,
                             show_alert: bool = False) -> Dict[str, Any]:
        """Answer callback query."""
        params = {"callback_query_id": callback_id}
        if text:
            params["text"] = text
        if show_alert:
            params["show_alert"] = show_alert
        
        return await self._api_request("answerCallbackQuery", **params)
    
    async def set_webhook(self, url: str, secret_token: Optional[str] = None) -> Dict[str, Any]:
        """Set webhook URL."""
        params = {
            "url": url,
            "max_connections": 40,
            "allowed_updates": ["message", "callback_query", "edited_message"],
        }
        if secret_token:
            params["secret_token"] = secret_token
        
        return await self._api_request("setWebhook", **params)
    
    async def delete_webhook(self) -> Dict[str, Any]:
        """Delete webhook and switch back to polling."""
        return await self._api_request("deleteWebhook", drop_pending_updates=True)
    
    async def get_me(self) -> Dict[str, Any]:
        """Get bot info."""
        return await self._api_request("getMe")
    
    # ============ Polling Mode ============
    
    async def _poll_updates(self) -> None:
        """Long-polling for updates."""
        while self._running:
            try:
                updates = await self._api_request(
                    "getUpdates",
                    offset=self._offset,
                    limit=100,
                    timeout=30
                )
                
                for update in updates:
                    self._offset = max(self._offset, update["update_id"] + 1)
                    await self._process_update(update)
                
            except Exception as e:
                log.error("Polling error: %s", e)
                await asyncio.sleep(5)
    
    async def _process_update(self, update: Dict[str, Any]) -> None:
        """Process a single update."""
        try:
            message = Message.from_telegram_update(update)
            if message:
                await self.middleware.process(message)
        except Exception as e:
            log.exception("Error processing update: %s", e)
    
    # ============ Webhook Mode ============
    
    async def _webhook_handler(self, request: web.Request) -> web.Response:
        """Handle incoming webhook requests."""
        # Verify secret token if configured
        if self.config.webhook and self.config.webhook.secret_token:
            token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if token != self.config.webhook.secret_token:
                log.warning("Invalid webhook secret token")
                return web.Response(status=401)
        
        try:
            update = await request.json()
            asyncio.create_task(self._process_update(update))
            return web.Response(status=200)
        except Exception as e:
            log.error("Webhook processing error: %s", e)
            return web.Response(status=500)
    
    async def _setup_webhook_server(self) -> None:
        """Set up webhook HTTP server."""
        if not self.config.webhook:
            raise ValueError("Webhook config not provided")
        
        self._webhook_app = web.Application()
        self._webhook_app.router.add_post(
            self.config.webhook.path,
            self._webhook_handler
        )
        
        self._webhook_runner = web.AppRunner(self._webhook_app)
        await self._webhook_runner.setup()
        
        site = web.TCPSite(
            self._webhook_runner,
            self.config.webhook.host,
            self.config.webhook.port,
            ssl_context=None  # Add SSL if needed
        )
        await site.start()
        
        log.info("Webhook server started on %s:%s", 
                 self.config.webhook.host, self.config.webhook.port)
    
    # ============ Lifecycle ============
    
    async def start(self) -> None:
        """Start the channel (polling or webhook)."""
        self._running = True
        self.session = aiohttp.ClientSession()
        
        # Get bot info
        me = await self.get_me()
        log.info("Starting Telegram channel as @%s", me.get("username"))
        
        if self.config.use_webhook and self.config.webhook:
            # Webhook mode
            await self._setup_webhook_server()
            await self.set_webhook(
                self.config.webhook.url,
                self.config.webhook.secret_token
            )
            log.info("Webhook set to %s", self.config.webhook.url)
        else:
            # Polling mode
            await self.delete_webhook()
            self._poll_task = asyncio.create_task(self._poll_updates())
            log.info("Started polling for updates")
    
    async def stop(self) -> None:
        """Stop the channel."""
        self._running = False
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        if self._webhook_runner:
            await self._webhook_runner.cleanup()
        
        if self.session:
            await self.session.close()
            self.session = None
        
        log.info("Telegram channel stopped")
    
    # ============ Conversation Helpers ============
    
    async def get_conversation(self, user_id: int, chat_id: int) -> Optional[ConversationContext]:
        """Get or create conversation for user."""
        return await self.state_manager.get_or_create(user_id, chat_id)
    
    async def reply_in_conversation(self, message: Message, text: str) -> Dict[str, Any]:
        """Reply within a conversation context."""
        # Add to conversation history
        if message.from_user:
            await self.state_manager.add_message(
                message.from_user.id,
                message.chat.id,
                "assistant",
                text
            )
        
        return await self.send_message(
            message.chat.id,
            text,
            reply_to_message_id=message.message_id
        )


# Convenience function for simple usage
def create_channel(token: Optional[str] = None,
                  owner_id: Optional[int] = None) -> EnhancedTelegramChannel:
    """Create a configured Telegram channel."""
    config = TelegramConfig(
        bot_token=token or os.getenv("TELEGRAM_BOT_TOKEN", ""),
        owner_id=owner_id or int(os.getenv("TELEGRAM_OWNER_ID", "0")),
    )
    return EnhancedTelegramChannel(config)
