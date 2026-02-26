"""
Middleware pattern for Telegram channel.

Allows processing messages through a chain of handlers,
enabling cross-cutting concerns like logging, rate limiting,
authentication, etc.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass, field

from .message_types import Message, CommandMessage, User

log = logging.getLogger(__name__)

T = TypeVar('T')
Context = Dict[str, Any]
NextFunction = Callable[[Message, Context], Any]


@dataclass
class MiddlewareContext:
    """Context passed through middleware chain."""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class Middleware(ABC):
    """Base middleware class.
    
    Middleware can:
    - Process/modify messages before they reach handlers
    - Short-circuit the chain (e.g., for rate limiting)
    - Post-process after handlers complete
    - Track timing and metrics
    """
    
    name: str = "base"
    
    @abstractmethod
    async def process(self, message: Message, context: MiddlewareContext, next_fn: NextFunction) -> Any:
        """Process message and call next middleware/handler.
        
        Args:
            message: The incoming message
            context: Shared context across middleware
            next_fn: Call this to continue to next middleware/handler
        
        Returns:
            Result from handler or middleware processing
        """
        pass
    
    async def on_error(self, message: Message, context: MiddlewareContext, error: Exception) -> Any:
        """Handle errors from downstream middleware/handlers.
        
        Override to provide custom error handling.
        """
        raise error


class LoggingMiddleware(Middleware):
    """Logs all incoming messages and processing time."""
    
    name = "logging"
    
    async def process(self, message: Message, context: MiddlewareContext, next_fn: NextFunction) -> Any:
        start_time = time.time()
        user = message.from_user.display_name if message.from_user else "unknown"
        
        log.info("→ [%s] %s from %s in chat %s", 
                 message.message_type.name, 
                 getattr(message, 'text', '')[ :50],
                 user,
                 message.chat.id)
        
        try:
            result = await next_fn(message, context)
            elapsed = time.time() - start_time
            log.info("✓ [%s] processed in %.3fs", message.message_type.name, elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            log.error("✗ [%s] failed after %.3fs: %s", message.message_type.name, elapsed, e)
            raise


class RateLimitMiddleware(Middleware):
    """Rate limiting per user."""
    
    name = "rate_limit"
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[int, List[float]] = {}  # user_id -> timestamps
    
    def _is_rate_limited(self, user_id: int) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Get recent requests for user
        user_requests = self._requests.get(user_id, [])
        recent = [ts for ts in user_requests if ts > window_start]
        self._requests[user_id] = recent
        
        return len(recent) >= self.max_requests
    
    async def process(self, message: Message, context: MiddlewareContext, next_fn: NextFunction) -> Any:
        if not message.from_user:
            return await next_fn(message, context)
        
        if self._is_rate_limited(message.from_user.id):
            log.warning("Rate limited user %s", message.from_user.id)
            context.set("rate_limited", True)
            return {"error": "rate_limited", "retry_after": self.window_seconds}
        
        # Record this request
        self._requests.setdefault(message.from_user.id, []).append(time.time())
        return await next_fn(message, context)


class AuthMiddleware(Middleware):
    """Authentication middleware - restrict to specific users."""
    
    name = "auth"
    
    def __init__(self, allowed_user_ids: Optional[List[int]] = None, 
                 allowed_usernames: Optional[List[str]] = None):
        self.allowed_user_ids = set(allowed_user_ids or [])
        self.allowed_usernames = set(u.lower() for u in (allowed_usernames or []))
    
    async def process(self, message: Message, context: MiddlewareContext, next_fn: NextFunction) -> Any:
        if not message.from_user:
            return {"error": "unauthorized", "reason": "no_user"}
        
        user_id = message.from_user.id
        username = (message.from_user.username or "").lower()
        
        # Allow if in whitelist
        if self.allowed_user_ids and user_id in self.allowed_user_ids:
            context.set("authorized", True)
            return await next_fn(message, context)
        
        if self.allowed_usernames and username in self.allowed_usernames:
            context.set("authorized", True)
            return await next_fn(message, context)
        
        # No whitelist = allow all
        if not self.allowed_user_ids and not self.allowed_usernames:
            context.set("authorized", True)
            return await next_fn(message, context)
        
        log.warning("Unauthorized access attempt from user %s (@%s)", user_id, username)
        return {"error": "unauthorized", "reason": "not_in_whitelist"}


class MetricsMiddleware(Middleware):
    """Collect metrics on message processing."""
    
    name = "metrics"
    
    def __init__(self):
        self._stats = {
            "total_messages": 0,
            "by_type": {},
            "errors": 0,
            "avg_processing_time": 0.0,
        }
    
    async def process(self, message: Message, context: MiddlewareContext, next_fn: NextFunction) -> Any:
        start = time.time()
        self._stats["total_messages"] += 1
        
        msg_type = message.message_type.name
        self._stats["by_type"][msg_type] = self._stats["by_type"].get(msg_type, 0) + 1
        
        try:
            result = await next_fn(message, context)
            elapsed = time.time() - start
            
            # Update rolling average
            old_avg = self._stats["avg_processing_time"]
            n = self._stats["total_messages"]
            self._stats["avg_processing_time"] = (old_avg * (n - 1) + elapsed) / n
            
            context.set("processing_time", elapsed)
            return result
        except Exception:
            self._stats["errors"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()


class CommandRouterMiddleware(Middleware):
    """Routes commands to specific handlers."""
    
    name = "command_router"
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._default_handler: Optional[Callable] = None
    
    def register(self, command: str, handler: Callable) -> None:
        """Register a handler for a specific command."""
        self._handlers[command.lower()] = handler
    
    def register_default(self, handler: Callable) -> None:
        """Register default handler for unmatched commands."""
        self._default_handler = handler
    
    async def process(self, message: Message, context: MiddlewareContext, next_fn: NextFunction) -> Any:
        if not isinstance(message, CommandMessage):
            return await next_fn(message, context)
        
        handler = self._handlers.get(message.command)
        if handler:
            context.set("command_handler", handler.__name__)
            return await handler(message, context)
        
        if self._default_handler:
            return await self._default_handler(message, context)
        
        return await next_fn(message, context)


class MiddlewareRegistry:
    """Registry for managing and executing middleware chain."""
    
    def __init__(self):
        self._middlewares: List[Middleware] = []
        self._final_handler: Optional[Callable] = None
    
    def use(self, middleware: Middleware) -> MiddlewareRegistry:
        """Add middleware to the chain."""
        self._middlewares.append(middleware)
        return self
    
    def set_final_handler(self, handler: Callable) -> None:
        """Set the final handler that processes messages after all middleware."""
        self._final_handler = handler
    
    async def process(self, message: Message) -> Any:
        """Process a message through the middleware chain."""
        context = MiddlewareContext()
        
        # Build the chain
        chain = self._build_chain(0)
        return await chain(message, context)
    
    def _build_chain(self, index: int) -> NextFunction:
        """Build the middleware chain recursively."""
        if index >= len(self._middlewares):
            # Final handler
            async def final_handler(message: Message, context: MiddlewareContext) -> Any:
                if self._final_handler:
                    return await self._final_handler(message, context)
                return None
            return final_handler
        
        middleware = self._middlewares[index]
        next_in_chain = self._build_chain(index + 1)
        
        async def handler(message: Message, context: MiddlewareContext) -> Any:
            try:
                return await middleware.process(message, context, next_in_chain)
            except Exception as e:
                return await middleware.on_error(message, context, e)
        
        return handler
    
    def get_middleware(self, name: str) -> Optional[Middleware]:
        """Get middleware by name."""
        for m in self._middlewares:
            if m.name == name:
                return m
        return None
    
    def remove(self, name: str) -> bool:
        """Remove middleware by name."""
        for i, m in enumerate(self._middlewares):
            if m.name == name:
                self._middlewares.pop(i)
                return True
        return False
    
    @property
    def middleware_names(self) -> List[str]:
        return [m.name for m in self._middlewares]
