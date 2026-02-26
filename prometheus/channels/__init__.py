"""
Enhanced Telegram channel architecture.

Provides structured message types, middleware pattern, webhook support,
and conversational state management.

Inspired by OpenClaw's channel architecture.
"""

from .message_types import (
    MessageType,
    Message,
    TextMessage,
    CommandMessage,
    PhotoMessage,
    DocumentMessage,
    AudioMessage,
    VideoMessage,
    LocationMessage,
    ContactMessage,
    CallbackMessage,
    Chat,
    User,
)

from .middleware import (
    Middleware,
    MiddlewareContext,
    MiddlewareRegistry,
    LoggingMiddleware,
    RateLimitMiddleware,
    AuthMiddleware,
    CommandRouterMiddleware,
    MetricsMiddleware,
)

from .state import (
    ConversationState,
    ConversationContext,
    StateBackend,
    MemoryStateBackend,
    RedisStateBackend,
    StateManager,
)

from .telegram_enhanced import (
    WebhookConfig,
    TelegramConfig,
    EnhancedTelegramChannel,
    create_channel,
)

__all__ = [
    # Message types
    "MessageType",
    "Message",
    "TextMessage",
    "CommandMessage",
    "PhotoMessage",
    "DocumentMessage",
    "AudioMessage",
    "VideoMessage",
    "LocationMessage",
    "ContactMessage",
    "CallbackMessage",
    "Chat",
    "User",
    # Middleware
    "Middleware",
    "MiddlewareContext",
    "MiddlewareRegistry",
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "AuthMiddleware",
    "CommandRouterMiddleware",
    "MetricsMiddleware",
    # State
    "ConversationState",
    "ConversationContext",
    "StateBackend",
    "MemoryStateBackend",
    "RedisStateBackend",
    "StateManager",
    # Channel
    "WebhookConfig",
    "TelegramConfig",
    "EnhancedTelegramChannel",
    "create_channel",
]
