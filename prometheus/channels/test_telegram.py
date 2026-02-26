"""
Tests for enhanced Telegram channel implementation.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from prometheus.channels.message_types import (
    Message, TextMessage, CommandMessage, PhotoMessage, 
    CallbackMessage, MessageType, User, Chat
)
from prometheus.channels.middleware import (
    MiddlewareRegistry, LoggingMiddleware, RateLimitMiddleware,
    AuthMiddleware, MetricsMiddleware, CommandRouterMiddleware,
    MiddlewareContext
)
from prometheus.channels.state import (
    ConversationContext, ConversationState, StateManager,
    MemoryStateBackend
)
from prometheus.channels.telegram_enhanced import (
    EnhancedTelegramChannel, TelegramConfig, WebhookConfig
)


# =============================================================================
# Message Types Tests
# =============================================================================

class TestUser:
    def test_display_name_username(self):
        user = User(id=123, username="testuser", first_name="Test")
        assert user.display_name == "@testuser"
    
    def test_display_name_full_name(self):
        user = User(id=123, first_name="Test", last_name="User")
        assert user.display_name == "Test User"
    
    def test_display_name_first_name_only(self):
        user = User(id=123, first_name="Test")
        assert user.display_name == "Test"
    
    def test_display_name_fallback(self):
        user = User(id=123)
        assert user.display_name == "user_123"


class TestChat:
    def test_is_private(self):
        chat = Chat(id=123, type="private")
        assert chat.is_private is True
        assert chat.is_group is False
        assert chat.is_channel is False
    
    def test_is_group(self):
        chat = Chat(id=123, type="supergroup")
        assert chat.is_private is False
        assert chat.is_group is True
        assert chat.is_channel is False
    
    def test_is_channel(self):
        chat = Chat(id=123, type="channel")
        assert chat.is_private is False
        assert chat.is_group is False
        assert chat.is_channel is True


class TestMessageFactory:
    def test_parse_text_message(self):
        update = {
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "from": {"id": 456, "first_name": "Test"},
                "date": 1700000000,
                "text": "Hello world"
            }
        }
        msg = Message.from_telegram_update(update)
        assert isinstance(msg, TextMessage)
        assert msg.message_type == MessageType.TEXT
        assert msg.text == "Hello world"
    
    def test_parse_command_message(self):
        update = {
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "from": {"id": 456, "first_name": "Test"},
                "date": 1700000000,
                "text": "/start arg1 arg2"
            }
        }
        msg = Message.from_telegram_update(update)
        assert isinstance(msg, CommandMessage)
        assert msg.message_type == MessageType.COMMAND
        assert msg.command == "start"
        assert msg.args == "arg1 arg2"
    
    def test_parse_command_with_bot_username(self):
        update = {
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "from": {"id": 456, "first_name": "Test"},
                "date": 1700000000,
                "text": "/start@mybot"
            }
        }
        msg = Message.from_telegram_update(update)
        assert isinstance(msg, CommandMessage)
        assert msg.command == "start"
    
    def test_parse_photo_message(self):
        update = {
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "from": {"id": 456, "first_name": "Test"},
                "date": 1700000000,
                "photo": [
                    {"file_id": "small", "width": 100, "height": 100},
                    {"file_id": "large", "width": 800, "height": 600}
                ],
                "caption": "My photo"
            }
        }
        msg = Message.from_telegram_update(update)
        assert isinstance(msg, PhotoMessage)
        assert msg.message_type == MessageType.PHOTO
        assert msg.file_id == "large"  # Gets largest photo
        assert msg.caption == "My photo"
    
    def test_parse_callback_message(self):
        update = {
            "callback_query": {
                "id": "callback_123",
                "from": {"id": 456, "first_name": "Test"},
                "data": "button_pressed",
                "message": {
                    "message_id": 1,
                    "chat": {"id": 123, "type": "private"}
                }
            }
        }
        msg = Message.from_telegram_update(update)
        assert isinstance(msg, CallbackMessage)
        assert msg.message_type == MessageType.CALLBACK
        assert msg.callback_id == "callback_123"
        assert msg.data == "button_pressed"
    
    def test_parse_unknown_update(self):
        update = {"unknown_key": "value"}
        msg = Message.from_telegram_update(update)
        assert msg is None


# =============================================================================
# Middleware Tests
# =============================================================================

class TestMiddlewareRegistry:
    @pytest.mark.asyncio
    async def test_middleware_chain_execution(self):
        registry = MiddlewareRegistry()
        
        calls = []
        
        class TestMiddleware:
            name = "test"
            def __init__(self, label):
                self.label = label
            async def process(self, msg, ctx, next_fn):
                calls.append(f"{self.label}_pre")
                result = await next_fn(msg, ctx)
                calls.append(f"{self.label}_post")
                return result
            async def on_error(self, msg, ctx, error):
                raise error
        
        async def final_handler(msg, ctx):
            calls.append("final")
            return "done"
        
        registry.use(TestMiddleware("A"))
        registry.use(TestMiddleware("B"))
        registry.set_final_handler(final_handler)
        
        msg = MagicMock()
        result = await registry.process(msg)
        
        assert result == "done"
        assert calls == ["A_pre", "B_pre", "final", "B_post", "A_post"]
    
    def test_get_middleware(self):
        registry = MiddlewareRegistry()
        middleware = LoggingMiddleware()
        registry.use(middleware)
        
        found = registry.get_middleware("logging")
        assert found is middleware
        
        not_found = registry.get_middleware("nonexistent")
        assert not_found is None


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        middleware = RateLimitMiddleware(max_requests=2, window_seconds=60)
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="test"
        )
        
        calls = []
        async def next_fn(m, c):
            calls.append("next")
            return "ok"
        
        ctx = MiddlewareContext()
        
        # First two requests should pass
        result1 = await middleware.process(msg, ctx, next_fn)
        result2 = await middleware.process(msg, ctx, next_fn)
        assert result1 == "ok"
        assert result2 == "ok"
        
        # Third request should be rate limited
        result3 = await middleware.process(msg, ctx, next_fn)
        assert result3 == {"error": "rate_limited", "retry_after": 60}
    
    @pytest.mark.asyncio
    async def test_no_user_bypasses_rate_limit(self):
        middleware = RateLimitMiddleware(max_requests=1, window_seconds=60)
        
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=None,
            date=datetime.now(), text="test"
        )
        
        async def next_fn(m, c):
            return "ok"
        
        ctx = MiddlewareContext()
        
        # Should pass without rate limiting (no user to track)
        result = await middleware.process(msg, ctx, next_fn)
        assert result == "ok"


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_authorized_user_by_id(self):
        middleware = AuthMiddleware(allowed_user_ids=[123])
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="test"
        )
        
        async def next_fn(m, c):
            return "ok"
        
        ctx = MiddlewareContext()
        result = await middleware.process(msg, ctx, next_fn)
        assert result == "ok"
        assert ctx.get("authorized") is True
    
    @pytest.mark.asyncio
    async def test_authorized_user_by_username(self):
        middleware = AuthMiddleware(allowed_usernames=["testuser"])
        
        user = User(id=123, username="TestUser")
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="test"
        )
        
        async def next_fn(m, c):
            return "ok"
        
        ctx = MiddlewareContext()
        result = await middleware.process(msg, ctx, next_fn)
        assert result == "ok"
    
    @pytest.mark.asyncio
    async def test_unauthorized_user(self):
        middleware = AuthMiddleware(allowed_user_ids=[999])
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="test"
        )
        
        async def next_fn(m, c):
            return "ok"
        
        ctx = MiddlewareContext()
        result = await middleware.process(msg, ctx, next_fn)
        assert result == {"error": "unauthorized", "reason": "not_in_whitelist"}
    
    @pytest.mark.asyncio
    async def test_no_whitelist_allows_all(self):
        middleware = AuthMiddleware()  # No whitelist
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="test"
        )
        
        async def next_fn(m, c):
            return "ok"
        
        ctx = MiddlewareContext()
        result = await middleware.process(msg, ctx, next_fn)
        assert result == "ok"
        assert ctx.get("authorized") is True


class TestMetricsMiddleware:
    @pytest.mark.asyncio
    async def test_collects_metrics(self):
        middleware = MetricsMiddleware()
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="test"
        )
        
        async def next_fn(m, c):
            return "ok"
        
        ctx = MiddlewareContext()
        
        # Process multiple messages
        await middleware.process(msg, ctx, next_fn)
        await middleware.process(msg, ctx, next_fn)
        
        stats = middleware.get_stats()
        assert stats["total_messages"] == 2
        assert stats["by_type"]["TEXT"] == 2
        assert stats["errors"] == 0
        assert stats["avg_processing_time"] > 0


class TestCommandRouterMiddleware:
    @pytest.mark.asyncio
    async def test_routes_commands(self):
        middleware = CommandRouterMiddleware()
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        cmd_msg = CommandMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="/test", command="test", args=""
        )
        
        handler_calls = []
        async def test_handler(msg, ctx):
            handler_calls.append("test")
            return "handled"
        
        middleware.register("test", test_handler)
        
        async def next_fn(m, c):
            return "not_handled"
        
        ctx = MiddlewareContext()
        result = await middleware.process(cmd_msg, ctx, next_fn)
        
        assert result == "handled"
        assert handler_calls == ["test"]
    
    @pytest.mark.asyncio
    async def test_non_command_passes_through(self):
        middleware = CommandRouterMiddleware()
        
        user = User(id=123)
        chat = Chat(id=456, type="private")
        text_msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Hello"
        )
        
        async def next_fn(m, c):
            return "passed_through"
        
        ctx = MiddlewareContext()
        result = await middleware.process(text_msg, ctx, next_fn)
        
        assert result == "passed_through"


# =============================================================================
# State Management Tests
# =============================================================================

class TestConversationContext:
    def test_conversation_creation(self):
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        assert ctx.conversation_id == "abc123"
        assert ctx.user_id == 123
        assert ctx.state == ConversationState.IDLE
        assert ctx.is_expired() is False
    
    def test_conversation_expiration(self):
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456,
            expires_at=1  # Already expired
        )
        assert ctx.is_expired() is True
    
    def test_add_message(self):
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        old_updated = ctx.updated_at
        
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there!")
        
        assert len(ctx.history) == 2
        assert ctx.history[0]["role"] == "user"
        assert ctx.history[0]["content"] == "Hello"
        assert ctx.history[1]["role"] == "assistant"
        assert ctx.updated_at > old_updated
    
    def test_state_transition(self):
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        
        ctx.transition_to(ConversationState.AWAITING_INPUT)
        assert ctx.state == ConversationState.AWAITING_INPUT
        
        ctx.transition_to(ConversationState.PROCESSING)
        assert ctx.state == ConversationState.PROCESSING
    
    def test_data_storage(self):
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        
        ctx.set_data("key1", "value1")
        ctx.set_data("key2", {"nested": "data"})
        
        assert ctx.get_data("key1") == "value1"
        assert ctx.get_data("key2") == {"nested": "data"}
        assert ctx.get_data("nonexistent", "default") == "default"
    
    def test_serialization(self):
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456,
            state=ConversationState.AWAITING_INPUT
        )
        ctx.add_message("user", "Hello")
        ctx.set_data("test", "value")
        
        data = ctx.to_dict()
        restored = ConversationContext.from_dict(data)
        
        assert restored.conversation_id == ctx.conversation_id
        assert restored.user_id == ctx.user_id
        assert restored.state == ctx.state
        assert restored.history == ctx.history
        assert restored.data == ctx.data


class TestMemoryStateBackend:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        backend = MemoryStateBackend()
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        
        await backend.set("test_key", ctx)
        retrieved = await backend.get("test_key")
        
        assert retrieved is not None
        assert retrieved.conversation_id == "abc123"
    
    @pytest.mark.asyncio
    async def test_delete(self):
        backend = MemoryStateBackend()
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        
        await backend.set("test_key", ctx)
        existed = await backend.delete("test_key")
        not_existed = await backend.delete("nonexistent")
        
        assert existed is True
        assert not_existed is False
        assert await backend.get("test_key") is None
    
    @pytest.mark.asyncio
    async def test_expiration(self):
        backend = MemoryStateBackend(default_ttl=0)  # Immediate expiration
        ctx = ConversationContext(
            conversation_id="abc123",
            user_id=123,
            chat_id=456
        )
        
        await backend.set("test_key", ctx)
        # Should be expired immediately
        retrieved = await backend.get("test_key")
        
        assert retrieved is None


class TestStateManager:
    @pytest.mark.asyncio
    async def test_get_or_create_conversation(self):
        manager = StateManager()
        
        # Create new
        ctx1 = await manager.get_or_create(123, 456)
        assert ctx1.conversation_id is not None
        
        # Get existing
        ctx2 = await manager.get_or_create(123, 456)
        assert ctx2.conversation_id == ctx1.conversation_id
    
    @pytest.mark.asyncio
    async def test_end_conversation(self):
        manager = StateManager()
        
        await manager.create_conversation(123, 456)
        existed = await manager.end_conversation(123, 456)
        not_existed = await manager.end_conversation(123, 456)
        
        assert existed is True
        assert not_existed is False
    
    @pytest.mark.asyncio
    async def test_update_state(self):
        manager = StateManager()
        
        ctx = await manager.create_conversation(123, 456)
        assert ctx.state == ConversationState.IDLE
        
        updated = await manager.update_state(123, 456, ConversationState.PROCESSING)
        assert updated.state == ConversationState.PROCESSING
    
    @pytest.mark.asyncio
    async def test_add_message(self):
        manager = StateManager()
        
        await manager.create_conversation(123, 456)
        await manager.add_message(123, 456, "user", "Hello")
        await manager.add_message(123, 456, "assistant", "Hi!")
        
        ctx = await manager.get_conversation(123, 456)
        assert len(ctx.history) == 2


# =============================================================================
# Telegram Channel Tests
# =============================================================================

class TestTelegramConfig:
    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
        monkeypatch.setenv("TELEGRAM_OWNER_ID", "12345")
        
        config = TelegramConfig.from_env()
        assert config.bot_token == "test_token"
        assert config.owner_id == 12345
    
    def test_webhook_config_url(self):
        config = WebhookConfig(host="example.com", port=8443, path="/bot")
        assert config.url == "https://example.com:8443/bot"


class TestEnhancedTelegramChannel:
    @pytest.mark.asyncio
    async def test_command_registration(self):
        config = TelegramConfig(bot_token="test", owner_id=123)
        channel = EnhancedTelegramChannel(config)
        
        @channel.command("test")
        async def test_handler(msg):
            return "test_result"
        
        assert "test" in channel._commands
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self):
        config = TelegramConfig(bot_token="test", owner_id=123)
        channel = EnhancedTelegramChannel(config)
        
        @channel.on_message
        async def msg_handler(msg):
            return "msg_result"
        
        assert channel._message_handler is not None
    
    @pytest.mark.asyncio
    async def test_middleware_setup(self):
        config = TelegramConfig(bot_token="test", owner_id=123)
        channel = EnhancedTelegramChannel(config)
        
        names = channel.middleware.middleware_names
        assert "logging" in names
        assert "rate_limit" in names
        assert "command_router" in names
        assert "metrics" in names


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_full_message_flow():
    """Test a complete message flow through the system."""
    
    # Setup
    config = TelegramConfig(bot_token="test", owner_id=123)
    channel = EnhancedTelegramChannel(config)
    
    handler_calls = []
    
    @channel.command("hello")
    async def hello_handler(msg):
        handler_calls.append("hello")
        return {"sent": True}
    
    @channel.on_message
    async def default_handler(msg):
        handler_calls.append("default")
        return {"sent": True}
    
    # Create test message
    update = {
        "message": {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 456, "first_name": "Test"},
            "date": 1700000000,
            "text": "/hello world"
        }
    }
    
    # Process
    await channel._process_update(update)
    
    # Verify
    assert "hello" in handler_calls


@pytest.mark.asyncio
async def test_conversation_flow():
    """Test multi-turn conversation with state."""
    
    manager = StateManager()
    
    # Start conversation
    ctx = await manager.get_or_create(user_id=123, chat_id=456)
    ctx.transition_to(ConversationState.AWAITING_INPUT)
    await manager.backend.set("123:456", ctx)
    
    # Add messages
    await manager.add_message(123, 456, "user", "What's the weather?")
    await manager.add_message(123, 456, "assistant", "It's sunny!")
    
    # Complete
    await manager.update_state(123, 456, ConversationState.COMPLETED)
    
    # Verify
    final_ctx = await manager.get_conversation(123, 456)
    assert final_ctx.state == ConversationState.COMPLETED
    assert len(final_ctx.history) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
