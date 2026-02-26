"""
Tests for Telegram group handler.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from prometheus.channels.message_types import (
    TextMessage, CommandMessage, User, Chat, MessageType
)
from prometheus.channels.group_handler import (
    GroupContext, GroupMessageHandler, GroupCommandRouter, GroupMiddleware
)


class TestGroupContext:
    def test_should_respond_private_chat(self):
        chat = Chat(id=123, type="private")
        bot = User(id=999, username="mybot")
        ctx = GroupContext(chat=chat, bot_user=bot, is_mentioned=False)
        assert ctx.should_respond is True  # Private chat always responds
    
    def test_should_respond_group_mentioned(self):
        chat = Chat(id=123, type="supergroup")
        bot = User(id=999, username="mybot")
        ctx = GroupContext(chat=chat, bot_user=bot, is_mentioned=True)
        assert ctx.should_respond is True
    
    def test_should_respond_group_reply(self):
        chat = Chat(id=123, type="supergroup")
        bot = User(id=999, username="mybot")
        ctx = GroupContext(chat=chat, bot_user=bot, is_reply_to_bot=True)
        assert ctx.should_respond is True
    
    def test_should_not_respond_group_no_mention(self):
        chat = Chat(id=123, type="supergroup")
        bot = User(id=999, username="mybot")
        ctx = GroupContext(chat=chat, bot_user=bot, is_mentioned=False, is_reply_to_bot=False)
        assert ctx.should_respond is False


class TestGroupMessageHandler:
    def test_process_private_message(self):
        handler = GroupMessageHandler("mybot")
        bot = User(id=999, username="mybot")
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Hello"
        )
        
        ctx = handler.process_message(msg, bot)
        assert ctx.should_respond is True
        assert ctx.is_mentioned is True
        assert ctx.mention_text == "Hello"
    
    def test_process_group_mention(self):
        handler = GroupMessageHandler("mybot")
        bot = User(id=999, username="mybot")
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="supergroup", title="Test Group")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Hello @mybot how are you?"
        )
        
        ctx = handler.process_message(msg, bot)
        assert ctx.should_respond is True
        assert ctx.is_mentioned is True
        assert ctx.mention_text == "Hello  how are you?"
    
    def test_process_group_case_insensitive_mention(self):
        handler = GroupMessageHandler("MyBot")
        bot = User(id=999, username="mybot")
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="supergroup")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Hey @MYBOT"
        )
        
        ctx = handler.process_message(msg, bot)
        assert ctx.is_mentioned is True
    
    def test_process_group_no_mention(self):
        handler = GroupMessageHandler("mybot")
        bot = User(id=999, username="mybot")
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="supergroup")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Just chatting here"
        )
        
        ctx = handler.process_message(msg, bot)
        assert ctx.should_respond is False
        assert ctx.is_mentioned is False
    
    def test_process_reply_to_bot(self):
        handler = GroupMessageHandler("mybot")
        bot = User(id=999, username="mybot")
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="supergroup")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Thanks!",
            raw_data={
                "message_id": 1,
                "chat": {"id": 456, "type": "supergroup"},
                "from": {"id": 123, "first_name": "Test"},
                "date": 1700000000,
                "text": "Thanks!",
                "reply_to_message": {
                    "message_id": 2,
                    "from": {"id": 999, "username": "mybot"}
                }
            }
        )
        
        ctx = handler.process_message(msg, bot)
        assert ctx.should_respond is True
        assert ctx.is_reply_to_bot is True
    
    def test_get_response_text_with_mention(self):
        handler = GroupMessageHandler("mybot")
        bot = User(id=999)
        
        user = User(id=123)
        chat = Chat(id=456, type="supergroup")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="@mybot hello"
        )
        
        ctx = handler.process_message(msg, bot)
        result = handler.get_response_text(msg, ctx)
        assert result == "hello"
    
    def test_get_response_text_should_not_respond(self):
        handler = GroupMessageHandler("mybot")
        bot = User(id=999)
        
        user = User(id=123)
        chat = Chat(id=456, type="supergroup")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Just chatting"
        )
        
        ctx = handler.process_message(msg, bot)
        result = handler.get_response_text(msg, ctx)
        assert result is None


class TestGroupCommandRouter:
    def test_register_and_get_handler(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            return "handled"
        
        router.register("test", handler)
        
        chat = Chat(id=123, type="private")
        result = router.get_handler("/test", chat, None)
        assert result is handler
    
    def test_get_handler_with_bot_suffix(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            return "handled"
        
        router.register("test", handler)
        
        chat = Chat(id=123, type="supergroup")
        result = router.get_handler("/test@mybot", chat, None)
        assert result is handler
    
    def test_get_handler_wrong_bot_suffix(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            return "handled"
        
        router.register("test", handler)
        
        chat = Chat(id=123, type="supergroup")
        result = router.get_handler("/test@otherbot", chat, None)
        assert result is None
    
    def test_private_only_handler_in_group(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            return "handled"
        
        router.register("secret", handler, private_only=True)
        
        # Should work in private
        private_chat = Chat(id=123, type="private")
        result = router.get_handler("/secret", private_chat, None)
        assert result is handler
        
        # Should not work in group
        group_chat = Chat(id=123, type="supergroup")
        result = router.get_handler("/secret", group_chat, None)
        assert result is None
    
    def test_group_only_handler_in_private(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            return "handled"
        
        router.register("announce", handler, group_only=True)
        
        # Should not work in private
        private_chat = Chat(id=123, type="private")
        result = router.get_handler("/announce", private_chat, None)
        assert result is None
        
        # Should work in group
        group_chat = Chat(id=123, type="supergroup")
        result = router.get_handler("/announce", group_chat, None)
        assert result is handler
    
    def test_admin_only_handler(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            return "handled"
        
        router.register("ban", handler, admin_only=True)
        
        chat = Chat(id=123, type="supergroup")
        
        # Non-admin should not work
        result = router.get_handler("/ban", chat, None, is_admin=False)
        assert result is None
        
        # Admin should work
        result = router.get_handler("/ban", chat, None, is_admin=True)
        assert result is handler
    
    def test_is_valid_command(self):
        router = GroupCommandRouter("mybot")
        
        def handler():
            pass
        
        router.register("valid", handler)
        
        assert router.is_valid_command("/valid") is True
        assert router.is_valid_command("/valid@mybot") is True
        assert router.is_valid_command("/invalid") is False


class TestGroupMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_sets_context(self):
        middleware = GroupMiddleware("mybot")
        bot = User(id=999, username="mybot")
        middleware.bot_user = bot
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="private")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Hello"
        )
        
        context = MagicMock()
        context.bot_user = bot
        
        async def next_fn(m, c):
            return "next_result"
        
        result = await middleware(msg, context, next_fn)
        
        assert hasattr(context, 'group_ctx')
        assert context.group_ctx.should_respond is True
        assert result == "next_result"
    
    @pytest.mark.asyncio
    async def test_middleware_skips_non_mentioned_group_messages(self):
        middleware = GroupMiddleware("mybot")
        bot = User(id=999, username="mybot")
        middleware.bot_user = bot
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="supergroup", title="Test Group")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Just chatting"
        )
        
        context = MagicMock()
        context.bot_user = bot
        
        async def next_fn(m, c):
            return "should_not_be_called"
        
        result = await middleware(msg, context, next_fn)
        
        # Should not call next middleware for non-mentioned group messages
        assert result is None
    
    @pytest.mark.asyncio
    async def test_middleware_processes_mentioned_group_messages(self):
        middleware = GroupMiddleware("mybot")
        bot = User(id=999, username="mybot")
        middleware.bot_user = bot
        
        user = User(id=123, first_name="Test")
        chat = Chat(id=456, type="supergroup")
        msg = TextMessage(
            message_id=1, chat=chat, from_user=user,
            date=datetime.now(), text="Hey @mybot"
        )
        
        context = MagicMock()
        context.bot_user = bot
        
        async def next_fn(m, c):
            return "processed"
        
        result = await middleware(msg, context, next_fn)
        
        assert result == "processed"
        assert context.group_ctx.is_mentioned is True
