"""
Tests for message_router.py
"""

import pytest
from unittest.mock import MagicMock, patch
from supervisor.message_router import (
    MessageContext,
    MessageRouter,
    classify_message,
    get_bot_username,
    clear_bot_cache,
)


class TestBotUsername:
    def test_get_bot_username_caches_result(self):
        clear_bot_cache()
        mock_tg = MagicMock()
        mock_tg.base = "https://api.telegram.org/botTOKEN"
        
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "ok": True,
                "result": {"username": "test_bot"}
            }
            mock_get.return_value.raise_for_status = MagicMock()
            
            # First call should fetch
            result1 = get_bot_username(mock_tg)
            assert result1 == "test_bot"
            assert mock_get.call_count == 1
            
            # Second call should use cache
            result2 = get_bot_username(mock_tg)
            assert result2 == "test_bot"
            assert mock_get.call_count == 1  # No additional call
    
    def test_get_bot_username_handles_error(self):
        clear_bot_cache()
        mock_tg = MagicMock()
        mock_tg.base = "https://api.telegram.org/botTOKEN"
        
        with patch("requests.get", side_effect=Exception("Network error")):
            result = get_bot_username(mock_tg)
            assert result is None


class TestClassifyMessage:
    def test_private_message(self):
        update = {
            "message": {
                "message_id": 123,
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 789, "username": "testuser"},
                "text": "Hello bot",
            }
        }
        
        ctx = classify_message(update, "my_bot")
        
        assert ctx is not None
        assert ctx.chat_id == 456
        assert ctx.chat_type == "private"
        assert ctx.user_id == 789
        assert ctx.username == "testuser"
        assert ctx.text == "Hello bot"
        assert ctx.is_bot_mention is False
    
    def test_group_message_with_mention(self):
        update = {
            "message": {
                "message_id": 123,
                "chat": {"id": -100123, "type": "supergroup"},
                "from": {"id": 789, "username": "testuser"},
                "text": "Hey @my_bot check this out",
            }
        }
        
        ctx = classify_message(update, "my_bot")
        
        assert ctx is not None
        assert ctx.chat_type == "supergroup"
        assert ctx.is_bot_mention is True
        assert ctx.text == "Hey check this out"  # Mention stripped
    
    def test_group_message_reply_to_bot(self):
        update = {
            "message": {
                "message_id": 123,
                "chat": {"id": -100123, "type": "group"},
                "from": {"id": 789, "username": "testuser"},
                "text": "Thanks for the help",
                "reply_to_message": {
                    "message_id": 100,
                    "from": {"is_bot": True, "username": "my_bot"},
                }
            }
        }
        
        ctx = classify_message(update, "my_bot")
        
        assert ctx is not None
        assert ctx.is_reply_to_bot is True
        assert ctx.reply_to_message_id == 100
    
    def test_no_message_in_update(self):
        update = {"edited_message": {}}  # No "message" key
        ctx = classify_message(update, "my_bot")
        assert ctx is None


class TestMessageRouter:
    @pytest.fixture
    def router(self):
        mock_tg = MagicMock()
        mock_handler = MagicMock()
        return MessageRouter(mock_tg, mock_handler, MagicMock())
    
    def test_should_process_private_message(self, router):
        ctx = MessageContext(
            chat_id=123,
            chat_type="private",
            user_id=456,
            username="user",
            text="Hello",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=False,
            bot_username=None,
            raw_message={},
        )
        assert router.should_process_message(ctx) is True
    
    def test_should_process_group_with_mention(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="supergroup",
            user_id=456,
            username="user",
            text="Hello",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=True,
            is_reply_to_bot=False,
            bot_username="bot",
            raw_message={},
        )
        assert router.should_process_message(ctx) is True
    
    def test_should_process_group_reply_to_bot(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="group",
            user_id=456,
            username="user",
            text="Hello",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=True,
            bot_username="bot",
            raw_message={},
        )
        assert router.should_process_message(ctx) is True
    
    def test_should_process_group_command(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="group",
            user_id=456,
            username="user",
            text="/status",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=False,
            bot_username="bot",
            raw_message={},
        )
        assert router.should_process_message(ctx) is True
    
    def test_should_not_process_random_group_message(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="group",
            user_id=456,
            username="user",
            text="Just chatting here",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=False,
            bot_username="bot",
            raw_message={},
        )
        assert router.should_process_message(ctx) is False
    
    def test_should_not_process_channel(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="channel",
            user_id=456,
            username="user",
            text="Channel post",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=False,
            bot_username=None,
            raw_message={},
        )
        assert router.should_process_message(ctx) is False
    
    def test_route_private_message(self, router):
        ctx = MessageContext(
            chat_id=123,
            chat_type="private",
            user_id=456,
            username="user",
            text="Hello bot",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=False,
            bot_username=None,
            raw_message={},
        )
        
        result = router.route_message(ctx)
        
        assert result is not None
        assert result["type"] == "direct_chat"
        assert result["chat_id"] == 123
        assert result["text"] == "Hello bot"
    
    def test_route_group_message(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="supergroup",
            user_id=456,
            username="user",
            text="@bot help me",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=True,
            is_reply_to_bot=False,
            bot_username="bot",
            raw_message={},
        )
        
        result = router.route_message(ctx)
        
        assert result is not None
        assert result["is_group_message"] is True
        assert result["is_bot_mention"] is True
        assert result["message_id"] == 1
        
        # Check conversation tracking
        assert -100123 in router._group_conversations
    
    def test_route_skips_untriggered_messages(self, router):
        ctx = MessageContext(
            chat_id=-100123,
            chat_type="group",
            user_id=456,
            username="user",
            text="Random chat",
            message_id=1,
            reply_to_message_id=None,
            is_bot_mention=False,
            is_reply_to_bot=False,
            bot_username="bot",
            raw_message={},
        )
        
        result = router.route_message(ctx)
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
