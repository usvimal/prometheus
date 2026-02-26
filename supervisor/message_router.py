"""
Supervisor — Message Router

Routes incoming Telegram messages to appropriate handlers based on chat type.
Integrates group handler with the existing direct chat system.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from supervisor.state import load_state, append_jsonl

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bot identity cache
# ---------------------------------------------------------------------------
_bot_info: Optional[Dict[str, Any]] = None
_bot_username: Optional[str] = None


def get_bot_username(tg_client) -> Optional[str]:
    """Get the bot's username (cached)."""
    global _bot_username, _bot_info
    
    if _bot_username:
        return _bot_username
    
    try:
        import requests
        r = requests.get(f"{tg_client.base}/getMe", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("ok"):
            _bot_info = data.get("result", {})
            _bot_username = _bot_info.get("username")
            return _bot_username
    except Exception as e:
        log.warning("Failed to get bot info: %s", e)
    
    return None


def clear_bot_cache():
    """Clear cached bot info (useful for testing)."""
    global _bot_info, _bot_username
    _bot_info = None
    _bot_username = None


# ---------------------------------------------------------------------------
# Message classification
# ---------------------------------------------------------------------------

@dataclass
class MessageContext:
    """Context for an incoming message."""
    chat_id: int
    chat_type: str  # 'private', 'group', 'supergroup', 'channel'
    user_id: int
    username: Optional[str]
    text: str
    message_id: int
    reply_to_message_id: Optional[int]
    is_bot_mention: bool
    is_reply_to_bot: bool
    bot_username: Optional[str]
    raw_message: Dict[str, Any]


def classify_message(update: Dict[str, Any], bot_username: Optional[str]) -> Optional[MessageContext]:
    """Classify an incoming Telegram update."""
    message = update.get("message", {})
    if not message:
        return None
    
    chat = message.get("chat", {})
    from_user = message.get("from", {})
    
    chat_id = chat.get("id")
    chat_type = chat.get("type", "private")
    user_id = from_user.get("id", 0)
    username = from_user.get("username")
    message_id = message.get("message_id", 0)
    text = message.get("text", "") or ""
    
    # Handle reply
    reply_to = message.get("reply_to_message", {})
    reply_to_message_id = reply_to.get("message_id") if reply_to else None
    
    # Check if message is a reply to the bot
    is_reply_to_bot = False
    if reply_to:
        reply_from = reply_to.get("from", {})
        if reply_from.get("is_bot") and bot_username:
            is_reply_to_bot = reply_from.get("username") == bot_username
    
    # Check for bot mention
    is_bot_mention = False
    if bot_username and text:
        # Check @username mention
        if f"@{bot_username}" in text:
            is_bot_mention = True
            # Remove the mention from text for cleaner processing
            text = re.sub(rf"\s*@{re.escape(bot_username)}\s*", " ", text).strip()
        
        # Check if bot was mentioned via text mention (for users without usernames)
        entities = message.get("entities", [])
        for entity in entities:
            if entity.get("type") == "mention":
                offset = entity.get("offset", 0)
                length = entity.get("length", 0)
                mention_text = text[offset:offset + length]
                if bot_username and mention_text == f"@{bot_username}":
                    is_bot_mention = True
    
    return MessageContext(
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        username=username,
        text=text,
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
        is_bot_mention=is_bot_mention,
        is_reply_to_bot=is_reply_to_bot,
        bot_username=bot_username,
        raw_message=message,
    )


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

class MessageRouter:
    """Routes messages to appropriate handlers based on chat type and context."""
    
    def __init__(
        self,
        tg_client,
        direct_chat_handler: Callable[[int, str, Optional[Tuple]], None],
        drive_root,
    ):
        self.tg_client = tg_client
        self.direct_chat_handler = direct_chat_handler
        self.drive_root = drive_root
        self._group_conversations: Dict[int, Dict[str, Any]] = {}
    
    def should_process_message(self, ctx: MessageContext) -> bool:
        """Determine if we should process this message."""
        # Always process private messages
        if ctx.chat_type == "private":
            return True
        
        # In groups, only process if:
        # 1. Bot is mentioned
        # 2. Message is a reply to bot
        # 3. Message starts with / (command)
        if ctx.chat_type in ("group", "supergroup"):
            if ctx.is_bot_mention:
                return True
            if ctx.is_reply_to_bot:
                return True
            if ctx.text.strip().startswith("/"):
                return True
            return False
        
        # Don't process channels for now
        if ctx.chat_type == "channel":
            return False
        
        return True
    
    def route_message(self, ctx: MessageContext) -> Optional[Dict[str, Any]]:
        """Route a message to the appropriate handler."""
        
        # Log the message
        self._log_message(ctx)
        
        # Check if we should process this message
        if not self.should_process_message(ctx):
            log.debug("Skipping message in %s (no trigger)", ctx.chat_type)
            return None
        
        # Route based on chat type
        if ctx.chat_type == "private":
            return self._handle_private_message(ctx)
        elif ctx.chat_type in ("group", "supergroup"):
            return self._handle_group_message(ctx)
        
        return None
    
    def _handle_private_message(self, ctx: MessageContext) -> Dict[str, Any]:
        """Handle a private message."""
        return {
            "type": "direct_chat",
            "chat_id": ctx.chat_id,
            "text": ctx.text,
            "user_id": ctx.user_id,
            "username": ctx.username,
        }
    
    def _handle_group_message(self, ctx: MessageContext) -> Dict[str, Any]:
        """Handle a group message."""
        # Build task with group context
        task = {
            "type": "task",
            "chat_id": ctx.chat_id,
            "text": ctx.text,
            "user_id": ctx.user_id,
            "username": ctx.username,
            "message_id": ctx.message_id,
            "reply_to_message_id": ctx.reply_to_message_id,
            "is_group_message": True,
            "is_bot_mention": ctx.is_bot_mention,
            "is_reply_to_bot": ctx.is_reply_to_bot,
        }
        
        # Store conversation context for reply threading
        self._group_conversations[ctx.chat_id] = {
            "last_message_id": ctx.message_id,
            "last_user_id": ctx.user_id,
            "last_interaction": __import__('time').time(),
        }
        
        return task
    
    def _log_message(self, ctx: MessageContext):
        """Log the message for analytics."""
        try:
            append_jsonl(
                self.drive_root / "logs" / "telegram_messages.jsonl",
                {
                    "ts": __import__('datetime').datetime.now(
                        __import__('datetime').timezone.utc
                    ).isoformat(),
                    "chat_id": ctx.chat_id,
                    "chat_type": ctx.chat_type,
                    "user_id": ctx.user_id,
                    "username": ctx.username,
                    "message_id": ctx.message_id,
                    "is_bot_mention": ctx.is_bot_mention,
                    "is_reply_to_bot": ctx.is_reply_to_bot,
                    "text_preview": ctx.text[:100] if ctx.text else "",
                },
            )
        except Exception:
            log.debug("Failed to log message", exc_info=True)


def create_router(tg_client, direct_chat_handler, drive_root) -> MessageRouter:
    """Factory function to create a message router."""
    return MessageRouter(
        tg_client=tg_client,
        direct_chat_handler=direct_chat_handler,
        drive_root=drive_root,
    )
