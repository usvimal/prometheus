"""
Group integration for supervisor.

Integrates the enhanced Telegram group handler with the existing supervisor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from prometheus.channels.group_handler import GroupHandler
from supervisor.state import load_state, save_state

log = logging.getLogger(__name__)

# Global handler instance
_group_handler: Optional[GroupHandler] = None

def init_group_handler(bot_username: str, bot_id: int) -> None:
    """Initialize the group handler with bot info."""
    global _group_handler
    _group_handler = GroupHandler(
        bot_username=bot_username,
        bot_id=bot_id,
        state_dir="data/state"
    )
    log.info("Group handler initialized for @%s (id=%d)", bot_username, bot_id)

def get_group_handler() -> Optional[GroupHandler]:
    """Get the group handler instance."""
    return _group_handler

def process_group_message(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a Telegram update for group messages.
    
    Returns the processed message if it should be handled,
    or None if it should be ignored (e.g., not @mentioned in group).
    """
    if _group_handler is None:
        return None
    
    result = _group_handler.process_message(update)
    
    if result.action == "ignore":
        return None
    
    if result.action == "respond":
        # Build message context for the agent
        return {
            "chat_id": result.chat_id,
            "user_id": result.user_id,
            "text": result.text,
            "message_id": result.message_id,
            "is_group": result.is_group,
            "group_title": result.group_title,
            "reply_to_message_id": result.reply_to_message_id,
            "thread_id": result.thread_id,
            "context": result.context,
        }
    
    return None

def should_respond_in_group(update: Dict[str, Any]) -> bool:
    """
    Quick check if we should respond to a group message.
    
    Used by supervisor to filter messages before processing.
    """
    if _group_handler is None:
        return False
    
    message = update.get("message", {})
    chat = message.get("chat", {})
    
    # Not a group
    if chat.get("type") not in ("group", "supergroup"):
        return True  # Private chat - always respond
    
    # Check if @mentioned or reply
    return _group_handler.should_respond(message)

def get_bot_info_from_telegram(tg_client) -> tuple[str, int]:
    """Fetch bot info from Telegram API."""
    import requests
    
    try:
        r = requests.get(f"{tg_client.base}/getMe", timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("ok"):
            result = data["result"]
            username = result.get("username", "")
            bot_id = result.get("id", 0)
            return username, bot_id
    except Exception as e:
        log.error("Failed to get bot info: %s", e)
    
    # Fallback to env vars
    import os
    username = os.environ.get("TELEGRAM_BOT_USERNAME", "")
    bot_id = int(os.environ.get("TELEGRAM_BOT_ID", "0"))
    return username, bot_id
