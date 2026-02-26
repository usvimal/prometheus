"""
Telegram group chat support for Prometheus.

Handles:
- @mention detection (respond when mentioned in groups)
- Reply threading (respond in context)
- Group-specific command prefixes
- Admin/member permission checks
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from .message_types import Message, CommandMessage, TextMessage, Chat, User

log = logging.getLogger(__name__)


@dataclass
class GroupContext:
    """Context for group interactions."""
    chat: Chat
    bot_user: User
    is_mentioned: bool = False
    is_reply_to_bot: bool = False
    mention_text: str = ""  # Text with @botname stripped
    
    @property
    def should_respond(self) -> bool:
        """Determine if bot should respond to this message."""
        # Always respond in private chats
        if self.chat.is_private:
            return True
        # Respond if mentioned or replied to
        return self.is_mentioned or self.is_reply_to_bot


class GroupMessageHandler:
    """
    Handles group-specific message logic:
    - Detects @mentions
    - Handles reply threading
    - Manages group context
    """
    
    def __init__(self, bot_username: str):
        self.bot_username = bot_username.lower()
        self.bot_mention_pattern = re.compile(
            rf'@{re.escape(bot_username)}\b',
            re.IGNORECASE
        )
        self._group_admins: Dict[int, List[int]] = {}  # chat_id -> list of admin user_ids
    
    def process_message(self, message: Message, bot_user: User) -> GroupContext:
        """
        Process a message and determine group context.
        
        Returns GroupContext with should_respond flag.
        """
        ctx = GroupContext(
            chat=message.chat,
            bot_user=bot_user,
        )
        
        # Private chat - always respond
        if message.chat.is_private:
            ctx.is_mentioned = True
            if isinstance(message, (TextMessage, CommandMessage)):
                ctx.mention_text = message.text
            return ctx
        
        # Group chat - check for mention or reply
        if isinstance(message, (TextMessage, CommandMessage)):
            text = message.text or ""
            
            # Check for @mention
            if self.bot_mention_pattern.search(text):
                ctx.is_mentioned = True
                # Strip the mention from text
                ctx.mention_text = self.bot_mention_pattern.sub('', text).strip()
            
            # Check if it's a reply to the bot
            if message.raw_data.get("reply_to_message"):
                reply_to = message.raw_data["reply_to_message"]
                reply_from = reply_to.get("from", {})
                if reply_from.get("id") == bot_user.id:
                    ctx.is_reply_to_bot = True
                    if not ctx.mention_text:
                        ctx.mention_text = text
        
        return ctx
    
    def get_response_text(self, message: Message, ctx: GroupContext) -> Optional[str]:
        """
        Extract the text the bot should respond to.
        
        Returns None if bot shouldn't respond.
        """
        if not ctx.should_respond:
            return None
        
        if ctx.mention_text:
            return ctx.mention_text
        
        if isinstance(message, (TextMessage, CommandMessage)):
            return message.text
        
        return None


class GroupCommandRouter:
    """
    Routes commands differently for groups vs private chats.
    
    In groups:
    - Commands can be prefixed with /command@botname
    - Some commands may be admin-only
    """
    
    def __init__(self, bot_username: str):
        self.bot_username = bot_username.lower()
        self._handlers: Dict[str, Callable] = {}
        self._admin_handlers: set = set()
        self._group_only_handlers: set = set()
        self._private_only_handlers: set = set()
    
    def register(self, command: str, handler: Callable, 
                 admin_only: bool = False,
                 group_only: bool = False,
                 private_only: bool = False) -> None:
        """Register a command handler with optional restrictions."""
        cmd = command.lower().lstrip('/')
        self._handlers[cmd] = handler
        
        if admin_only:
            self._admin_handlers.add(cmd)
        if group_only:
            self._group_only_handlers.add(cmd)
        if private_only:
            self._private_only_handlers.add(cmd)
    
    def get_handler(self, command: str, chat: Chat, user: Optional[User],
                    is_admin: bool = False) -> Optional[Callable]:
        """
        Get the appropriate handler for a command.
        
        Returns None if command is restricted for this context.
        """
        # Strip @botname suffix if present
        cmd = command.lower().lstrip('/')
        if '@' in cmd:
            cmd, mentioned_bot = cmd.split('@', 1)
            # If @botname is specified but it's not us, ignore
            if mentioned_bot.lower() != self.bot_username:
                return None
        
        handler = self._handlers.get(cmd)
        if not handler:
            return None
        
        # Check restrictions
        is_group = chat.is_group or chat.is_channel
        is_private = chat.is_private
        
        if cmd in self._private_only_handlers and not is_private:
            return None
        if cmd in self._group_only_handlers and not is_group:
            return None
        if cmd in self._admin_handlers and not is_admin:
            return None
        
        return handler
    
    def is_valid_command(self, command: str) -> bool:
        """Check if a command is registered (ignoring @botname suffix)."""
        cmd = command.lower().lstrip('/')
        if '@' in cmd:
            cmd = cmd.split('@')[0]
        return cmd in self._handlers


class GroupMiddleware:
    """
    Middleware for group-specific processing.
    
    Integrates with the enhanced Telegram channel.
    """
    
    def __init__(self, bot_username: str):
        self.handler = GroupMessageHandler(bot_username)
        self.router = GroupCommandRouter(bot_username)
        self.bot_user: Optional[User] = None
    
    async def __call__(self, message: Message, context: Any, next_middleware: Callable) -> Any:
        """
        Process message through group middleware.
        
        Sets context.group_ctx for downstream handlers.
        """
        if not self.bot_user:
            # Try to get bot user from context
            self.bot_user = getattr(context, 'bot_user', None)
        
        if not self.bot_user:
            log.warning("Bot user not set, skipping group processing")
            return await next_middleware(message, context)
        
        # Process group context
        group_ctx = self.handler.process_message(message, self.bot_user)
        context.group_ctx = group_ctx
        
        # For commands, check if we should handle them
        if isinstance(message, CommandMessage):
            handler = self.router.get_handler(
                message.command,
                message.chat,
                message.from_user,
                is_admin=False  # TODO: Implement admin check
            )
            
            if handler:
                return await handler(message, context)
            elif self.router.is_valid_command(message.command):
                # Command exists but restricted for this context
                return await self._send_restriction_message(message, context)
        
        # Continue to next middleware if we shouldn't respond
        if not group_ctx.should_respond:
            # Silently ignore - don't call next middleware
            return None
        
        return await next_middleware(message, context)
    
    async def _send_restriction_message(self, message: CommandMessage, context: Any) -> None:
        """Send a message explaining command restrictions."""
        # This would be implemented to send a Telegram message
        # For now, just log
        log.info("Command %s restricted for chat %s", message.command, message.chat.id)


def create_group_aware_channel(bot_token: str, bot_username: str, owner_id: int):
    """
    Factory function to create a group-aware Telegram channel.
    
    This integrates with the existing EnhancedTelegramChannel.
    """
    from .telegram_enhanced import EnhancedTelegramChannel, TelegramConfig
    
    config = TelegramConfig(
        bot_token=bot_token,
        owner_id=owner_id,
    )
    
    channel = EnhancedTelegramChannel(config)
    
    # Add group middleware
    group_middleware = GroupMiddleware(bot_username)
    channel.use(group_middleware)
    
    return channel, group_middleware
