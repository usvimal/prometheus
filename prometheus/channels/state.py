"""
Conversational state management for Telegram channel.

Tracks multi-turn conversations with state persistence.
Supports both in-memory and Redis backends.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum, auto

log = logging.getLogger(__name__)


class ConversationState(Enum):
    """Possible states in a conversation flow."""
    IDLE = auto()
    AWAITING_INPUT = auto()
    AWAITING_CONFIRMATION = auto()
    AWAITING_SELECTION = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    EXPIRED = auto()


@dataclass
class ConversationContext:
    """Context for a single conversation."""
    conversation_id: str  # Unique ID for this conversation
    user_id: int
    chat_id: int
    state: ConversationState = ConversationState.IDLE
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.expires_at is None:
            # Default 30 minute expiration
            self.expires_at = time.time() + 1800
    
    def is_expired(self) -> bool:
        """Check if conversation has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.updated_at = time.time()
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to conversation history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            **kwargs
        })
        self.touch()
    
    def set_data(self, key: str, value: Any) -> None:
        """Store data in conversation context."""
        self.data[key] = value
        self.touch()
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from conversation context."""
        return self.data.get(key, default)
    
    def transition_to(self, new_state: ConversationState) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.touch()
        log.debug("Conversation %s: %s -> %s", 
                  self.conversation_id, old_state.name, new_state.name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "state": self.state.name,
            "data": self.data,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationContext:
        """Deserialize from dictionary."""
        ctx = cls(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            state=ConversationState[data.get("state", "IDLE")],
            data=data.get("data", {}),
            history=data.get("history", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
        )
        return ctx


class StateBackend(ABC):
    """Abstract base for state storage backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[ConversationContext]:
        """Get conversation by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, context: ConversationContext) -> None:
        """Store conversation."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete conversation. Returns True if existed."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        pass
    
    @abstractmethod
    async def expire(self, key: str, seconds: int) -> None:
        """Set expiration on a key."""
        pass


class MemoryStateBackend(StateBackend):
    """In-memory state storage (for development/single-instance)."""
    
    def __init__(self, default_ttl: int = 3600):
        self._store: Dict[str, ConversationContext] = {}
        self._expirations: Dict[str, float] = {}
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[ConversationContext]:
        # Check expiration
        if key in self._expirations:
            if time.time() > self._expirations[key]:
                await self.delete(key)
                return None
        
        ctx = self._store.get(key)
        if ctx and ctx.is_expired():
            await self.delete(key)
            return None
        
        return ctx
    
    async def set(self, key: str, context: ConversationContext) -> None:
        self._store[key] = context
        self._expirations[key] = time.time() + self.default_ttl
    
    async def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        self._expirations.pop(key, None)
        return existed
    
    async def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def expire(self, key: str, seconds: int) -> None:
        self._expirations[key] = time.time() + seconds
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [
            k for k, exp in self._expirations.items()
            if now > exp or (k in self._store and self._store[k].is_expired())
        ]
        for k in expired:
            await self.delete(k)
        return len(expired)


class RedisStateBackend(StateBackend):
    """Redis-backed state storage (for production/multi-instance)."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 key_prefix: str = "prometheus:conv:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None
        self._connect()
    
    def _connect(self):
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self.redis_url)
        except ImportError:
            log.error("redis not installed. Run: pip install redis")
            raise
    
    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[ConversationContext]:
        if not self._redis:
            return None
        
        data = await self._redis.get(self._make_key(key))
        if not data:
            return None
        
        try:
            ctx_dict = json.loads(data)
            return ConversationContext.from_dict(ctx_dict)
        except (json.JSONDecodeError, KeyError) as e:
            log.error("Failed to deserialize conversation %s: %s", key, e)
            return None
    
    async def set(self, key: str, context: ConversationContext) -> None:
        if not self._redis:
            return
        
        data = json.dumps(context.to_dict())
        await self._redis.set(self._make_key(key), data)
    
    async def delete(self, key: str) -> bool:
        if not self._redis:
            return False
        
        result = await self._redis.delete(self._make_key(key))
        return result > 0
    
    async def keys(self, pattern: str = "*") -> List[str]:
        if not self._redis:
            return []
        
        full_pattern = self._make_key(pattern)
        keys = await self._redis.keys(full_pattern)
        # Strip prefix
        prefix_len = len(self.key_prefix)
        return [k[prefix_len:] if k.startswith(self.key_prefix) else k 
                for k in keys]
    
    async def expire(self, key: str, seconds: int) -> None:
        if self._redis:
            await self._redis.expire(self._make_key(key), seconds)


class StateManager:
    """High-level manager for conversation state."""
    
    def __init__(self, backend: Optional[StateBackend] = None):
        self.backend = backend or MemoryStateBackend()
        self._handlers: Dict[ConversationState, List[Callable]] = {
            state: [] for state in ConversationState
        }
        self._state_transitions: Dict[ConversationState, List[ConversationState]] = {}
    
    def _make_key(self, user_id: int, chat_id: int) -> str:
        """Create unique key for user+chat combination."""
        return f"{user_id}:{chat_id}"
    
    async def get_conversation(self, user_id: int, chat_id: int) -> Optional[ConversationContext]:
        """Get existing conversation or None."""
        key = self._make_key(user_id, chat_id)
        return await self.backend.get(key)
    
    async def create_conversation(self, user_id: int, chat_id: int,
                                   initial_state: ConversationState = ConversationState.IDLE,
                                   **kwargs) -> ConversationContext:
        """Create a new conversation."""
        import uuid
        
        key = self._make_key(user_id, chat_id)
        conv_id = str(uuid.uuid4())[:8]
        
        ctx = ConversationContext(
            conversation_id=conv_id,
            user_id=user_id,
            chat_id=chat_id,
            state=initial_state,
            metadata=kwargs,
        )
        
        await self.backend.set(key, ctx)
        log.debug("Created conversation %s for user %s", conv_id, user_id)
        return ctx
    
    async def get_or_create(self, user_id: int, chat_id: int,
                            **kwargs) -> ConversationContext:
        """Get existing conversation or create new one."""
        ctx = await self.get_conversation(user_id, chat_id)
        if ctx and not ctx.is_expired():
            return ctx
        return await self.create_conversation(user_id, chat_id, **kwargs)
    
    async def end_conversation(self, user_id: int, chat_id: int) -> bool:
        """End and delete a conversation."""
        key = self._make_key(user_id, chat_id)
        return await self.backend.delete(key)
    
    async def update_state(self, user_id: int, chat_id: int,
                          new_state: ConversationState) -> Optional[ConversationContext]:
        """Update conversation state."""
        ctx = await self.get_conversation(user_id, chat_id)
        if ctx:
            ctx.transition_to(new_state)
            await self.backend.set(self._make_key(user_id, chat_id), ctx)
        return ctx
    
    async def set_data(self, user_id: int, chat_id: int,
                      key: str, value: Any) -> Optional[ConversationContext]:
        """Set data in conversation context."""
        ctx = await self.get_conversation(user_id, chat_id)
        if ctx:
            ctx.set_data(key, value)
            await self.backend.set(self._make_key(user_id, chat_id), ctx)
        return ctx
    
    async def add_message(self, user_id: int, chat_id: int,
                         role: str, content: str, **kwargs) -> Optional[ConversationContext]:
        """Add message to conversation history."""
        ctx = await self.get_conversation(user_id, chat_id)
        if ctx:
            ctx.add_message(role, content, **kwargs)
            await self.backend.set(self._make_key(user_id, chat_id), ctx)
        return ctx
    
    def on_state(self, state: ConversationState):
        """Decorator to register handler for a specific state."""
        def decorator(func: Callable):
            self._handlers[state].append(func)
            return func
        return decorator
    
    async def handle_message(self, message, context: ConversationContext) -> Any:
        """Route message to appropriate state handler."""
        handlers = self._handlers.get(context.state, [])
        
        for handler in handlers:
            try:
                result = await handler(message, context)
                if result is not None:
                    return result
            except Exception as e:
                log.error("State handler error: %s", e)
        
        return None
    
    async def cleanup_expired(self) -> int:
        """Clean up expired conversations."""
        if isinstance(self.backend, MemoryStateBackend):
            return await self.backend.cleanup_expired()
        return 0
