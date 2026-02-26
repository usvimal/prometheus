"""
Structured message types for Telegram channel.

Replaces raw dict parsing with typed message classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum, auto


class MessageType(Enum):
    TEXT = auto()
    COMMAND = auto()
    PHOTO = auto()
    DOCUMENT = auto()
    AUDIO = auto()
    VIDEO = auto()
    VOICE = auto()
    LOCATION = auto()
    CONTACT = auto()
    CALLBACK = auto()
    UNKNOWN = auto()


@dataclass
class User:
    """Telegram user info."""
    id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_bot: bool = False
    language_code: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        """Best available name for the user."""
        if self.username:
            return f"@{self.username}"
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or f"user_{self.id}"


@dataclass  
class Chat:
    """Telegram chat info."""
    id: int
    type: str  # private, group, supergroup, channel
    title: Optional[str] = None
    username: Optional[str] = None
    
    @property
    def is_private(self) -> bool:
        return self.type == "private"
    
    @property
    def is_group(self) -> bool:
        return self.type in ("group", "supergroup")
    
    @property
    def is_channel(self) -> bool:
        return self.type == "channel"


@dataclass
class Message:
    """Base message class."""
    message_id: int
    chat: Chat
    from_user: Optional[User]
    date: datetime
    message_type: MessageType = MessageType.UNKNOWN
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> Optional[Message]:
        """Factory method to create appropriate message type from Telegram update."""
        message_data = update.get("message") or update.get("edited_message")
        callback_data = update.get("callback_query")
        
        if callback_data:
            return CallbackMessage.from_telegram_update(update)
        
        if not message_data:
            return None
            
        # Determine message type
        if message_data.get("text", "").startswith("/"):
            return CommandMessage.from_telegram_update(update)
        elif message_data.get("photo"):
            return PhotoMessage.from_telegram_update(update)
        elif message_data.get("document"):
            return DocumentMessage.from_telegram_update(update)
        elif message_data.get("audio"):
            return AudioMessage.from_telegram_update(update)
        elif message_data.get("video"):
            return VideoMessage.from_telegram_update(update)
        elif message_data.get("location"):
            return LocationMessage.from_telegram_update(update)
        elif message_data.get("contact"):
            return ContactMessage.from_telegram_update(update)
        elif message_data.get("text"):
            return TextMessage.from_telegram_update(update)
        else:
            # Generic message for other types
            return cls._parse_base(message_data, MessageType.UNKNOWN)
    
    @classmethod
    def _parse_base(cls, message_data: Dict[str, Any], msg_type: MessageType) -> Message:
        """Parse common fields from Telegram message data."""
        from_data = message_data.get("from", {})
        chat_data = message_data.get("chat", {})
        
        user = User(
            id=from_data.get("id", 0),
            username=from_data.get("username"),
            first_name=from_data.get("first_name"),
            last_name=from_data.get("last_name"),
            is_bot=from_data.get("is_bot", False),
            language_code=from_data.get("language_code"),
        ) if from_data else None
        
        chat = Chat(
            id=chat_data.get("id", 0),
            type=chat_data.get("type", "unknown"),
            title=chat_data.get("title"),
            username=chat_data.get("username"),
        )
        
        date_ts = message_data.get("date", 0)
        date = datetime.fromtimestamp(date_ts) if date_ts else datetime.now()
        
        return cls(
            message_id=message_data.get("message_id", 0),
            chat=chat,
            from_user=user,
            date=date,
            message_type=msg_type,
            raw_data=message_data,
        )


@dataclass
class TextMessage(Message):
    """Plain text message."""
    text: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.TEXT
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> TextMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.TEXT)
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.TEXT,
            raw_data=base.raw_data,
            text=message_data.get("text", ""),
            entities=message_data.get("entities", []),
        )


@dataclass
class CommandMessage(TextMessage):
    """Command message (e.g., /start, /help)."""
    command: str = ""
    args: str = ""
    
    def __post_init__(self):
        self.message_type = MessageType.COMMAND
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> CommandMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.COMMAND)
        text = message_data.get("text", "")
        
        # Parse command and args
        parts = text.split(maxsplit=1)
        command = parts[0][1:] if parts else ""  # Remove leading /
        args = parts[1] if len(parts) > 1 else ""
        
        # Remove bot username from command if present
        if "@" in command:
            command = command.split("@")[0]
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.COMMAND,
            raw_data=base.raw_data,
            text=text,
            entities=message_data.get("entities", []),
            command=command.lower(),
            args=args,
        )


@dataclass
class PhotoMessage(Message):
    """Photo message with caption."""
    caption: Optional[str] = None
    photo_sizes: List[Dict[str, Any]] = field(default_factory=list)
    file_id: Optional[str] = None  # Largest photo
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.PHOTO
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> PhotoMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.PHOTO)
        
        photos = message_data.get("photo", [])
        # Get largest photo (last in array)
        largest = photos[-1] if photos else None
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.PHOTO,
            raw_data=base.raw_data,
            caption=message_data.get("caption"),
            photo_sizes=photos,
            file_id=largest.get("file_id") if largest else None,
        )


@dataclass
class DocumentMessage(Message):
    """Document/file message."""
    file_id: str = ""
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: int = 0
    caption: Optional[str] = None
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.DOCUMENT
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> DocumentMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.DOCUMENT)
        doc = message_data.get("document", {})
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.DOCUMENT,
            raw_data=base.raw_data,
            file_id=doc.get("file_id", ""),
            file_name=doc.get("file_name"),
            mime_type=doc.get("mime_type"),
            file_size=doc.get("file_size", 0),
            caption=message_data.get("caption"),
        )


@dataclass
class AudioMessage(Message):
    """Audio message (music file)."""
    file_id: str = ""
    duration: int = 0
    performer: Optional[str] = None
    title: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: int = 0
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.AUDIO
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> AudioMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.AUDIO)
        audio = message_data.get("audio", {})
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.AUDIO,
            raw_data=base.raw_data,
            file_id=audio.get("file_id", ""),
            duration=audio.get("duration", 0),
            performer=audio.get("performer"),
            title=audio.get("title"),
            mime_type=audio.get("mime_type"),
            file_size=audio.get("file_size", 0),
        )


@dataclass
class VideoMessage(Message):
    """Video message."""
    file_id: str = ""
    width: int = 0
    height: int = 0
    duration: int = 0
    caption: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: int = 0
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.VIDEO
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> VideoMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.VIDEO)
        video = message_data.get("video", {})
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.VIDEO,
            raw_data=base.raw_data,
            file_id=video.get("file_id", ""),
            width=video.get("width", 0),
            height=video.get("height", 0),
            duration=video.get("duration", 0),
            caption=message_data.get("caption"),
            mime_type=video.get("mime_type"),
            file_size=video.get("file_size", 0),
        )


@dataclass
class LocationMessage(Message):
    """Location message."""
    latitude: float = 0.0
    longitude: float = 0.0
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.LOCATION
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> LocationMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.LOCATION)
        location = message_data.get("location", {})
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.LOCATION,
            raw_data=base.raw_data,
            latitude=location.get("latitude", 0.0),
            longitude=location.get("longitude", 0.0),
        )


@dataclass
class ContactMessage(Message):
    """Contact message."""
    phone_number: str = ""
    first_name: str = ""
    last_name: Optional[str] = None
    user_id: Optional[int] = None
    vcard: Optional[str] = None
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.CONTACT
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> ContactMessage:
        message_data = update.get("message") or update.get("edited_message", {})
        base = cls._parse_base(message_data, MessageType.CONTACT)
        contact = message_data.get("contact", {})
        
        return cls(
            message_id=base.message_id,
            chat=base.chat,
            from_user=base.from_user,
            date=base.date,
            message_type=MessageType.CONTACT,
            raw_data=base.raw_data,
            phone_number=contact.get("phone_number", ""),
            first_name=contact.get("first_name", ""),
            last_name=contact.get("last_name"),
            user_id=contact.get("user_id"),
            vcard=contact.get("vcard"),
        )


@dataclass
class CallbackMessage(Message):
    """Callback query from inline keyboard."""
    callback_id: str = ""
    data: str = ""
    original_message: Optional[Message] = None
    
    def __post_init__(self):
        if self.message_type == MessageType.UNKNOWN:
            self.message_type = MessageType.CALLBACK
    
    @classmethod
    def from_telegram_update(cls, update: Dict[str, Any]) -> CallbackMessage:
        callback_data = update.get("callback_query", {})
        
        from_data = callback_data.get("from", {})
        user = User(
            id=from_data.get("id", 0),
            username=from_data.get("username"),
            first_name=from_data.get("first_name"),
            last_name=from_data.get("last_name"),
            is_bot=from_data.get("is_bot", False),
        )
        
        # Get chat from original message
        msg_data = callback_data.get("message", {})
        chat_data = msg_data.get("chat", {})
        chat = Chat(
            id=chat_data.get("id", 0),
            type=chat_data.get("type", "private"),
            title=chat_data.get("title"),
            username=chat_data.get("username"),
        )
        
        return cls(
            message_id=msg_data.get("message_id", 0),
            chat=chat,
            from_user=user,
            date=datetime.now(),
            message_type=MessageType.CALLBACK,
            raw_data=callback_data,
            callback_id=callback_data.get("id", ""),
            data=callback_data.get("data", ""),
        )


# Type alias for message handlers
MessageHandler = Any  # Callable[[Message], Any]