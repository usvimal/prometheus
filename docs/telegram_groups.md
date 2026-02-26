# Telegram Group Support

This document describes how Prometheus handles Telegram groups.

## Overview

Prometheus now supports group chats with the following features:

- **@mention detection**: Bot responds when mentioned in groups
- **Reply threading**: Bot responds to replies to its messages
- **Private chat mode**: Full access for owner in private chats
- **Admin commands**: Restricted commands only work for owner

## How It Works

### Message Routing

The `MessageRouter` class decides whether to process a message:

1. **Private chats** (owner): Always process
2. **Private chats** (non-owner): Ignore
3. **Groups** (mentioned): Process the message
4. **Groups** (reply to bot): Process the message
5. **Groups** (no mention): Ignore

### Mention Detection

The bot checks for mentions in several formats:
- `@bot_username` - Standard mention
- `@bot_username text` - Mention at start
- `text @bot_username` - Mention in middle
- `@bot_usernametext` - Mention without space (Telegram sometimes does this)

### Reply Threading

When someone replies to the bot's message:
- The bot extracts the original message
- Creates context for the conversation
- Responds in the same thread

## Configuration

### Bot Username

The bot automatically fetches its username from Telegram on startup.

### Environment Variables

No additional configuration needed. The bot uses existing:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_OWNER_ID`

## Usage Examples

### In Private Chat (Owner)
```
User: /status
Bot: 🔥 Prometheus 6.7.0...
```

### In Group (Mention)
```
User: @PrometheusBot what's the weather?
Bot: I don't have weather tools enabled...
```

### In Group (Reply)
```
Bot: I've completed the analysis
User: [reply] Can you explain more?
Bot: Certainly! Here's a deeper explanation...
```

### In Group (No Mention)
```
User: Hey everyone, what's up?
Bot: (no response - not mentioned)
```

## Security

- Only the owner can use admin commands (`/restart`, `/evolve`, etc.)
- Group members can only interact via @mentions or replies
- The bot never responds to unsolicited messages in groups

## Files

- `prometheus/channels/group_handler.py` - Core group logic
- `prometheus/channels/message_types.py` - Message type definitions
- `supervisor/message_router.py` - Message routing logic
- `supervisor/group_integration.py` - Integration utilities

## Testing

Run tests:
```bash
python -m pytest prometheus/channels/test_group_handler.py -v
python -m pytest supervisor/test_message_router.py -v
```

## Future Enhancements

- [ ] Thread/conversation persistence across restarts
- [ ] Rate limiting for group mentions
- [ ] Group-specific settings (e.g., only respond to admins)
- [ ] Inline query support in groups
