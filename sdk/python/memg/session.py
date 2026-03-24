"""Session management with sliding window expiry."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .store import SQLiteStore
from .types import Message, Session

logger = logging.getLogger("memg")

DEFAULT_SESSION_TIMEOUT = 1800  # 30 minutes


def ensure_session(
    store: SQLiteStore,
    entity_uuid: str,
    process_uuid: str = "",
    timeout_seconds: int = DEFAULT_SESSION_TIMEOUT,
) -> Tuple[Session, bool]:
    """Get or create a session with sliding window expiry.

    Returns (session, is_new).
    """
    return store.ensure_session(entity_uuid, process_uuid, timeout_seconds)


def get_or_create_conversation(
    store: SQLiteStore,
    session_uuid: str,
    entity_uuid: str = "",
) -> str:
    """Get the active conversation for a session, or create a new one.

    Returns the conversation UUID.
    """
    conv = store.active_conversation(session_uuid)
    if conv is not None:
        return conv.uuid
    return store.start_conversation(session_uuid, entity_uuid)


def save_exchange(
    store: SQLiteStore,
    session_uuid: str,
    entity_uuid: str,
    user_messages: list,
    assistant_content: Optional[str] = None,
) -> Optional[str]:
    """Persist user messages and assistant response to the conversation log.

    user_messages: list of dicts with 'role' and 'content' keys,
                   or list of Message objects.
    Returns the conversation UUID.
    """
    if not session_uuid:
        return None

    conv_uuid = get_or_create_conversation(store, session_uuid, entity_uuid)

    for msg in user_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        elif isinstance(msg, Message):
            role = msg.role
            content = msg.content
        else:
            continue

        if role not in ("user", "assistant"):
            continue
        content = (content or "").strip()
        if not content:
            continue

        store.append_message(
            conv_uuid,
            Message(uuid="", conversation_id=conv_uuid, role=role, content=content),
        )

    if assistant_content and assistant_content.strip():
        store.append_message(
            conv_uuid,
            Message(
                uuid="",
                conversation_id=conv_uuid,
                role="assistant",
                content=assistant_content.strip(),
            ),
        )

    return conv_uuid


def load_recent_history(
    store: SQLiteStore,
    session_uuid: str,
    max_turns: int = 20,
) -> list:
    """Load recent conversation messages for working memory.

    Returns list of dicts with 'role' and 'content'.
    """
    conv = store.active_conversation(session_uuid)
    if conv is None:
        return []

    if max_turns > 0:
        messages = store.read_recent_messages(conv.uuid, max_turns)
    else:
        messages = store.read_messages(conv.uuid)

    return [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant") and m.content.strip()
    ]
