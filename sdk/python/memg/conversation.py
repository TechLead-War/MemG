"""Conversation message normalization and deduplication."""

from __future__ import annotations

from typing import List


def normalize_conversation_messages(messages: List[dict]) -> List[dict]:
    """Filter to user/assistant roles, trim content, remove empty messages.

    Logic matches Go memory/conversation.go NormalizeConversationMessages.
    """
    out = []
    for msg in messages:
        if msg is None:
            continue
        role = (msg.get("role") or "").strip()
        if role not in ("user", "assistant"):
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _overlap_length(existing: List[dict], incoming: List[dict]) -> int:
    """Find the longest tail of existing that matches a prefix of incoming."""
    existing_len = len(existing)
    max_overlap = min(len(incoming), existing_len)
    for overlap in range(max_overlap, 0, -1):
        match = True
        for i in range(overlap):
            e = existing[existing_len - overlap + i]
            m = incoming[i]
            if e.get("role") != m.get("role") or e.get("content") != m.get("content"):
                match = False
                break
        if match:
            return overlap
    return 0


def diff_incoming_messages(existing: List[dict], incoming: List[dict]) -> List[dict]:
    """Return only messages from incoming that are not already in existing.

    Logic matches Go memory/conversation.go DiffIncomingMessages.
    Uses sequence-based tail comparison.
    """
    incoming = normalize_conversation_messages(incoming)
    existing = normalize_conversation_messages(existing)

    if not existing:
        return incoming

    overlap = _overlap_length(existing, incoming)
    if overlap >= len(incoming):
        return []
    return incoming[overlap:]


def merge_history(history: List[dict], incoming: List[dict]) -> List[dict]:
    """Merge stored history with incoming messages, deduplicating overlap.

    Logic matches Go memory/conversation.go MergeHistory.
    """
    history = normalize_conversation_messages(history)
    incoming = normalize_conversation_messages(incoming)

    if not history:
        return incoming
    if not incoming:
        return history

    overlap = _overlap_length(history, incoming)
    merged = list(history)
    merged.extend(incoming[overlap:])
    return merged
