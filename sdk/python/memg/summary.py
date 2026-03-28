"""Conversation summary generation and storage."""

from __future__ import annotations

import logging
from typing import Callable

from .store import Store

logger = logging.getLogger("memg")

SUMMARY_PROMPT = (
    "Summarize this conversation. Focus on:\n"
    "- What was discussed\n"
    "- What decisions were made\n"
    "- What is still pending or unresolved\n"
    "- Any new information learned about the user\n\n"
    "Be concise — 2-5 sentences. Only include what is meaningful and worth remembering.\n"
    "If the conversation contains no meaningful content worth remembering "
    "(e.g. just greetings or trivial exchanges), respond with exactly: NONE"
)


def generate_and_store_summary(
    store: Store,
    embedder,
    llm_chat: Callable[[str], str],
    conversation_uuid: str,
) -> None:
    """Generate a summary of the conversation and store it with its embedding.

    Logic matches Go memory/summary.go GenerateAndStoreSummary:
    1. Read messages, build transcript.
    2. LLM call with summary prompt.
    3. Skip if "NONE" or empty.
    4. Embed, store via update_conversation_summary.
    """
    messages = store.read_messages(conversation_uuid)
    if not messages:
        return

    transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)

    try:
        summary = llm_chat(SUMMARY_PROMPT + "\n\n" + transcript)
    except Exception as e:
        logger.warning("memg summary: llm call failed: %s", e)
        return

    summary = summary.strip()
    if not summary or summary.upper() == "NONE":
        return

    embedding = None
    if embedder:
        try:
            vecs = embedder.embed([summary])
            if vecs:
                embedding = vecs[0]
        except Exception:
            pass

    model_name = ""
    if embedder and hasattr(embedder, "model_name"):
        try:
            model_name = embedder.model_name()
        except Exception:
            model_name = ""
    store.update_conversation_summary(conversation_uuid, summary, embedding, embedding_model=model_name)
