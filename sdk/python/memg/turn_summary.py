"""Turn summary maintenance: generate immutable turn-range summaries and consolidate."""

from __future__ import annotations

import logging
from typing import Callable, List

from .types import TurnSummary

logger = logging.getLogger("memg")

_TURN_SUMMARY_PROMPT = (
    "Summarize these conversation turns. Focus on: decisions made, "
    "questions asked, code discussed, data referenced, open items. "
    "Keep under 200 tokens."
)

_OVERVIEW_CONSOLIDATE_PROMPT = (
    "Consolidate these summaries into one overview. Keep under 300 tokens."
)


def maintain_turn_summaries(
    store,
    embedder,
    llm_chat: Callable[[str], str],
    conversation_id: str,
    entity_id: str,
    messages: List[dict],
    working_memory_turns: int = 10,
) -> None:
    """Check if messages fell off the working memory window and generate
    immutable turn-range summaries. Consolidate when count exceeds 3.

    Args:
        store: Store instance with turn summary methods.
        embedder: Has embed(texts) -> List[List[float]].
        llm_chat: Callable(prompt) -> response text. Single string input.
        conversation_id: Conversation UUID.
        entity_id: Entity UUID.
        messages: List of message dicts with 'role' and 'content'.
        working_memory_turns: Number of recent turns to keep in working memory.
    """
    if len(messages) <= working_memory_turns:
        return

    existing = store.list_turn_summaries(conversation_id)

    highest_end = 0
    for ts in existing:
        if not ts.is_overview and ts.end_turn > highest_end:
            highest_end = ts.end_turn

    new_end = len(messages) - working_memory_turns
    if new_end <= highest_end:
        return

    to_summarize = messages[highest_end:new_end]
    if not to_summarize:
        return

    transcript = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in to_summarize
    )

    try:
        summary = llm_chat(_TURN_SUMMARY_PROMPT + "\n\n" + transcript)
    except Exception as e:
        logger.warning("memg turn_summary: llm call failed: %s", e)
        return

    summary = summary.strip()
    if not summary:
        return

    try:
        vectors = embedder.embed([summary])
    except Exception as e:
        logger.warning("memg turn_summary: embed failed: %s", e)
        return

    if not vectors:
        return

    ts = TurnSummary(
        uuid="",
        conversation_id=conversation_id,
        entity_id=entity_id,
        start_turn=highest_end + 1,
        end_turn=new_end,
        summary=summary,
        summary_embedding=vectors[0],
        is_overview=False,
    )
    store.insert_turn_summary(ts)

    _consolidate_turn_summaries(
        store, embedder, llm_chat, conversation_id, entity_id,
    )


def _consolidate_turn_summaries(
    store,
    embedder,
    llm_chat: Callable[[str], str],
    conversation_id: str,
    entity_id: str,
) -> None:
    """Consolidate old turn summaries when non-overview count exceeds 3."""
    all_summaries = store.list_turn_summaries(conversation_id)

    non_overview: List[TurnSummary] = []
    overview: TurnSummary = None
    for ts in all_summaries:
        if ts.is_overview:
            overview = ts
        else:
            non_overview.append(ts)

    if len(non_overview) <= 3:
        return

    non_overview.sort(key=lambda ts: ts.start_turn)
    oldest = non_overview[:2]

    combined_parts = []
    if overview is not None:
        combined_parts.append(overview.summary)
    for s in oldest:
        combined_parts.append(s.summary)
    combined = "\n".join(combined_parts)

    try:
        overview_text = llm_chat(_OVERVIEW_CONSOLIDATE_PROMPT + "\n\n" + combined)
    except Exception as e:
        logger.warning("memg turn_summary: consolidate llm failed: %s", e)
        return

    overview_text = overview_text.strip()
    if not overview_text:
        return

    try:
        vectors = embedder.embed([overview_text])
    except Exception as e:
        logger.warning("memg turn_summary: consolidate embed failed: %s", e)
        return

    if not vectors:
        return

    new_overview = TurnSummary(
        uuid="",
        conversation_id=conversation_id,
        entity_id=entity_id,
        start_turn=1,
        end_turn=oldest[-1].end_turn,
        summary=overview_text,
        summary_embedding=vectors[0],
        is_overview=True,
    )
    store.insert_turn_summary(new_overview)

    to_delete = []
    if overview is not None:
        to_delete.append(overview.uuid)
    for s in oldest:
        to_delete.append(s.uuid)
    store.delete_turn_summaries(conversation_id, to_delete)
