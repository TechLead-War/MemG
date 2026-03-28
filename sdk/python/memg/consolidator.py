"""Consolidation: cluster old event facts into pattern facts."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable, List

from .extract import content_key
from .store import Store
from .types import Fact, FactFilter

logger = logging.getLogger("memg")


def consolidate_entity(
    store: Store,
    embedder,
    llm_chat: Callable[[str], str],
    entity_uuid: str,
) -> int:
    """Consolidate old event facts into pattern facts for a single entity.

    Logic matches Go memory/consolidator.go:
    1. Query event facts older than 30 days, current status, significance <= 5.
    2. Group by tag.
    3. For groups with 3+ facts, ask LLM to produce a behavioral pattern.
    4. Embed the pattern, insert as pattern fact with type="pattern", significance=5.
    5. Mark originals as historical.
    6. Return count of patterns created.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")

    filt = FactFilter(
        types=["event"],
        statuses=["current"],
        exclude_expired=True,
        reference_time_before=cutoff,
        max_significance=5,
    )

    facts = store.list_facts_filtered(entity_uuid, filt, limit=500)
    if len(facts) < 3:
        return 0

    by_tag: dict = defaultdict(list)
    for f in facts:
        tag = f.tag if f.tag else "_untagged"
        by_tag[tag].append(f)

    patterns_created = 0

    for tag, group in by_tag.items():
        if len(group) < 3:
            continue

        group.sort(key=lambda f: f.content)

        lines = []
        for f in group:
            lines.append(f"- {f.content}")
        events_text = "\n".join(lines)

        prompt = (
            f"Summarize these {len(group)} related events into a single behavioral pattern statement.\n"
            "The pattern should describe a recurring behavior or tendency.\n"
            "Return ONLY the pattern statement, nothing else.\n"
            "If these events don't form a meaningful pattern, respond with exactly: NONE\n\n"
            f"Events:\n{events_text}"
        )

        try:
            pattern = llm_chat(prompt)
        except Exception as e:
            logger.warning("memg consolidator: llm call for entity %s tag %r: %s", entity_uuid, tag, e)
            continue

        pattern = pattern.strip()
        if not pattern or pattern.upper() == "NONE":
            continue

        try:
            vectors = embedder.embed([pattern])
        except Exception as e:
            logger.warning("memg consolidator: embed pattern for entity %s tag %r: %s", entity_uuid, tag, e)
            continue

        if not vectors:
            continue

        pattern_fact = Fact(
            uuid="",
            content=pattern,
            embedding=vectors[0],
            fact_type="pattern",
            temporal_status="current",
            significance=5,
            tag=tag,
            content_key=content_key(pattern),
            embedding_model=embedder.model_name() if hasattr(embedder, "model_name") else "",
        )

        store.insert_fact(entity_uuid, pattern_fact)

        for f in group:
            store.update_temporal_status(f.uuid, "historical")

        logger.info(
            "memg consolidator: created pattern from %d events (entity=%s tag=%s): %s",
            len(group), entity_uuid, tag, pattern,
        )
        patterns_created += 1

    return patterns_created
