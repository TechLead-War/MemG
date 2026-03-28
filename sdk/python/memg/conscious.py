"""Conscious context: load top facts by significance with staleness decay."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from .store import Store
from .types import ConsciousFact, FactFilter


def load_conscious_context(
    store: Store,
    entity_uuid: str,
    limit: int = 10,
) -> List[ConsciousFact]:
    """Load the top facts by significance for conscious mode.

    Logic matches Go memory/conscious.go:
    1. Fetch limit*5 current non-expired facts.
    2. Use list_facts_metadata if available, else list_facts_filtered.
    3. Score with staleness penalty for mutable identity facts:
       - Identity + significance < 10: apply decay after 30 days confirmed.
       - staleness = 0 if < 30 days.
       - staleness = (days-30)/90 if 30-120 days, capped at 0.5.
       - score = significance * (1 - staleness).
    4. Sort by score descending, return top limit as ConsciousFact.
    """
    if limit <= 0:
        limit = 10

    filt = FactFilter(statuses=["current"], exclude_expired=True)
    fetch_limit = max(50, limit * 5)

    if hasattr(store, "list_facts_metadata"):
        facts = store.list_facts_metadata(entity_uuid, filt, fetch_limit)
    else:
        facts = store.list_facts_filtered(entity_uuid, filt, fetch_limit)

    if not facts:
        return []

    now = datetime.now(timezone.utc)
    scored = []

    for f in facts:
        base = float(f.significance)

        if f.fact_type == "identity" and f.significance < 10:
            last_confirmed = f.created_at
            if f.reinforced_at and (last_confirmed is None or f.reinforced_at > last_confirmed):
                last_confirmed = f.reinforced_at
            if f.last_recalled_at and (last_confirmed is None or f.last_recalled_at > last_confirmed):
                last_confirmed = f.last_recalled_at
            if last_confirmed:
                days_since = (now - last_confirmed).total_seconds() / 86400
                if days_since > 30:
                    staleness = (days_since - 30) / 90
                    if staleness > 0.5:
                        staleness = 0.5
                    base *= (1 - staleness)

        scored.append((f, base))

    scored.sort(key=lambda x: (-x[1], x[0].uuid))

    if len(scored) > limit:
        scored = scored[:limit]

    return [
        ConsciousFact(
            id=f.uuid,
            content=f.content,
            significance=f.significance,
            tag=f.tag,
        )
        for f, _ in scored
    ]
