"""Decay and pruning: remove expired facts and stale summaries."""

from __future__ import annotations

import logging
from typing import Tuple

from .store import Store

logger = logging.getLogger("memg")


def prune_expired_and_stale(
    store: Store,
    summary_max_age_days: int = 90,
) -> Tuple[int, int]:
    """Prune expired facts and stale conversation summaries.

    Logic matches Go memory/decay.go:
    - Call store.prune_expired_facts with empty entity (all entities).
    - Call store.prune_stale_summaries.

    Returns (facts_pruned, summaries_cleared).
    """
    facts_pruned = 0
    try:
        facts_pruned = store.prune_expired_facts("")
    except Exception:
        logger.warning("memg pruner: prune expired facts failed", exc_info=True)

    summaries_cleared = 0
    try:
        summaries_cleared = store.prune_stale_summaries(summary_max_age_days)
    except Exception:
        logger.warning("memg pruner: prune stale summaries failed", exc_info=True)

    return facts_pruned, summaries_cleared
