"""Recall layer: loads facts and summaries, scores via hybrid search."""

from __future__ import annotations

import logging
from typing import List, Optional

from .search import HybridSearchEngine
from .store import SQLiteStore
from .types import FactFilter, RecalledFact, RecalledSummary

logger = logging.getLogger("memg")


def recall_facts(
    engine: HybridSearchEngine,
    store: SQLiteStore,
    query_vec: List[float],
    query_text: str,
    entity_uuid: str,
    limit: int = 100,
    threshold: float = 0.10,
    max_candidates: int = 10000,
    fact_filter: Optional[FactFilter] = None,
) -> List[RecalledFact]:
    """Load candidate facts and rank them with hybrid search."""
    if not query_vec:
        return []

    filt = fact_filter or FactFilter(statuses=["current"], exclude_expired=True)
    if not filt.exclude_expired:
        filt.exclude_expired = True

    facts = store.list_facts_for_recall(entity_uuid, filt, max_candidates)
    if not facts:
        return []

    return engine.rank(query_vec, query_text, facts, limit, threshold)


def recall_summaries(
    engine: HybridSearchEngine,
    store: SQLiteStore,
    query_vec: List[float],
    query_text: str,
    entity_uuid: str,
    limit: int = 5,
    threshold: float = 0.10,
) -> List[RecalledSummary]:
    """Load conversation summaries and rank them with hybrid search."""
    if not query_vec:
        return []

    summaries = store.list_conversation_summaries(entity_uuid, limit=0)
    if not summaries:
        return []

    return engine.rank_summaries(query_vec, query_text, summaries, limit, threshold)
