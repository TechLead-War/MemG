"""Recall layer: loads facts and summaries, scores via hybrid search."""

from __future__ import annotations

import copy
import logging
import threading
from typing import List, Optional

from .search import HybridSearchEngine
from .store import Store
from .types import FactFilter, RecalledFact, RecalledSummary

logger = logging.getLogger("memg")

_backfill_active: set = set()
_backfill_lock = threading.Lock()


def recall_facts(
    engine: HybridSearchEngine,
    store: Store,
    query_vec: List[float],
    query_text: str,
    entity_uuid: str,
    limit: int = 100,
    threshold: float = 0.10,
    max_candidates: int = 50,
    fact_filter: Optional[FactFilter] = None,
    embedder=None,
    query_model: str = "",
) -> List[RecalledFact]:
    """Load candidate facts and rank them with hybrid search."""
    if not query_vec:
        return []

    filt = copy.copy(fact_filter) if fact_filter else FactFilter(exclude_expired=True)
    filt.exclude_expired = True

    facts = store.list_facts_for_recall(entity_uuid, filt, max_candidates)
    if not facts:
        return []

    # Trigger background backfill for facts with NULL embeddings.
    has_unembedded = any(f.embedding is None or len(f.embedding) == 0 for f in facts)
    if has_unembedded and embedder is not None and hasattr(store, "list_unembedded_facts"):
        with _backfill_lock:
            if entity_uuid in _backfill_active:
                has_unembedded = False
            else:
                _backfill_active.add(entity_uuid)
        if has_unembedded:
            threading.Thread(
                target=_backfill_embeddings,
                args=(store, embedder, entity_uuid),
                daemon=True,
            ).start()

    return engine.rank(query_vec, query_text, facts, limit, threshold, query_model=query_model)


def _backfill_embeddings(store, embedder, entity_uuid: str) -> None:
    try:
        unembedded = store.list_unembedded_facts(entity_uuid, limit=50)
        if not unembedded:
            return
        contents = [f.content for f in unembedded]
        vectors = embedder.embed(contents)
        model_name = getattr(embedder, "model_name", lambda: "unknown")()
        for i, f in enumerate(unembedded):
            if i < len(vectors):
                store.update_fact_embedding(f.uuid, vectors[i], model_name)
        logger.info("memg recall: backfilled %d facts with missing embeddings", min(len(vectors), len(unembedded)))
    except Exception:
        logger.warning("memg recall: background backfill failed", exc_info=True)
    finally:
        with _backfill_lock:
            _backfill_active.discard(entity_uuid)


def recall_summaries(
    engine: HybridSearchEngine,
    store: Store,
    query_vec: List[float],
    query_text: str,
    entity_uuid: str,
    limit: int = 5,
    threshold: float = 0.10,
    query_model: str = "",
) -> List[RecalledSummary]:
    """Load conversation summaries and rank them with hybrid search."""
    if not query_vec:
        return []

    summaries = store.list_conversation_summaries(entity_uuid, limit=100)
    if not summaries:
        return []

    return engine.rank_summaries(query_vec, query_text, summaries, limit, threshold, query_model=query_model)
