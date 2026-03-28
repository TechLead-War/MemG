"""RecallAndBuildContext: single entry point for the full memory recall pipeline.

Matches Go memory/recall_context.go.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .conscious import load_conscious_context
from .context import build_context
from .recall import recall_facts, recall_summaries
from .search import HybridSearchEngine

logger = logging.getLogger("memg")


@dataclass
class RecallConfig:
    recall_limit: int = 50
    recall_threshold: float = 0.05
    max_candidates: int = 50
    memory_token_budget: int = 4000
    summary_token_budget: int = 1000
    conscious_limit: int = 10
    summary_limit: int = 5
    conversation_id: str = ""


def recall_and_build_context(
    store,
    embedder,
    entity_uuid: str,
    query: str,
    cfg: Optional[RecallConfig] = None,
) -> str:
    """Single entry point for the full memory recall pipeline.

    1. Embeds the query (once, reused for all recall passes)
    2. Loads conscious facts (user profile, always present)
    3. Recalls relevant facts via hybrid search
    4. Recalls relevant conversation summaries
    5. Loads turn summaries for the active conversation
    6. Loads relevant artifacts
    7. Assembles everything via build_context with token budgeting

    Callers should not orchestrate these steps manually.
    """
    if cfg is None:
        cfg = RecallConfig()

    engine = HybridSearchEngine()

    # Step 1: Embed query once.
    try:
        vectors = embedder.embed([query])
    except Exception:
        logger.warning("memg recall_context: embed query failed", exc_info=True)
        return ""
    if not vectors or not vectors[0]:
        return ""
    query_vec = vectors[0]

    # Get embedding model name for mismatch detection.
    query_model = ""
    if hasattr(embedder, "model_name"):
        try:
            query_model = embedder.model_name()
        except Exception:
            query_model = ""

    # Step 2: Conscious facts.
    conscious_facts = load_conscious_context(store, entity_uuid, cfg.conscious_limit)

    # Step 3: Recalled facts.
    recalled_facts = recall_facts(
        engine, store, query_vec, query, entity_uuid,
        limit=cfg.recall_limit,
        threshold=cfg.recall_threshold,
        max_candidates=cfg.max_candidates,
        embedder=embedder,
        query_model=query_model,
    )

    # Step 4: Recalled summaries.
    recalled_summaries = recall_summaries(
        engine, store, query_vec, query, entity_uuid,
        limit=cfg.summary_limit,
        threshold=cfg.recall_threshold,
        query_model=query_model,
    )

    # Step 5: Turn summaries.
    turn_summaries = None
    if cfg.conversation_id and hasattr(store, "list_turn_summaries"):
        try:
            turn_summaries = store.list_turn_summaries(cfg.conversation_id)
        except Exception:
            logger.warning("memg recall_context: list_turn_summaries failed", exc_info=True)

    # Step 6: Artifacts.
    artifacts = None
    if cfg.conversation_id and hasattr(store, "list_active_artifacts"):
        try:
            artifacts = store.list_active_artifacts(entity_uuid, cfg.conversation_id)
        except Exception:
            logger.warning("memg recall_context: list_active_artifacts failed", exc_info=True)

    # Step 7: Build context.
    return build_context(
        conscious_facts=conscious_facts,
        recalled_facts=recalled_facts,
        summaries=recalled_summaries,
        turn_summaries=turn_summaries,
        artifacts=artifacts,
        total_token_budget=cfg.memory_token_budget,
        summary_token_budget=cfg.summary_token_budget,
    )
