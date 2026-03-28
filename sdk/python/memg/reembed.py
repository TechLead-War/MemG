"""Re-embedding: backfill missing embeddings and re-embed with new models."""

from __future__ import annotations

import logging
from typing import List

from .store import Store
from .types import FactFilter

logger = logging.getLogger("memg")


def backfill_missing_embeddings(
    store: Store,
    embedder,
    entity_uuid: str,
    limit: int = 50,
) -> int:
    """Re-embed facts with NULL embeddings.

    Logic matches Go memory/reembed.go BackfillMissingEmbeddings.
    """
    if limit <= 0:
        limit = 50

    filt = FactFilter(unembedded_only=True, exclude_expired=True)
    facts = store.list_facts_filtered(entity_uuid, filt, limit)
    if not facts:
        return 0

    contents = [f.content for f in facts]

    try:
        vectors = embedder.embed(contents)
    except Exception:
        return 0

    model_name = embedder.model_name() if hasattr(embedder, "model_name") else ""
    updated = 0
    for i, f in enumerate(facts):
        if i < len(vectors):
            try:
                store.update_fact_embedding(f.uuid, vectors[i], model_name)
                updated += 1
            except Exception as e:
                logger.warning("memg: backfill embed %s: %s", f.uuid, e)

    return updated


def re_embed_facts(
    store: Store,
    embedder,
    entity_uuid: str,
    model_name: str,
    batch_size: int = 50,
) -> int:
    """Re-embed all facts for the given entity using the provided embedder.

    Logic matches Go memory/reembed.go ReEmbedFacts. Processes facts in
    batches and updates each fact's embedding and embedding_model in place.
    """
    if batch_size <= 0:
        batch_size = 50

    filt = FactFilter(exclude_expired=True)
    facts = store.list_facts_filtered(entity_uuid, filt, limit=100000)
    if not facts:
        return 0

    updated = 0
    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]

        contents = [f.content for f in batch]

        try:
            vectors = embedder.embed(contents)
        except Exception as e:
            logger.warning("memg reembed: embed batch at %d: %s", i, e)
            break

        for j, f in enumerate(batch):
            if j < len(vectors):
                try:
                    store.update_fact_embedding(f.uuid, vectors[j], model_name)
                    updated += 1
                except Exception as e:
                    logger.warning("memg reembed: update fact %s: %s", f.uuid, e)

    return updated
