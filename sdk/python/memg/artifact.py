"""Artifact storage and recall: persist detected artifacts, retrieve relevant ones."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from .artifact_detect import DetectedArtifact
from .search import HybridSearchEngine, cosine_similarity
from .types import Artifact, Fact

logger = logging.getLogger("memg")


def store_artifacts(
    store,
    embedder,
    llm_chat: Callable[[str], str],
    detected: List[DetectedArtifact],
    existing: List[Artifact],
    conversation_id: str,
    entity_id: str,
    turn_number: int,
) -> None:
    """Persist detected artifacts with descriptions and superseding.

    Args:
        store: Store instance with insert_artifact/supersede_artifact methods.
        embedder: Has embed(texts) -> List[List[float]] and model_name() -> str.
        llm_chat: Callable that takes a prompt string and returns response text.
        detected: List of DetectedArtifact from detect_artifacts().
        existing: List of currently active Artifact objects.
        conversation_id: Conversation UUID.
        entity_id: Entity UUID.
        turn_number: Current turn number.
    """
    if not detected:
        return

    descriptions = []
    for d in detected:
        content = d.content
        if len(content) > 500:
            content = content[:500]
        desc = llm_chat("Describe this code/data in one sentence: " + content)
        descriptions.append(desc.strip())

    vectors = embedder.embed(descriptions)
    if len(vectors) != len(descriptions):
        logger.error(
            "memg artifact: expected %d vectors, got %d",
            len(descriptions), len(vectors),
        )
        return

    for i, d in enumerate(detected):
        new_vec = vectors[i]

        a = Artifact(
            uuid="",
            conversation_id=conversation_id,
            entity_id=entity_id,
            content=d.content,
            artifact_type=d.artifact_type,
            language=d.language,
            description=descriptions[i],
            description_embedding=new_vec,
            turn_number=turn_number,
        )
        store.insert_artifact(a)

        for ex in existing:
            if ex.superseded_by:
                continue
            if not ex.description_embedding:
                continue
            if len(new_vec) != len(ex.description_embedding):
                continue
            if cosine_similarity(new_vec, ex.description_embedding) > 0.8:
                try:
                    store.supersede_artifact(ex.uuid, a.uuid)
                except Exception as e:
                    logger.warning("memg artifact: supersede %s: %s", ex.uuid, e)


def recall_artifacts(
    store,
    engine: HybridSearchEngine,
    query_vec: List[float],
    query_text: str,
    entity_id: str,
    conversation_id: str,
    limit: int = 5,
    threshold: float = 0.10,
) -> List[Artifact]:
    """Retrieve relevant artifacts using hybrid search.

    Args:
        store: Store instance with list_active_artifacts methods.
        engine: HybridSearchEngine for ranking.
        query_vec: Query embedding vector.
        query_text: Query text for BM25.
        entity_id: Entity UUID.
        conversation_id: Conversation UUID.
        limit: Max artifacts to return.
        threshold: Score threshold.

    Returns:
        List of matching Artifact objects with full content.
    """
    if conversation_id:
        artifacts = store.list_active_artifacts(entity_id, conversation_id)
    else:
        artifacts = store.list_active_artifacts_by_entity(entity_id)

    if not artifacts:
        return []

    facts = []
    for a in artifacts:
        facts.append(Fact(
            uuid=a.uuid,
            content=a.description,
            embedding=a.description_embedding,
            created_at=a.created_at,
        ))

    results = engine.rank(query_vec, query_text, facts, limit, threshold)

    by_uuid = {a.uuid: a for a in artifacts}
    out = []
    for r in results:
        artifact = by_uuid.get(r.id)
        if artifact is not None:
            out.append(artifact)
    return out
