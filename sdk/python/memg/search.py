"""Hybrid vector + lexical search engine.

Port of the Go search package: cosine similarity, BM25, Kneedle cutoff.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import List, Optional

from .types import Fact, RecalledFact

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "like", "through", "after", "over", "between", "out",
    "against", "during", "without", "before", "under", "around", "among",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
}

_SPLIT_RE = re.compile(r"(?:[^\w]|_)+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    lower = text.lower()
    words = _SPLIT_RE.split(lower)
    return [w for w in words if w and w not in STOP_WORDS]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        fa, fb = a[i], b[i]
        dot += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


def _vector_scores(query: List[float], candidates: List[_Candidate]) -> List[float]:
    scores = []
    for c in candidates:
        if c.embedding is not None and len(c.embedding) == len(query):
            scores.append(cosine_similarity(query, c.embedding))
        else:
            scores.append(0.0)
    return scores


def _bm25_scores(query_text: str, candidates: List[_Candidate]) -> List[float]:
    terms = tokenize(query_text)
    n_cands = len(candidates)
    if not terms or n_cands == 0:
        return [0.0] * n_cands

    docs = [tokenize(c.content) for c in candidates]
    doc_sets = [set(d) for d in docs]
    avg_len = sum(len(d) for d in docs) / len(docs) if docs else 1.0

    n = float(len(docs))
    idf = {}
    for t in terms:
        df = sum(1 for ds in doc_sets if t in ds)
        idf[t] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    k1 = 1.2
    b = 0.75

    raw = []
    peak = 0.0
    for doc in docs:
        dl = float(len(doc))
        tf = {}
        for w in doc:
            if w in idf:
                tf[w] = tf.get(w, 0.0) + 1.0
        score = 0.0
        for t in terms:
            f = tf.get(t, 0.0)
            num = f * (k1 + 1)
            denom = f + k1 * (1 - b + b * (dl / avg_len))
            score += idf[t] * num / denom
        raw.append(score)
        if score > peak:
            peak = score

    if peak > 0:
        return [r / peak for r in raw]
    return [0.0] * n_cands


def _blend_weights(query_text: str):
    tokens = tokenize(query_text)
    if len(tokens) <= 2:
        return 0.70, 0.30
    return 0.85, 0.15


@dataclass
class _Candidate:
    idx: int
    id: str
    content: str
    embedding: Optional[List[float]]
    created_at: object
    temporal_status: str
    significance: int
    confidence: float


@dataclass
class _Scored:
    idx: int
    score: float


def kneedle_cutoff(items: List[_Scored]) -> int:
    """Find the knee using the Kneedle algorithm (Satopaa et al., ICDCS 2011).

    Items must be sorted by score descending. Normalizes both axes to [0,1],
    draws a diagonal from the first to the last point, and finds where the
    curve deviates most below the diagonal. Returns number of items to keep.
    """
    if len(items) <= 3:
        return len(items)

    n = len(items)
    max_score = items[0].score
    min_score = items[n - 1].score
    score_range = max_score - min_score

    if score_range < 1e-9:
        return n

    best_deviation = 0.0
    knee_idx = n

    for i in range(n):
        x = i / (n - 1)
        y = (items[i].score - min_score) / score_range
        diagonal = 1.0 - x
        deviation = diagonal - y

        if deviation > best_deviation:
            best_deviation = deviation
            knee_idx = i

    if knee_idx <= 0:
        return 1
    if knee_idx >= n:
        return n
    return knee_idx


class HybridSearchEngine:
    """Hybrid dense + lexical search engine matching the Go implementation."""

    def rank(
        self,
        query_vec: List[float],
        query_text: str,
        facts: List[Fact],
        limit: int = 100,
        threshold: float = 0.10,
        query_model: str = "",
    ) -> List[RecalledFact]:
        if not facts:
            return []

        candidates = []
        mismatch_count = 0
        for i, f in enumerate(facts):
            confidence = f.confidence if f.confidence > 0 else 1.0
            embedding = f.embedding
            if embedding is not None and len(query_vec) > 0 and len(embedding) != len(query_vec):
                embedding = None
                mismatch_count += 1
            elif query_model and getattr(f, "embedding_model", "") and f.embedding_model != query_model:
                embedding = None
                mismatch_count += 1
            candidates.append(_Candidate(
                idx=i,
                id=f.uuid,
                content=f.content,
                embedding=embedding,
                created_at=f.created_at,
                temporal_status=f.temporal_status or "current",
                significance=f.significance if f.significance is not None else 0,
                confidence=confidence,
            ))

        if mismatch_count > 0:
            logger.warning(
                "memg recall: skipped %d facts with embedding dimension mismatch (query=%d)",
                mismatch_count, len(query_vec),
            )

        dense = _vector_scores(query_vec, candidates)
        lexical = _bm25_scores(query_text, candidates)
        w_dense, w_lex = _blend_weights(query_text)

        merged = []
        for i, c in enumerate(candidates):
            score = w_dense * dense[i] + w_lex * lexical[i]

            if c.temporal_status == "historical":
                score *= 0.85

            if 0 < c.confidence < 1.0:
                score *= (0.95 + 0.05 * c.confidence)

            merged.append(_Scored(idx=i, score=score))

        merged.sort(key=lambda s: s.score, reverse=True)

        above = []
        for s in merged:
            if s.score < threshold:
                break
            above.append(s)
            if len(above) >= limit:
                break

        for s in above:
            c = candidates[s.idx]
            s.score += c.significance * 0.001

        above.sort(key=lambda s: s.score, reverse=True)

        cutoff = kneedle_cutoff(above)

        results = []
        for i in range(cutoff):
            c = candidates[above[i].idx]
            results.append(RecalledFact(
                id=c.id,
                content=c.content,
                score=above[i].score,
                temporal_status=c.temporal_status,
                significance=c.significance,
                created_at=c.created_at,
            ))
        return results

    def rank_summaries(
        self,
        query_vec: List[float],
        query_text: str,
        summaries: list,
        limit: int = 5,
        threshold: float = 0.10,
        query_model: str = "",
    ) -> list:
        """Rank conversation summaries using hybrid search.

        summaries: list of Conversation objects with summary and summary_embedding.
        Returns list of RecalledSummary-like dicts.
        """
        from .types import RecalledSummary

        if not summaries:
            return []

        candidates = []
        for i, s in enumerate(summaries):
            embedding = s.summary_embedding
            if embedding is not None and len(query_vec) > 0 and len(embedding) != len(query_vec):
                embedding = None
            elif query_model and getattr(s, "summary_embedding_model", "") and s.summary_embedding_model != query_model:
                embedding = None
            candidates.append(_Candidate(
                idx=i,
                id=s.uuid,
                content=s.summary,
                embedding=embedding,
                created_at=s.created_at,
                temporal_status="current",
                significance=0,
                confidence=1.0,
            ))

        dense = _vector_scores(query_vec, candidates)
        lexical = _bm25_scores(query_text, candidates)
        w_dense, w_lex = _blend_weights(query_text)

        merged = []
        for i, c in enumerate(candidates):
            score = w_dense * dense[i] + w_lex * lexical[i]
            score += c.significance * 0.001
            merged.append(_Scored(idx=i, score=score))

        merged.sort(key=lambda s: s.score, reverse=True)

        above = []
        for s in merged:
            if s.score < threshold:
                break
            above.append(s)
            if len(above) >= limit:
                break

        cutoff = kneedle_cutoff(above)

        results = []
        for i in range(cutoff):
            c = candidates[above[i].idx]
            original = summaries[c.idx]
            results.append(RecalledSummary(
                conversation_id=c.id,
                summary=c.content,
                score=above[i].score,
                created_at=c.created_at,
            ))
        return results
