"""Hybrid vector + lexical search engine.

Port of the Go search package: cosine similarity, BM25, Kneedle cutoff.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional

from .types import Fact, RecalledFact

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "him", "her", "his", "its", "do", "did", "does",
    "have", "has", "had", "what", "which", "who", "whom", "when", "where",
    "how", "can", "could", "would", "should", "may", "might", "shall",
    "must", "am", "been", "being", "were", "so", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "any",
    "because", "before", "between", "both", "each", "few", "from",
    "further", "here", "more", "most", "nor", "only", "other", "out",
    "over", "own", "same", "some", "through", "under", "until", "up",
    "while", "why", "down", "during", "off", "once", "like", "against",
    "without", "around", "among", "yet", "either", "neither", "every",
    "those", "them", "need",
}

_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


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
    avg_len = sum(len(d) for d in docs) / len(docs) if docs else 1.0

    n = float(len(docs))
    idf = {}
    for t in terms:
        df = sum(1 for doc in docs if t in doc)
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
    n = len(items)
    if n <= 2:
        return n

    y_min = items[-1].score
    y_max = items[0].score
    y_range = y_max - y_min
    if y_range == 0:
        return n

    max_diff = 0.0
    knee_idx = 0
    for i in range(n):
        x_norm = i / (n - 1)
        y_norm = (items[i].score - y_min) / y_range
        diagonal = 1.0 - x_norm
        diff = y_norm - diagonal
        if diff > max_diff:
            max_diff = diff
            knee_idx = i

    if max_diff < 0.10:
        return n

    return max(1, knee_idx + 1)


class HybridSearchEngine:
    """Hybrid dense + lexical search engine matching the Go implementation."""

    def rank(
        self,
        query_vec: List[float],
        query_text: str,
        facts: List[Fact],
        limit: int = 100,
        threshold: float = 0.10,
    ) -> List[RecalledFact]:
        if not facts:
            return []

        candidates = []
        for i, f in enumerate(facts):
            confidence = f.confidence if f.confidence > 0 else 1.0
            embedding = f.embedding
            if embedding is not None and len(query_vec) > 0 and len(embedding) != len(query_vec):
                embedding = None
            candidates.append(_Candidate(
                idx=i,
                id=f.uuid,
                content=f.content,
                embedding=embedding,
                created_at=f.created_at,
                temporal_status=f.temporal_status or "current",
                significance=f.significance if f.significance else 5,
                confidence=confidence,
            ))

        dense = _vector_scores(query_vec, candidates)
        lexical = _bm25_scores(query_text, candidates)
        w_dense, w_lex = _blend_weights(query_text)

        merged = []
        for i, c in enumerate(candidates):
            score = w_dense * dense[i] + w_lex * lexical[i]

            if c.temporal_status == "historical":
                score *= 0.85

            score += c.significance * 0.001

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
            candidates.append(_Candidate(
                idx=i,
                id=s.uuid,
                content=s.summary,
                embedding=embedding,
                created_at=s.created_at,
                temporal_status="current",
                significance=5,
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
