"""Query transformation for retrieval-optimized queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class QueryTransform:
    """Result of transforming a raw chat query into a retrieval-optimized form.

    Matches Go memory/query.go QueryTransform.
    """

    rewritten_query: str = ""


class QueryTransformer(ABC):
    """Rewrites follow-up chat queries into standalone retrieval queries.

    Matches Go memory/query.go QueryTransformer interface.
    """

    @abstractmethod
    def transform_query(self, query: str, recent_history: List[str]) -> QueryTransform: ...
