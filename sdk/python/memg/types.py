from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class Memory:
    """A single memory (fact) stored for an entity."""

    id: str
    content: str
    type: str = "identity"
    temporal_status: str = "current"
    significance: str = "medium"
    created_at: Optional[datetime] = None
    tag: Optional[str] = None
    score: Optional[float] = None
    reinforced_count: int = 0


@dataclass
class AddResult:
    """Result of an add_memories call."""

    inserted: int
    reinforced: int


@dataclass
class SearchResult:
    """Result of a search or list call."""

    memories: List[Memory]
    count: int


@dataclass
class MemoryInput:
    """Input for adding a memory."""

    content: str
    type: str = "identity"
    significance: str = "medium"
    tag: Optional[str] = None


@dataclass
class Fact:
    """Internal representation of a stored fact with full metadata."""

    uuid: str
    content: str
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fact_type: str = "identity"
    temporal_status: str = "current"
    significance: int = 5
    content_key: str = ""
    reference_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    reinforced_at: Optional[datetime] = None
    reinforced_count: int = 0
    tag: str = ""
    slot: str = ""
    confidence: float = 1.0
    embedding_model: str = ""
    source_role: str = ""
    recall_count: int = 0
    last_recalled_at: Optional[datetime] = None


@dataclass
class Entity:
    """A tracked external identity."""

    uuid: str
    external_id: str
    created_at: Optional[datetime] = None


@dataclass
class Session:
    """A bounded interaction window for an entity."""

    uuid: str
    entity_id: str
    process_id: str = ""
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    entity_mentions: List[str] = field(default_factory=list)
    message_count: int = 0


@dataclass
class Conversation:
    """A sequence of messages within a session."""

    uuid: str
    session_id: str
    entity_id: str = ""
    summary: str = ""
    summary_embedding: Optional[List[float]] = None
    summary_embedding_model: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Message:
    """A single turn in a conversation."""

    uuid: str
    conversation_id: str
    role: str
    content: str
    kind: str = "text"
    created_at: Optional[datetime] = None


@dataclass
class FactFilter:
    """Constraints for filtered fact queries."""

    types: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_significance: int = 0
    max_significance: int = 0
    exclude_expired: bool = False
    reference_time_after: Optional[str] = None
    reference_time_before: Optional[str] = None
    slots: Optional[List[str]] = None
    min_confidence: float = 0.0
    source_roles: Optional[List[str]] = None
    unembedded_only: bool = False


@dataclass
class RecalledFact:
    """A stored fact that matched a recall query."""

    id: str
    content: str
    score: float
    temporal_status: str = "current"
    significance: int = 5
    created_at: Optional[datetime] = None


@dataclass
class ConsciousFact:
    """A high-significance fact loaded for conscious mode."""

    id: str
    content: str
    significance: int
    tag: str = ""


@dataclass
class TurnSummary:
    """A summary covering a range of turns in a conversation."""

    uuid: str
    conversation_id: str
    entity_id: str
    start_turn: int
    end_turn: int
    summary: str
    summary_embedding: Optional[List[float]] = None
    is_overview: bool = False
    created_at: Optional[datetime] = None


@dataclass
class Artifact:
    """A produced artifact (code, JSON, SQL, etc.) within a conversation."""

    uuid: str
    conversation_id: str
    entity_id: str
    content: str
    artifact_type: str = "code"
    language: str = ""
    description: str = ""
    description_embedding: Optional[List[float]] = None
    superseded_by: Optional[str] = None
    turn_number: int = 0
    created_at: Optional[datetime] = None


@dataclass
class RecalledSummary:
    """A past conversation summary that matched a recall query."""

    conversation_id: str
    summary: str
    score: float
    created_at: Optional[datetime] = None


@dataclass
class CanonicalSlot:
    """A globally shared slot name with its embedding vector."""

    name: str
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None


@dataclass
class Process:
    """A tracked process identity."""

    uuid: str
    external_id: str
    created_at: Optional[datetime] = None


@dataclass
class Attribute:
    """A key-value pair associated with a process."""

    uuid: str
    process_id: str
    key: str
    value: str
    created_at: Optional[datetime] = None
