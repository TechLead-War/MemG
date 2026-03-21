// Package store defines the persistence contracts for MemG.
// Concrete implementations live in sub-packages (sqlstore, mongodb, etc.).
package store

import (
	"context"
	"io"
	"time"
)

// EntityWriter creates or updates entity records.
type EntityWriter interface {
	UpsertEntity(ctx context.Context, externalID string) (uuid string, err error)
}

// EntityReader retrieves entity records.
type EntityReader interface {
	LookupEntity(ctx context.Context, externalID string) (*Entity, error)
}

// EntityLister lists entity UUIDs for administrative operations such as
// background consolidation that must iterate over all known entities.
type EntityLister interface {
	ListEntityUUIDs(ctx context.Context, limit int) ([]string, error)
}

// FactWriter persists knowledge facts for an entity.
type FactWriter interface {
	InsertFact(ctx context.Context, entityUUID string, fact *Fact) error
	InsertFacts(ctx context.Context, entityUUID string, facts []*Fact) error
}

// FactReader loads facts belonging to an entity.
type FactReader interface {
	ListFacts(ctx context.Context, entityUUID string, limit int) ([]*Fact, error)
}

// FactFilter constrains which facts are returned by filtered queries.
type FactFilter struct {
	Types              []FactType       // nil = all types
	Statuses           []TemporalStatus // nil = all statuses
	Tags               []string         // nil = all tags; filter by category label
	MinSignificance    Significance     // 0 = no minimum
	ExcludeExpired     bool             // true = skip facts past ExpiresAt
	ReferenceTimeAfter  *time.Time      // nil = no lower bound on reference_time
	ReferenceTimeBefore *time.Time      // nil = no upper bound on reference_time
	Slots              []string         // nil = all slots
	MinConfidence      float64          // 0 = no minimum
	SourceRoles        []string         // nil = all roles; e.g. ["user"]
}

// FactManager provides lifecycle operations for facts beyond simple CRUD.
type FactManager interface {
	// FindFactByKey returns a fact matching the given entity and content key.
	// Returns nil, nil if no match is found.
	FindFactByKey(ctx context.Context, entityUUID, contentKey string) (*Fact, error)

	// UpdateTemporalStatus changes a fact's temporal status (e.g. current → historical).
	UpdateTemporalStatus(ctx context.Context, factUUID string, status TemporalStatus) error

	// ReinforceFact bumps the reinforced_at timestamp, increments reinforced_count,
	// and optionally resets the TTL (ExpiresAt). Pass nil to leave ExpiresAt unchanged.
	ReinforceFact(ctx context.Context, factUUID string, newExpiresAt *time.Time) error

	// PruneExpiredFacts deletes facts whose ExpiresAt is before now.
	// Returns the number of facts removed.
	PruneExpiredFacts(ctx context.Context, entityUUID string, now time.Time) (int64, error)

	// DeleteFact removes a single fact by UUID, scoped to the given entity.
	DeleteFact(ctx context.Context, entityUUID, factUUID string) error

	// DeleteEntityFacts removes all facts for the given entity.
	// Returns the number of facts removed.
	DeleteEntityFacts(ctx context.Context, entityUUID string) (int64, error)

	// UpdateSignificance changes a fact's significance level.
	UpdateSignificance(ctx context.Context, factUUID string, sig Significance) error

	// UpdateFactEmbedding replaces a fact's embedding vector and model name.
	UpdateFactEmbedding(ctx context.Context, factUUID string, embedding []float32, model string) error
}

// FactFilteredReader loads facts with metadata-based filtering.
type FactFilteredReader interface {
	// ListFactsFiltered is like ListFacts but applies a FactFilter.
	ListFactsFiltered(ctx context.Context, entityUUID string, filter FactFilter, limit int) ([]*Fact, error)
}

// FactRecallReader provides a recall-optimized read path that does not
// bias results by creation time.
type FactRecallReader interface {
	// ListFactsForRecall loads facts matching the filter up to limit,
	// without ordering by creation time. This prevents newest-first bias
	// in recall queries.
	ListFactsForRecall(ctx context.Context, entityUUID string, filter FactFilter, limit int) ([]*Fact, error)
}

// FactMetadataReader loads fact metadata without decoding embeddings.
// This is cheaper for queries that only need content and metadata.
type FactMetadataReader interface {
	// ListFactsMetadata is like ListFactsFiltered but does not load embedding data.
	ListFactsMetadata(ctx context.Context, entityUUID string, filter FactFilter, limit int) ([]*Fact, error)
}

// RecallUsageTracker tracks fact recall usage for long-horizon hygiene.
type RecallUsageTracker interface {
	// UpdateRecallUsage increments recall_count and sets last_recalled_at
	// for the given fact UUIDs. Called when facts are injected into a prompt.
	UpdateRecallUsage(ctx context.Context, factUUIDs []string) error
}

// CanonicalSlotStore manages the shared canonical slot registry for slot
// normalization. This is an optional interface — repositories that do not
// implement it will silently skip slot normalization in the pipeline.
type CanonicalSlotStore interface {
	ListCanonicalSlots(ctx context.Context) ([]*CanonicalSlot, error)
	InsertCanonicalSlot(ctx context.Context, slot *CanonicalSlot) error
	FindCanonicalSlotByName(ctx context.Context, name string) (*CanonicalSlot, error)
}

// ConversationWriter manages conversation records.
type ConversationWriter interface {
	StartConversation(ctx context.Context, sessionUUID, entityUUID string) (uuid string, err error)
	// UpdateConversationSummary stores a summary and its embedding for a conversation.
	UpdateConversationSummary(ctx context.Context, conversationUUID, summary string, embedding []float32) error
}

// ConversationPruner manages conversation lifecycle cleanup.
type ConversationPruner interface {
	// PruneStaleSummaries removes summary text and embeddings from conversations
	// older than the given cutoff, keeping the conversation record intact.
	// Returns the number of summaries cleared.
	PruneStaleSummaries(ctx context.Context, olderThan time.Time) (int64, error)
}

// ConversationReader queries conversation state.
type ConversationReader interface {
	FindConversation(ctx context.Context, uuid string) (*Conversation, error)
	ActiveConversation(ctx context.Context, sessionUUID string) (*Conversation, error)
	// ListConversationSummaries loads conversations with non-empty summaries for an entity.
	// Returns only conversations that have both a summary and an embedding.
	ListConversationSummaries(ctx context.Context, entityUUID string, limit int) ([]*Conversation, error)
	// FindUnsummarizedConversation returns the most recent conversation for
	// the entity that has no summary yet, excluding the given session.
	FindUnsummarizedConversation(ctx context.Context, entityUUID, excludeSessionUUID string) (*Conversation, error)
}

// MessageWriter appends messages to a conversation log.
type MessageWriter interface {
	AppendMessage(ctx context.Context, conversationUUID string, msg *Message) error
}

// MessageReader retrieves messages from a conversation log.
type MessageReader interface {
	ReadMessages(ctx context.Context, conversationUUID string) ([]*Message, error)
	// ReadRecentMessages retrieves the last N messages from a conversation,
	// returned in chronological order. This avoids loading all messages when
	// only the tail is needed.
	ReadRecentMessages(ctx context.Context, conversationUUID string, limit int) ([]*Message, error)
}

// SessionWriter manages session lifecycle.
type SessionWriter interface {
	EnsureSession(ctx context.Context, entityUUID, processUUID string, timeout time.Duration) (*Session, bool, error)
}

// ProcessWriter creates or updates process records.
type ProcessWriter interface {
	UpsertProcess(ctx context.Context, externalID string) (uuid string, err error)
}

// ProcessAttributeWriter persists attributes for a process.
type ProcessAttributeWriter interface {
	InsertProcessAttribute(ctx context.Context, processUUID string, attr *Attribute) error
}

// Migrator handles schema creation and version upgrades.
type Migrator interface {
	Migrate(ctx context.Context) error
	SchemaVersion(ctx context.Context) (int, error)
}

// Repository is the full persistence contract that backing stores must satisfy.
type Repository interface {
	EntityWriter
	EntityReader
	EntityLister
	FactWriter
	FactReader
	FactManager
	FactFilteredReader
	FactRecallReader
	FactMetadataReader
	RecallUsageTracker
	ConversationWriter
	ConversationReader
	ConversationPruner
	MessageWriter
	MessageReader
	SessionWriter
	ProcessWriter
	ProcessAttributeWriter
	Migrator
	io.Closer
}
