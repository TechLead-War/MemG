package store

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"time"
	"unicode"
)

// TTLForSignificance returns the expiry time for a fact based on its
// significance level. High significance facts never expire (nil).
func TTLForSignificance(sig Significance) *time.Time {
	var d time.Duration
	switch {
	case sig >= SignificanceHigh:
		return nil // never expires
	case sig >= SignificanceMedium:
		d = 30 * 24 * time.Hour // ~1 month
	default:
		d = 7 * 24 * time.Hour // ~1 week
	}
	t := time.Now().UTC().Round(0).Add(d)
	return &t
}

// FactType classifies the category of knowledge a fact represents.
type FactType string

const (
	// FactTypeIdentity represents enduring attributes: "User is vegetarian".
	FactTypeIdentity FactType = "identity"
	// FactTypeEvent represents point-in-time occurrences: "User adopted a dog on March 5".
	FactTypeEvent FactType = "event"
	// FactTypePattern represents observed behavioral tendencies: "User prefers concise answers".
	FactTypePattern FactType = "pattern"
)

// TemporalStatus indicates whether a fact is currently believed to be true.
type TemporalStatus string

const (
	// TemporalCurrent means the fact is presently valid.
	TemporalCurrent TemporalStatus = "current"
	// TemporalHistorical means the fact was once true but has been superseded.
	TemporalHistorical TemporalStatus = "historical"
)

// Significance represents how important a fact is for determining its lifespan.
// Higher values result in longer TTLs. The value does not affect recall ranking.
type Significance int

const (
	SignificanceLow    Significance = 1
	SignificanceMedium Significance = 5
	SignificanceHigh   Significance = 10
)

// Entity represents a tracked external identity (e.g. a user or service).
type Entity struct {
	UUID       string
	ExternalID string
	CreatedAt  time.Time
}

// Fact is a single unit of knowledge tied to an entity.
type Fact struct {
	UUID      string
	Content   string
	Embedding []float32
	CreatedAt time.Time
	UpdatedAt time.Time

	// Type classifies what kind of knowledge this fact represents.
	// Zero value is treated as FactTypeIdentity by the pipeline.
	Type FactType

	// TemporalStatus indicates whether this fact is current or historical.
	// Zero value is treated as TemporalCurrent by the pipeline.
	TemporalStatus TemporalStatus

	// Significance controls how long the fact survives without reinforcement.
	// Zero value is treated as SignificanceMedium by the pipeline.
	Significance Significance

	// ContentKey is a normalized hash used for deduplication. If empty, the
	// pipeline computes one from Content using DefaultContentKey.
	ContentKey string

	// ReferenceTime is when the fact's content refers to (e.g. the date the
	// user ate pasta). Nil means unspecified / not time-bound.
	ReferenceTime *time.Time

	// ExpiresAt is the TTL expiry. Facts past this time are pruned.
	// Nil means the fact never expires.
	ExpiresAt *time.Time

	// ReinforcedAt records when this fact was last reinforced by a duplicate
	// extraction. Nil means the fact has never been reinforced.
	ReinforcedAt *time.Time

	// ReinforcedCount tracks how many times this fact has been re-encountered.
	ReinforcedCount int

	// Tag is an optional freeform category label for organization and filtering.
	// Examples: "skill", "preference", "relationship", "medical", "location".
	// Does not affect lifecycle behavior — that's controlled by Type.
	Tag string

	// Slot identifies the semantic slot this fact occupies (e.g. "location",
	// "job", "diet"). Mutable slots allow conflict detection to replace
	// prior values instead of accumulating duplicates.
	Slot string

	// Confidence is the extraction confidence score (0.0-1.0).
	// Lower confidence facts from assistant-originated guesses can be
	// filtered or deprioritized.
	Confidence float64

	// EmbeddingModel records which embedding model produced the vector,
	// allowing mismatch detection when the model changes.
	EmbeddingModel string

	// SourceRole records whether the fact came from user or assistant
	// utterances. Assistant-sourced facts are treated with lower trust.
	SourceRole string

	// RecallCount tracks how many times this fact has been injected into
	// a prompt. Facts that are never recalled may be candidates for consolidation.
	RecallCount int

	// LastRecalledAt records when this fact was last included in a prompt.
	// Nil means the fact has never been recalled.
	LastRecalledAt *time.Time
}

// DefaultContentKey computes a content key by lowercasing, stripping
// punctuation, collapsing whitespace, and hashing the result.
func DefaultContentKey(content string) string {
	lower := strings.ToLower(content)
	cleaned := strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsSpace(r) {
			return r
		}
		return -1
	}, lower)
	normalized := strings.Join(strings.Fields(cleaned), " ")
	h := sha256.Sum256([]byte(normalized))
	return fmt.Sprintf("%x", h[:8])
}

// CanonicalSlot is a globally shared slot name with its embedding vector,
// used to normalize free-form slot strings from extraction stages.
type CanonicalSlot struct {
	Name      string
	Embedding []float32
	CreatedAt time.Time
}

// Conversation groups a sequence of messages within a session.
type Conversation struct {
	UUID                  string
	SessionID             string
	EntityID              string
	Summary               string
	SummaryEmbedding      []float32
	SummaryEmbeddingModel string
	CreatedAt             time.Time
	UpdatedAt             time.Time
}

// Message is a single turn in a conversation.
type Message struct {
	UUID           string
	ConversationID string
	Role           string
	Content        string
	Kind           string
	CreatedAt      time.Time
}

// TurnSummary is a condensed summary covering a range of conversation turns.
type TurnSummary struct {
	UUID             string
	ConversationID   string
	EntityID         string
	StartTurn        int
	EndTurn          int
	Summary          string
	SummaryEmbedding []float32
	IsOverview       bool
	CreatedAt        time.Time
}

// Artifact is a structured output (code, JSON, SQL, etc.) produced during a conversation.
type Artifact struct {
	UUID                 string
	ConversationID       string
	EntityID             string
	Content              string
	ArtifactType         string
	Language             string
	Description          string
	DescriptionEmbedding []float32
	SupersededBy         string
	TurnNumber           int
	CreatedAt            time.Time
}

// Session represents a bounded interaction window for an entity.
type Session struct {
	UUID           string
	EntityID       string
	ProcessID      string
	CreatedAt      time.Time
	ExpiresAt      time.Time
	EntityMentions []string
	MessageCount   int
}

// Process represents a tracked workflow or agent.
type Process struct {
	UUID       string
	ExternalID string
	CreatedAt  time.Time
}

// Attribute is a key-value pair associated with a process.
type Attribute struct {
	UUID      string
	ProcessID string
	Key       string
	Value     string
	CreatedAt time.Time
}
