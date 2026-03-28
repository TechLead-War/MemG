package memg

import (
	"os"
	"strconv"
	"time"

	"memg/embed"
	"memg/llm"
)

// Config holds runtime settings for a MemG instance.
type Config struct {
	// EntityID ties stored facts to a specific external entity (e.g. a user).
	EntityID string
	// ProcessID ties stored attributes to a workflow or agent.
	ProcessID string
	// SessionID groups conversations under a resumable session.
	SessionID string

	// RecallFactsLimit caps the number of facts returned by a recall query.
	RecallFactsLimit int
	// RecallEmbeddingsLimit caps the number of stored embeddings loaded for
	// a single recall pass.
	RecallEmbeddingsLimit int
	// RecallThreshold is the minimum hybrid score a fact must reach to be
	// included in recall results.
	RecallThreshold float64

	// SessionTimeout controls how long a session stays active without new
	// messages before a fresh one is created.
	SessionTimeout time.Duration
	// RequestTimeout is the per-request timeout for outbound calls.
	RequestTimeout time.Duration
	// MaxRetries is the maximum number of retry attempts for transient errors.
	MaxRetries int
	// RetryBackoff is the base duration between retries (doubled each attempt).
	RetryBackoff time.Duration

	// EmbedDimension is the dimensionality of the embedding vectors.
	EmbedDimension int

	// PruneInterval controls how often the background pruner checks for and
	// removes expired facts. A zero value disables automatic pruning.
	PruneInterval time.Duration

	// LLMProvider selects an LLM provider by registry name (e.g. "openai", "anthropic").
	// The provider is auto-constructed during New() if SetProvider hasn't been called.
	LLMProvider string
	// LLMConfig holds connection settings for the LLM provider.
	LLMConfig llm.ProviderConfig

	// EmbedProvider selects an embedding provider by registry name (e.g. "openai", "ollama").
	// The embedder is auto-constructed during New() if SetEmbedder hasn't been called.
	EmbedProvider string
	// Embedder supplies pre-built embedding infrastructure. When set, it takes
	// precedence over EmbedProvider/EmbedConfig and is probed during New().
	Embedder embed.Embedder
	// EmbedConfig holds connection settings for the embedding provider.
	EmbedConfig embed.ProviderConfig

	// ConsciousMode enables startup context injection. When true, the top
	// facts by significance are loaded and injected into every request,
	// regardless of query relevance. This ensures the LLM always knows
	// critical user attributes (name, allergies, job, etc.) even when
	// the user's message is vague like "hey".
	ConsciousMode bool

	// ConsciousLimit caps the number of facts loaded for conscious mode.
	ConsciousLimit int

	// MaxRecallCandidates is the safety cap on how many facts are loaded
	// for a single recall pass. Prevents unbounded scans.
	MaxRecallCandidates int

	// WorkingMemoryTurns caps how many recent turns LoadRecentHistory returns.
	WorkingMemoryTurns int
	// MemoryTokenBudget is the max token budget for the merged context builder.
	MemoryTokenBudget int
	// SummaryTokenBudget is the max tokens for summary text in context.
	SummaryTokenBudget int

	// ConsciousCacheTTL controls how long conscious facts are cached before refresh.
	ConsciousCacheTTL time.Duration

	// Debug enables verbose logging when true.
	Debug bool
}

// DefaultConfig returns a Config populated with production-ready defaults.
// Environment variables prefixed with MEMG_ override the defaults.
func DefaultConfig() *Config {
	cfg := &Config{
		RecallFactsLimit:      100,
		RecallEmbeddingsLimit: 100_000,
		RecallThreshold:       0.10,
		SessionTimeout:        30 * time.Minute,
		RequestTimeout:        5 * time.Second,
		MaxRetries:            5,
		RetryBackoff:          time.Second,
		EmbedDimension:        384,
		PruneInterval:         5 * time.Minute,
		MaxRecallCandidates:   50,
		ConsciousMode:         true,
		ConsciousLimit:        10,
		ConsciousCacheTTL:     30 * time.Second,
		WorkingMemoryTurns:    10,
		MemoryTokenBudget:     4000,
		SummaryTokenBudget:    1000,
	}
	cfg.applyEnv()
	return cfg
}

func (c *Config) applyEnv() {
	if v := os.Getenv("MEMG_RECALL_FACTS_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.RecallFactsLimit = n
		}
	}
	if v := os.Getenv("MEMG_RECALL_EMBEDDINGS_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.RecallEmbeddingsLimit = n
		}
	}
	if v := os.Getenv("MEMG_MAX_RECALL_CANDIDATES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.MaxRecallCandidates = n
		}
	}
	if v := os.Getenv("MEMG_RECALL_THRESHOLD"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			c.RecallThreshold = f
		}
	}
	if v := os.Getenv("MEMG_SESSION_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			c.SessionTimeout = d
		}
	}
	if v := os.Getenv("MEMG_WORKING_MEMORY_TURNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.WorkingMemoryTurns = n
		}
	}
	if v := os.Getenv("MEMG_MEMORY_TOKEN_BUDGET"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.MemoryTokenBudget = n
		}
	}
	if v := os.Getenv("MEMG_SUMMARY_TOKEN_BUDGET"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.SummaryTokenBudget = n
		}
	}
	if v := os.Getenv("MEMG_CONSCIOUS_CACHE_TTL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			c.ConsciousCacheTTL = d
		}
	}
	if v := os.Getenv("MEMG_DEBUG"); v == "1" || v == "true" {
		c.Debug = true
	}
	if v := os.Getenv("MEMG_LLM_PROVIDER"); v != "" && c.LLMProvider == "" {
		c.LLMProvider = v
	}
	if v := os.Getenv("MEMG_EMBED_PROVIDER"); v != "" && c.EmbedProvider == "" {
		c.EmbedProvider = v
	}
}
