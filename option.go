package memg

import (
	"time"

	"memg/embed"
	"memg/llm"
)

// Option configures a MemG instance at construction time.
type Option func(*Config)

// WithEntity binds all stored facts to the given external entity identifier.
func WithEntity(id string) Option {
	return func(c *Config) { c.EntityID = id }
}

// WithProcess binds stored attributes to the given workflow identifier.
func WithProcess(id string) Option {
	return func(c *Config) { c.ProcessID = id }
}

// WithSession assigns a session identifier for conversation grouping.
func WithSession(id string) Option {
	return func(c *Config) { c.SessionID = id }
}

// WithRecallLimit sets the maximum number of facts returned per recall query.
func WithRecallLimit(n int) Option {
	return func(c *Config) { c.RecallFactsLimit = n }
}

// WithRecallThreshold sets the minimum score for a fact to be recalled.
func WithRecallThreshold(t float64) Option {
	return func(c *Config) { c.RecallThreshold = t }
}

// WithMaxRecallCandidates sets the safety cap on facts loaded per recall pass.
func WithMaxRecallCandidates(n int) Option {
	return func(c *Config) { c.MaxRecallCandidates = n }
}

// WithSessionTimeout sets how long a session remains active without activity.
func WithSessionTimeout(d time.Duration) Option {
	return func(c *Config) { c.SessionTimeout = d }
}

// WithEmbedDimension overrides the default embedding vector dimensionality.
func WithEmbedDimension(dim int) Option {
	return func(c *Config) { c.EmbedDimension = dim }
}

// WithPruneInterval sets how often the background pruner checks for and
// removes expired facts. A zero value disables automatic pruning.
func WithPruneInterval(d time.Duration) Option {
	return func(c *Config) { c.PruneInterval = d }
}

// WithLLMProvider selects an LLM provider by registry name and configures it.
// The provider is auto-constructed during New(). Requires the provider package
// to be imported (e.g. _ "memg/llm/openai").
func WithLLMProvider(name string, cfg llm.ProviderConfig) Option {
	return func(c *Config) {
		c.LLMProvider = name
		c.LLMConfig = cfg
	}
}

// WithEmbedProvider selects an embedding provider by registry name and configures it.
// The embedder is auto-constructed during New(). Requires the provider package
// to be imported (e.g. _ "memg/embed/openai").
func WithEmbedProvider(name string, cfg embed.ProviderConfig) Option {
	return func(c *Config) {
		c.EmbedProvider = name
		c.EmbedConfig = cfg
	}
}

// WithConsciousMode enables or disables startup context injection.
// When enabled, the top facts by significance are always injected,
// ensuring the LLM knows critical user attributes even for vague messages.
func WithConsciousMode(enabled bool) Option {
	return func(c *Config) { c.ConsciousMode = enabled }
}

// WithConsciousLimit sets the maximum number of facts loaded for conscious mode.
func WithConsciousLimit(n int) Option {
	return func(c *Config) { c.ConsciousLimit = n }
}

// WithWorkingMemoryTurns sets the maximum number of recent turns loaded for working memory.
func WithWorkingMemoryTurns(n int) Option {
	return func(c *Config) { c.WorkingMemoryTurns = n }
}

// WithMemoryTokenBudget sets the total token budget for the merged context builder.
func WithMemoryTokenBudget(n int) Option {
	return func(c *Config) { c.MemoryTokenBudget = n }
}

// WithSummaryTokenBudget sets the max tokens allocated to summaries in context.
func WithSummaryTokenBudget(n int) Option {
	return func(c *Config) { c.SummaryTokenBudget = n }
}

// WithDebug enables verbose diagnostic logging.
func WithDebug() Option {
	return func(c *Config) { c.Debug = true }
}
