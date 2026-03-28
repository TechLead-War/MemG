package memory

import (
	"context"
	"log"

	"memg/embed"
	"memg/search"
	"memg/store"
)

// RecallConfig controls the behavior of RecallAndBuildContext.
type RecallConfig struct {
	RecallLimit        int     // max facts recalled per query (default: 50)
	RecallThreshold    float64 // minimum score for recalled facts (default: 0.05)
	MaxCandidates      int     // max candidate facts loaded from DB (default: 50)
	MemoryTokenBudget  int     // total token budget for context (default: 4000)
	SummaryTokenBudget int     // sub-budget for summaries (default: 1000)
	ConsciousLimit     int     // max conscious profile facts (default: 10)
	SummaryLimit       int     // max summaries recalled (default: 5)
	ConversationID     string  // active conversation for turn summaries (optional)
}

func (c *RecallConfig) withDefaults() RecallConfig {
	out := *c
	if out.RecallLimit <= 0 {
		out.RecallLimit = 50
	}
	if out.RecallThreshold <= 0 {
		out.RecallThreshold = 0.05
	}
	if out.MaxCandidates <= 0 {
		out.MaxCandidates = 50
	}
	if out.MemoryTokenBudget <= 0 {
		out.MemoryTokenBudget = 4000
	}
	if out.SummaryTokenBudget <= 0 {
		out.SummaryTokenBudget = 1000
	}
	if out.ConsciousLimit <= 0 {
		out.ConsciousLimit = 10
	}
	if out.SummaryLimit <= 0 {
		out.SummaryLimit = 5
	}
	return out
}

// RecallAndBuildContext is the single entry point for the full memory recall
// pipeline. It performs all steps internally:
//
//  1. Embeds the query (once, reused for all recall passes)
//  2. Loads conscious facts (user profile, always present)
//  3. Recalls relevant facts via hybrid search
//  4. Recalls relevant conversation summaries
//  5. Loads turn summaries for the active conversation (if ConversationID set)
//  6. Recalls relevant artifacts (if ConversationID set)
//  7. Assembles everything via BuildContext with token budgeting
//
// Callers should not orchestrate these steps manually — this function
// encapsulates all memory recall decisions so improvements to the pipeline
// automatically propagate to every caller.
func RecallAndBuildContext(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	engine search.Engine,
	entityUUID string,
	query string,
	cfg RecallConfig,
) (string, error) {
	cfg = cfg.withDefaults()

	// Ensure backfill embedder is available for RecallWithVector's backfill path.
	if getRecallEmbedder() == nil && embedder != nil {
		SetRecallEmbedder(embedder)
	}

	// Step 1: Embed query once.
	vectors, err := embedder.Embed(ctx, []string{query})
	if err != nil {
		return "", err
	}
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return "", nil
	}
	queryVec := vectors[0]
	queryModel := embed.ModelNameOf(embedder)

	// Step 2: Conscious facts.
	consciousFacts, err := LoadConsciousContext(ctx, repo, entityUUID, cfg.ConsciousLimit)
	if err != nil {
		log.Printf("memg recall: conscious facts: %v", err)
	}

	// Step 3: Recalled facts.
	recalledFacts, err := RecallWithVector(
		ctx, engine, repo,
		queryVec, queryModel, query, entityUUID,
		cfg.RecallLimit, cfg.RecallThreshold, cfg.MaxCandidates,
	)
	if err != nil {
		log.Printf("memg recall: recalled facts: %v", err)
	}

	// Step 4: Recalled summaries.
	recalledSummaries, err := RecallSummariesWithVector(
		ctx, engine, repo,
		queryVec, queryModel, query, entityUUID,
		cfg.SummaryLimit, cfg.RecallThreshold,
	)
	if err != nil {
		log.Printf("memg recall: recalled summaries: %v", err)
	}

	// Step 5: Turn summaries (if active conversation).
	var turnSummaries []*store.TurnSummary
	if cfg.ConversationID != "" {
		if tsr, ok := repo.(store.TurnSummaryReader); ok {
			turnSummaries, err = tsr.ListTurnSummaries(ctx, cfg.ConversationID)
			if err != nil {
				log.Printf("memg recall: turn summaries: %v", err)
			}
		}
	}

	// Step 6: Artifacts (if active conversation).
	var artifacts []*store.Artifact
	if cfg.ConversationID != "" {
		if ar, ok := repo.(store.ArtifactReader); ok {
			artifacts, err = ar.ListActiveArtifacts(ctx, entityUUID, cfg.ConversationID)
			if err != nil {
				log.Printf("memg recall: artifacts: %v", err)
			}
		}
	}

	// Step 7: Build context.
	contextStr := BuildContext(ContextInput{
		ConsciousFacts: consciousFacts,
		RecalledFacts:  recalledFacts,
		Summaries:      recalledSummaries,
		TurnSummaries:  turnSummaries,
		Artifacts:      artifacts,
		Budget: ContextBudget{
			TotalTokens:   cfg.MemoryTokenBudget,
			SummaryTokens: cfg.SummaryTokenBudget,
		},
	})

	return contextStr, nil
}
