package bench

import (
	"context"

	"memg/embed"
	"memg/memory"
	"memg/search"
	"memg/store"
)

// recallAndBuildContext delegates entirely to the library's single entry point.
// No orchestration logic here — the library owns all recall decisions.
func recallAndBuildContext(
	ctx context.Context,
	embedder embed.Embedder,
	repo store.Repository,
	entityUUID string,
	question string,
	limit int,
	threshold float64,
	maxCandidates int,
	memoryTokenBudget int,
	summaryTokenBudget int,
	consciousLimit int,
) (string, error) {
	return memory.RecallAndBuildContext(
		ctx, repo, embedder, search.NewHybrid(),
		entityUUID, question,
		memory.RecallConfig{
			RecallLimit:        limit,
			RecallThreshold:    threshold,
			MaxCandidates:      maxCandidates,
			MemoryTokenBudget:  memoryTokenBudget,
			SummaryTokenBudget: summaryTokenBudget,
			ConsciousLimit:     consciousLimit,
		},
	)
}
