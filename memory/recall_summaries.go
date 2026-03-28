package memory

import (
	"context"
	"log"
	"time"

	"memg/embed"
	"memg/search"
	"memg/store"
)

// RecallSummariesWithVector is like RecallSummaries but accepts a pre-computed
// query vector to avoid redundant embedding calls.
func RecallSummariesWithVector(
	ctx context.Context,
	engine search.Engine,
	repo store.ConversationReader,
	queryVec []float32,
	queryModel string,
	queryText string,
	entityUUID string,
	limit int,
	threshold float64,
) ([]*RecalledSummary, error) {
	if len(queryVec) == 0 {
		return nil, nil
	}

	convs, err := repo.ListConversationSummaries(ctx, entityUUID, 100)
	if err != nil {
		return nil, err
	}
	if len(convs) == 0 {
		return nil, nil
	}

	candidates := make([]search.Candidate, 0, len(convs))
	var dimensionFallbackCount int
	var modelFallbackCount int
	for _, c := range convs {
		embedding := c.SummaryEmbedding
		if len(queryVec) > 0 && !search.DimensionMatch(queryVec, embedding) {
			embedding = nil
			dimensionFallbackCount++
		} else if queryModel != "" && c.SummaryEmbeddingModel != "" && c.SummaryEmbeddingModel != queryModel {
			embedding = nil
			modelFallbackCount++
		}
		candidates = append(candidates, search.Candidate{
			ID:        c.UUID,
			Content:   c.Summary,
			Embedding: embedding,
			CreatedAt: c.CreatedAt,
		})
	}
	if dimensionFallbackCount > 0 {
		log.Printf("memg recall summaries: using lexical-only fallback for %d summaries with incompatible embedding dimensions", dimensionFallbackCount)
	}
	if modelFallbackCount > 0 {
		log.Printf("memg recall summaries: using lexical-only fallback for %d summaries with embedding model mismatch (%s)", modelFallbackCount, queryModel)
	}

	results := engine.Rank(queryVec, queryText, candidates, limit, threshold)

	out := make([]*RecalledSummary, len(results))
	for i, r := range results {
		out[i] = &RecalledSummary{
			ConversationID: r.ID,
			Summary:        r.Content,
			Score:          r.Score,
			CreatedAt:      r.CreatedAt,
		}
	}
	return out, nil
}

// RecalledSummary is a past conversation summary that matched a recall query.
type RecalledSummary struct {
	ConversationID string
	Summary        string
	Score          float64
	CreatedAt      time.Time
}

// RecallSummaries searches past conversation summaries by semantic relevance
// and returns the top matches. This is a separate recall layer from fact recall —
// summaries capture conversation narratives, not atomic knowledge.
func RecallSummaries(
	ctx context.Context,
	embedder embed.Embedder,
	engine search.Engine,
	repo store.ConversationReader,
	query string,
	entityUUID string,
	limit int,
	threshold float64,
) ([]*RecalledSummary, error) {
	vectors, err := embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, nil
	}
	queryVec := vectors[0]

	// Load all conversation summaries for unbiased ranking.
	convs, err := repo.ListConversationSummaries(ctx, entityUUID, 100)
	if err != nil {
		return nil, err
	}
	if len(convs) == 0 {
		return nil, nil
	}

	// Build candidates from conversation summaries.
	queryModel := embed.ModelNameOf(embedder)
	candidates := make([]search.Candidate, 0, len(convs))
	var dimensionFallbackCount int
	var modelFallbackCount int
	for _, c := range convs {
		embedding := c.SummaryEmbedding
		if len(queryVec) > 0 && !search.DimensionMatch(queryVec, embedding) {
			embedding = nil
			dimensionFallbackCount++
		} else if queryModel != "" && c.SummaryEmbeddingModel != "" && c.SummaryEmbeddingModel != queryModel {
			embedding = nil
			modelFallbackCount++
		}
		candidates = append(candidates, search.Candidate{
			ID:        c.UUID,
			Content:   c.Summary,
			Embedding: embedding,
			CreatedAt: c.CreatedAt,
		})
	}
	if dimensionFallbackCount > 0 {
		log.Printf("memg recall summaries: using lexical-only fallback for %d summaries with incompatible embedding dimensions", dimensionFallbackCount)
	}
	if modelFallbackCount > 0 {
		log.Printf("memg recall summaries: using lexical-only fallback for %d summaries with embedding model mismatch (%s)", modelFallbackCount, queryModel)
	}

	results := engine.Rank(queryVec, query, candidates, limit, threshold)

	out := make([]*RecalledSummary, len(results))
	for i, r := range results {
		out[i] = &RecalledSummary{
			ConversationID: r.ID,
			Summary:        r.Content,
			Score:          r.Score,
			CreatedAt:      r.CreatedAt,
		}
	}
	return out, nil
}
