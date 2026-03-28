// Package memory implements recall, conversation tracking, and knowledge
// extraction for the MemG system.
package memory

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"memg/embed"
	"memg/search"
	"memg/store"
)

// RecallWithVector is like Recall but accepts a pre-computed query vector
// instead of embedding the query text. This avoids redundant embedding calls
// when the same query is used for both fact and summary recall.
//
// If an embedder is provided via RecallOpts, facts with NULL embeddings are
// re-embedded in the background so they are available on the next recall.
func RecallWithVector(
	ctx context.Context,
	engine search.Engine,
	repo FactSource,
	queryVec []float32,
	queryModel string,
	queryText string,
	entityUUID string,
	limit int,
	threshold float64,
	maxCandidates int,
	filters ...store.FactFilter,
) ([]*RecalledFact, error) {
	if len(queryVec) == 0 {
		return nil, nil
	}

	filter := store.FactFilter{
		ExcludeExpired: true,
	}
	if len(filters) > 0 {
		filter = filters[0]
		filter.ExcludeExpired = true
	}

	if maxCandidates <= 0 {
		maxCandidates = 50
	}
	facts, err := listRecallFacts(ctx, repo, entityUUID, filter, maxCandidates)
	if err != nil {
		return nil, err
	}
	if len(facts) == 0 {
		return nil, nil
	}

	// Detect facts with NULL embeddings for background backfill.
	hasUnembedded := false
	for _, f := range facts {
		if len(f.Embedding) == 0 {
			hasUnembedded = true
			break
		}
	}

	candidates := buildRecallCandidates(queryVec, queryModel, facts)
	results := engine.Rank(queryVec, queryText, candidates, limit, threshold)

	out := make([]*RecalledFact, len(results))
	for i, r := range results {
		out[i] = &RecalledFact{
			ID:             r.ID,
			Content:        r.Content,
			Score:          r.Score,
			CreatedAt:      r.CreatedAt,
			TemporalStatus: r.TemporalStatus,
			Significance:   r.Significance,
		}
	}

	// Trigger background backfill for facts stored during embedding outages.
	// Use atomic flag to prevent spawning multiple concurrent backfill goroutines.
	if hasUnembedded {
		if fullRepo, ok := repo.(store.Repository); ok {
			if emb := getRecallEmbedder(); emb != nil {
				if backfillRunning.CompareAndSwap(false, true) {
					go func() {
						defer backfillRunning.Store(false)
						bgCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
						defer cancel()
						n := BackfillMissingEmbeddings(bgCtx, fullRepo, emb, entityUUID, 50)
						if n > 0 {
							log.Printf("memg recall: backfilled %d facts with missing embeddings", n)
						}
					}()
				}
			}
		}
	}

	return out, nil
}

// recallEmbedder is set by SetRecallEmbedder and used for background
// backfill of facts with NULL embeddings during recall.
var (
	recallEmbedder   embed.Embedder
	recallEmbedderMu sync.RWMutex
	backfillRunning  atomic.Bool
)

func getRecallEmbedder() embed.Embedder {
	recallEmbedderMu.RLock()
	defer recallEmbedderMu.RUnlock()
	return recallEmbedder
}

// SetRecallEmbedder configures the embedder used for background backfill
// during recall. Call this at startup when initializing the MemG instance.
func SetRecallEmbedder(e embed.Embedder) {
	recallEmbedderMu.Lock()
	defer recallEmbedderMu.Unlock()
	recallEmbedder = e
}

// RecalledFact is a stored fact that matched a recall query.
type RecalledFact struct {
	ID             string
	Content        string
	Score          float64
	CreatedAt      time.Time
	TemporalStatus string
	Significance   int
}

// FactSource provides facts for recall queries. Both store.FactReader
// and store.FactFilteredReader satisfy this through the combined Repository.
type FactSource interface {
	store.FactReader
	store.FactFilteredReader
}

// Recall loads facts for the given entity, embeds the query, and returns
// the top matches according to the hybrid search engine. An optional filter
// can narrow the candidate set before ranking (e.g. by type, tag, or time range).
func Recall(
	ctx context.Context,
	embedder embed.Embedder,
	engine search.Engine,
	repo FactSource,
	query string,
	entityUUID string,
	limit int,
	threshold float64,
	maxCandidates int,
	filters ...store.FactFilter,
) ([]*RecalledFact, error) {
	// Ensure backfill embedder is available for background re-embedding.
	if getRecallEmbedder() == nil && embedder != nil {
		SetRecallEmbedder(embedder)
	}
	vectors, err := embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, nil
	}

	return RecallWithVector(
		ctx,
		engine,
		repo,
		vectors[0],
		embed.ModelNameOf(embedder),
		query,
		entityUUID,
		limit,
		threshold,
		maxCandidates,
		filters...,
	)
}

func listRecallFacts(
	ctx context.Context,
	repo FactSource,
	entityUUID string,
	filter store.FactFilter,
	limit int,
) ([]*store.Fact, error) {
	if rr, ok := repo.(store.FactRecallReader); ok {
		return rr.ListFactsForRecall(ctx, entityUUID, filter, limit)
	}
	return repo.ListFactsFiltered(ctx, entityUUID, filter, limit)
}

func buildRecallCandidates(queryVec []float32, queryModel string, facts []*store.Fact) []search.Candidate {
	candidates := make([]search.Candidate, 0, len(facts))
	var dimensionFallbackCount int
	var modelFallbackCount int

	for _, f := range facts {
		confidence := f.Confidence
		if confidence == 0 {
			confidence = 1.0
		}

		embedding := f.Embedding
		if len(queryVec) > 0 && !search.DimensionMatch(queryVec, embedding) {
			embedding = nil
			dimensionFallbackCount++
		} else if queryModel != "" && f.EmbeddingModel != "" && f.EmbeddingModel != queryModel {
			embedding = nil
			modelFallbackCount++
		}

		candidates = append(candidates, search.Candidate{
			ID:             f.UUID,
			Content:        f.Content,
			Embedding:      embedding,
			CreatedAt:      f.CreatedAt,
			TemporalStatus: string(f.TemporalStatus),
			Significance:   int(f.Significance),
			Confidence:     confidence,
		})
	}

	if dimensionFallbackCount > 0 {
		log.Printf("memg recall: using lexical-only fallback for %d facts with incompatible embedding dimensions", dimensionFallbackCount)
	}
	if modelFallbackCount > 0 {
		log.Printf("memg recall: using lexical-only fallback for %d facts with embedding model mismatch (%s)", modelFallbackCount, queryModel)
	}

	return candidates
}
