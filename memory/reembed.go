package memory

import (
	"context"
	"fmt"
	"log"

	"memg/embed"
	"memg/store"
)

// BackfillMissingEmbeddings re-embeds facts with NULL embeddings.
func BackfillMissingEmbeddings(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	entityUUID string,
	limit int,
) int {
	if limit <= 0 {
		limit = 50
	}

	filter := store.FactFilter{UnembeddedOnly: true, ExcludeExpired: true}
	facts, err := repo.ListFactsFiltered(ctx, entityUUID, filter, limit)
	if err != nil || len(facts) == 0 {
		return 0
	}

	contents := make([]string, len(facts))
	for i, f := range facts {
		contents[i] = f.Content
	}

	vectors, err := embedder.Embed(ctx, contents)
	if err != nil {
		return 0
	}

	modelName := embed.ModelNameOf(embedder)
	updated := 0
	for i, f := range facts {
		if i < len(vectors) {
			if err := repo.UpdateFactEmbedding(ctx, f.UUID, vectors[i], modelName); err != nil {
				log.Printf("memg: backfill embed %s: %v", f.UUID, err)
				continue
			}
			updated++
		}
	}
	return updated
}

// ReEmbedFacts re-embeds all facts for the given entity using the provided
// embedder. This is needed when the embedding model changes — old facts
// have incompatible vectors that produce zero similarity during recall.
// It processes facts in batches and updates each fact's embedding and
// embedding_model in place.
func ReEmbedFacts(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	entityUUID string,
	modelName string,
	batchSize int,
) (int, error) {
	if batchSize <= 0 {
		batchSize = 50
	}

	// Load all facts (we need their content to re-embed).
	filter := store.FactFilter{ExcludeExpired: true}
	facts, err := repo.ListFactsFiltered(ctx, entityUUID, filter, 100000)
	if err != nil {
		return 0, fmt.Errorf("reembed: load facts: %w", err)
	}

	updated := 0
	for i := 0; i < len(facts); i += batchSize {
		end := i + batchSize
		if end > len(facts) {
			end = len(facts)
		}
		batch := facts[i:end]

		// Collect content for batch embedding.
		contents := make([]string, len(batch))
		for j, f := range batch {
			contents[j] = f.Content
		}

		// Embed the batch.
		vectors, err := embedder.Embed(ctx, contents)
		if err != nil {
			return updated, fmt.Errorf("reembed: embed batch at %d: %w", i, err)
		}

		// Update each fact's embedding.
		for j, f := range batch {
			if j < len(vectors) {
				f.Embedding = vectors[j]
				f.EmbeddingModel = modelName
				if err := repo.UpdateFactEmbedding(ctx, f.UUID, vectors[j], modelName); err != nil {
					log.Printf("reembed: update fact %s: %v", f.UUID, err)
					continue
				}
				updated++
			}
		}
	}

	return updated, nil
}
