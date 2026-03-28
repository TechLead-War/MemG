package memory

import (
	"context"
	"fmt"
	"log"
	"strings"

	"memg/embed"
	"memg/llm"
	"memg/search"
	"memg/store"
)

// StoreArtifacts persists detected artifacts, generating descriptions and
// checking for superseding of existing artifacts.
func StoreArtifacts(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	provider llm.Provider,
	detected []DetectedArtifact,
	existing []*store.Artifact,
	conversationID, entityID string,
	turnNumber int,
) error {
	if len(detected) == 0 {
		return nil
	}

	descriptions := make([]string, len(detected))
	for i, d := range detected {
		content := d.Content
		if len(content) > 500 {
			content = content[:500]
		}
		req := &llm.Request{
			Messages:  []*llm.Message{llm.UserMessage("Describe this code/data in one sentence: " + content)},
			MaxTokens: 60,
		}
		resp, err := provider.Chat(ctx, req)
		if err != nil {
			return fmt.Errorf("artifact description: %w", err)
		}
		descriptions[i] = strings.TrimSpace(resp.Content)
	}

	vectors, err := embedder.Embed(ctx, descriptions)
	if err != nil {
		return fmt.Errorf("artifact embed: %w", err)
	}
	if len(vectors) != len(descriptions) {
		return fmt.Errorf("artifact embed: expected %d vectors, got %d", len(descriptions), len(vectors))
	}

	for i, d := range detected {
		newVec := vectors[i]

		a := &store.Artifact{
			ConversationID:       conversationID,
			EntityID:             entityID,
			Content:              d.Content,
			ArtifactType:         d.ArtifactType,
			Language:             d.Language,
			Description:          descriptions[i],
			DescriptionEmbedding: newVec,
			TurnNumber:           turnNumber,
		}
		if err := repo.InsertArtifact(ctx, a); err != nil {
			return fmt.Errorf("artifact insert: %w", err)
		}

		for _, ex := range existing {
			if ex.SupersededBy != "" {
				continue
			}
			if len(ex.DescriptionEmbedding) == 0 || !search.DimensionMatch(newVec, ex.DescriptionEmbedding) {
				continue
			}
			if search.CosineSimilarity(newVec, ex.DescriptionEmbedding) > 0.8 {
				if err := repo.SupersedeArtifact(ctx, ex.UUID, a.UUID); err != nil {
					log.Printf("memg artifact: supersede %s: %v", ex.UUID, err)
				}
			}
		}
	}

	return nil
}

// RecallArtifacts retrieves relevant artifacts using hybrid search.
func RecallArtifacts(
	ctx context.Context,
	engine search.Engine,
	repo store.Repository,
	queryVec []float32,
	queryText string,
	entityID string,
	conversationID string,
	limit int,
	threshold float64,
) ([]*store.Artifact, error) {
	var artifacts []*store.Artifact
	var err error
	if conversationID != "" {
		artifacts, err = repo.ListActiveArtifacts(ctx, entityID, conversationID)
	} else {
		artifacts, err = repo.ListActiveArtifactsByEntity(ctx, entityID)
	}
	if err != nil {
		return nil, fmt.Errorf("artifact recall: list: %w", err)
	}
	if len(artifacts) == 0 {
		return nil, nil
	}

	candidates := make([]search.Candidate, 0, len(artifacts))
	for _, a := range artifacts {
		candidates = append(candidates, search.Candidate{
			ID:        a.UUID,
			Content:   a.Description,
			Embedding: a.DescriptionEmbedding,
			CreatedAt: a.CreatedAt,
		})
	}

	results := engine.Rank(queryVec, queryText, candidates, limit, threshold)

	byUUID := make(map[string]*store.Artifact, len(artifacts))
	for _, a := range artifacts {
		byUUID[a.UUID] = a
	}

	out := make([]*store.Artifact, 0, len(results))
	for _, r := range results {
		if a, ok := byUUID[r.ID]; ok {
			out = append(out, a)
		}
	}
	return out, nil
}
