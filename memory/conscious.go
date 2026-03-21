package memory

import (
	"context"
	"sort"
	"time"

	"memg/store"
)

// ConsciousFact is a high-significance fact loaded for conscious mode.
type ConsciousFact struct {
	ID           string
	Content      string
	Significance int
	Tag          string
}

// LoadConsciousContext returns the top facts by significance for the given
// entity. These are the user's most important attributes — the facts that
// should always be present regardless of query relevance.
//
// When the repository implements store.FactMetadataReader, the lighter
// metadata-only query is used (skips embedding decoding).
func LoadConsciousContext(
	ctx context.Context,
	repo store.FactFilteredReader,
	entityUUID string,
	limit int,
) ([]*ConsciousFact, error) {
	if limit <= 0 {
		limit = 10
	}

	// Load current, non-expired facts. Over-fetch since we'll sort by significance.
	filter := store.FactFilter{
		Statuses:       []store.TemporalStatus{store.TemporalCurrent},
		ExcludeExpired: true,
	}
	fetchLimit := limit * 5
	if fetchLimit < 50 {
		fetchLimit = 50
	}

	// Try metadata-only read if available (avoids decoding embeddings).
	var facts []*store.Fact
	var err error
	if mr, ok := repo.(store.FactMetadataReader); ok {
		facts, err = mr.ListFactsMetadata(ctx, entityUUID, filter, fetchLimit)
	} else {
		facts, err = repo.ListFactsFiltered(ctx, entityUUID, filter, fetchLimit)
	}
	if err != nil {
		return nil, err
	}
	if len(facts) == 0 {
		return nil, nil
	}

	// Score facts for conscious ranking. Immutable facts (events, high-significance
	// identity with strong reinforcement) keep their rank. Mutable identity facts
	// that haven't been reinforced or recalled recently are demoted.
	type scored struct {
		fact  *store.Fact
		score float64
	}
	scoredFacts := make([]scored, len(facts))
	now := time.Now()

	for i, f := range facts {
		base := float64(f.Significance)

		// Demote stale mutable facts: low/medium significance identity facts
		// that haven't been confirmed (reinforced) recently.
		if f.Type == store.FactTypeIdentity && f.Significance < store.SignificanceHigh {
			staleness := 0.0
			lastConfirmed := f.CreatedAt
			if f.ReinforcedAt != nil && f.ReinforcedAt.After(lastConfirmed) {
				lastConfirmed = *f.ReinforcedAt
			}
			if f.LastRecalledAt != nil && f.LastRecalledAt.After(lastConfirmed) {
				lastConfirmed = *f.LastRecalledAt
			}
			daysSinceConfirmed := now.Sub(lastConfirmed).Hours() / 24
			if daysSinceConfirmed > 30 {
				staleness = (daysSinceConfirmed - 30) / 90 // linear decay after 30 days
				if staleness > 0.5 {
					staleness = 0.5 // cap at 50% demotion
				}
			}
			base *= (1 - staleness)
		}

		scoredFacts[i] = scored{fact: f, score: base}
	}

	// Sort by score descending, UUID for stability.
	sort.Slice(scoredFacts, func(i, j int) bool {
		if scoredFacts[i].score != scoredFacts[j].score {
			return scoredFacts[i].score > scoredFacts[j].score
		}
		return scoredFacts[i].fact.UUID < scoredFacts[j].fact.UUID
	})

	if len(scoredFacts) > limit {
		scoredFacts = scoredFacts[:limit]
	}

	out := make([]*ConsciousFact, len(scoredFacts))
	for i, sf := range scoredFacts {
		out[i] = &ConsciousFact{
			ID:           sf.fact.UUID,
			Content:      sf.fact.Content,
			Significance: int(sf.fact.Significance),
			Tag:          sf.fact.Tag,
		}
	}
	return out, nil
}
