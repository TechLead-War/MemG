package search

import "sort"

// scored pairs a candidate index with its computed relevance score.
type scored struct {
	idx   int
	score float64
}

// Hybrid combines dense vector similarity with sparse lexical scoring.
type Hybrid struct{}

// NewHybrid creates a hybrid search engine.
func NewHybrid() *Hybrid { return &Hybrid{} }

// Rank scores candidates using a weighted combination of cosine similarity
// and BM25 lexical relevance, then returns the top results above threshold.
func (h *Hybrid) Rank(query []float32, queryText string, candidates []Candidate, limit int, threshold float64) []Result {
	if len(candidates) == 0 {
		return nil
	}

	dense := vectorScores(query, candidates)
	lexical := bm25Scores(queryText, candidates)

	wDense, wLex := blendWeights(queryText)

	merged := make([]scored, len(candidates))
	for i := range candidates {
		score := wDense*dense[i] + wLex*lexical[i]

		// Historical facts receive a 15% penalty — they're still findable
		// but demoted below equally-relevant current facts.
		if candidates[i].TemporalStatus == "historical" {
			score *= 0.85
		}

		// Penalize low-confidence facts (max penalty: 5% of score).
		// Default confidence to 1.0 when not set (0 means unset).
		conf := candidates[i].Confidence
		if conf <= 0 {
			conf = 1.0
		}
		if conf < 1.0 {
			score *= (0.95 + 0.05*conf) // 95%-100% of original score
		}

		merged[i] = scored{idx: i, score: score}
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].score > merged[j].score
	})

	// Collect all candidates above threshold, up to hard limit.
	// Threshold comparison uses the raw relevance score (without significance
	// tiebreaker) so that significance cannot promote below-threshold results.
	var above []scored
	for _, s := range merged {
		if s.score < threshold {
			break
		}
		// Add significance tiebreaker after threshold check: max 0.01 boost
		// so it only affects sort order, never threshold admission.
		s.score += float64(candidates[s.idx].Significance) * 0.001
		above = append(above, s)
		if len(above) >= limit {
			break
		}
	}

	// Re-sort after adding significance tiebreaker.
	sort.Slice(above, func(i, j int) bool {
		return above[i].score > above[j].score
	})

	// Apply Kneedle algorithm: find where the score curve bends and cut there.
	cutoff := kneedleCutoff(above)

	results := make([]Result, cutoff)
	for i := 0; i < cutoff; i++ {
		c := candidates[above[i].idx]
		results[i] = Result{
			ID:             c.ID,
			Content:        c.Content,
			Score:          above[i].score,
			CreatedAt:      c.CreatedAt,
			TemporalStatus: c.TemporalStatus,
			Significance:   c.Significance,
		}
	}
	return results
}

// kneedleCutoff finds the "knee" in sorted scores using the Kneedle algorithm
// (Satopaa et al., ICDCS 2011). Items must be sorted by score descending.
// It normalizes both axes to [0,1], draws a diagonal from the first point
// (highest score) to the last point (lowest score), and finds the point
// where the actual curve deviates most below the diagonal. Everything
// before that point is kept; everything after is discarded.
func kneedleCutoff(items []scored) int {
	if len(items) <= 3 {
		return len(items)
	}

	n := len(items)
	maxScore := items[0].score
	minScore := items[n-1].score
	scoreRange := maxScore - minScore

	if scoreRange < 1e-9 {
		return n
	}

	bestDeviation := 0.0
	kneeIdx := n

	for i := 0; i < n; i++ {
		x := float64(i) / float64(n-1)
		y := (items[i].score - minScore) / scoreRange
		diagonal := 1.0 - x
		deviation := diagonal - y

		if deviation > bestDeviation {
			bestDeviation = deviation
			kneeIdx = i
		}
	}

	if kneeIdx <= 0 {
		return 1
	}
	if kneeIdx >= n {
		return n
	}
	return kneeIdx
}

// blendWeights returns (dense, lexical) weights. Short queries receive a
// higher lexical weight because BM25 is more reliable for keyword input.
func blendWeights(queryText string) (float64, float64) {
	tokens := tokenize(queryText)
	if len(tokens) <= 2 {
		return 0.70, 0.30
	}
	return 0.85, 0.15
}
