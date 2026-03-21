package search

import (
	"math"
	"sort"
)

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

		// Significance tiebreaker: max 0.01 boost so it never overrides relevance.
		score += float64(candidates[i].Significance) * 0.001

		// Penalize low-confidence facts (max penalty: 5% of score).
		if candidates[i].Confidence > 0 && candidates[i].Confidence < 1.0 {
			score *= (0.95 + 0.05*candidates[i].Confidence) // 95%-100% of original score
		}

		merged[i] = scored{idx: i, score: score}
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].score > merged[j].score
	})

	// Collect all candidates above threshold, up to hard limit.
	var above []scored
	for _, s := range merged {
		if s.score < threshold {
			break
		}
		above = append(above, s)
		if len(above) >= limit {
			break
		}
	}

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

// kneedleCutoff implements the Kneedle algorithm (Satopaa et al., ICDCS 2011)
// to find the "knee" in a sorted score curve — the point where scores
// transition from genuinely relevant to noise.
//
// The algorithm works by:
//  1. Normalizing both axes (position and score) to [0, 1].
//  2. Computing the difference between each normalized score and the
//     diagonal line connecting the first and last points.
//  3. Finding the position with the maximum difference — this is where
//     the curve deviates most from a straight decline, i.e. the knee.
//
// If no significant knee is found, all results are returned.
func kneedleCutoff(items []scored) int {
	n := len(items)
	if n <= 2 {
		return n
	}

	// Step 1: Normalize x (position) and y (score) to [0, 1].
	yMin := items[n-1].score
	yMax := items[0].score
	yRange := yMax - yMin
	if yRange == 0 {
		// All scores are identical — no knee, keep everything.
		return n
	}

	// Step 2: Compute the difference curve.
	// The diagonal goes from (0, 1) to (1, 0) in normalized space.
	// difference[i] = normalized_y[i] - diagonal[i]
	// The knee is where this difference is maximized — the curve is
	// farthest above the straight line connecting endpoints.
	maxDiff := 0.0
	kneeIdx := 0

	for i := 0; i < n; i++ {
		xNorm := float64(i) / float64(n-1)        // 0 to 1
		yNorm := (items[i].score - yMin) / yRange // 1 to 0
		diagonal := 1.0 - xNorm                   // straight line from (0,1) to (1,0)
		diff := yNorm - diagonal

		if diff > maxDiff {
			maxDiff = diff
			kneeIdx = i
		}
	}

	// Step 3: Validate the knee.
	// If the maximum difference is too small, there's no meaningful knee —
	// the scores decline roughly linearly. Keep everything.
	// A threshold of 0.1 in normalized space means the curve must deviate
	// at least 10% from the diagonal to be considered a real knee.
	const minKneeStrength = 0.10
	if maxDiff < minKneeStrength {
		return n
	}

	// Cut after the knee point — keep the knee and everything above it.
	// math.Min ensures we return at least 1.
	return int(math.Max(1, float64(kneeIdx+1)))
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
