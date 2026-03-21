package search

import "math"

// DimensionMatch checks if two vectors have compatible dimensions.
// Returns false if either is empty or they differ in length.
func DimensionMatch(a, b []float32) bool {
	return len(a) > 0 && len(b) > 0 && len(a) == len(b)
}

// CosineSimilarity computes the cosine of the angle between two vectors.
// Both vectors must have the same length; returns 0 if either has zero norm.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// vectorScores computes cosine similarity between a query vector and each
// candidate's embedding. Returns a score slice aligned with candidates.
func vectorScores(query []float32, candidates []Candidate) []float64 {
	scores := make([]float64, len(candidates))
	for i, c := range candidates {
		scores[i] = CosineSimilarity(query, c.Embedding)
	}
	return scores
}
