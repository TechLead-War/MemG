// Package search implements hybrid vector + lexical retrieval.
package search

import "time"

// Engine scores a query against a set of candidates and returns the top matches.
type Engine interface {
	Rank(query []float32, queryText string, candidates []Candidate, limit int, threshold float64) []Result
}

// Candidate is a single stored fact eligible for recall.
type Candidate struct {
	ID             string
	Content        string
	Embedding      []float32
	CreatedAt      time.Time
	TemporalStatus string  // "current" or "historical"
	Significance   int     // 1–10
	Confidence     float64 // 0.0-1.0, default 1.0
}

// Result is a scored candidate that passed the relevance threshold.
type Result struct {
	ID             string
	Content        string
	Score          float64
	CreatedAt      time.Time
	TemporalStatus string
	Significance   int
}
