package search

import (
	"math"
	"strings"
	"unicode"
)

// bm25Scores computes Okapi BM25 relevance scores for each candidate against
// the given query text. Scores are normalised to [0, 1].
func bm25Scores(queryText string, candidates []Candidate) []float64 {
	terms := tokenize(queryText)
	if len(terms) == 0 || len(candidates) == 0 {
		return make([]float64, len(candidates))
	}

	docs := make([][]string, len(candidates))
	avgLen := 0.0
	for i, c := range candidates {
		docs[i] = tokenize(c.Content)
		avgLen += float64(len(docs[i]))
	}
	avgLen /= float64(len(docs))

	n := float64(len(docs))
	idf := make(map[string]float64, len(terms))
	for _, t := range terms {
		df := 0.0
		for _, doc := range docs {
			if containsTerm(doc, t) {
				df++
			}
		}
		idf[t] = math.Log((n-df+0.5)/(df+0.5) + 1.0)
	}

	const k1 = 1.2
	const b = 0.75

	raw := make([]float64, len(docs))
	peak := 0.0
	for i, doc := range docs {
		dl := float64(len(doc))
		tf := termFrequencies(doc, terms)
		for _, t := range terms {
			f := tf[t]
			num := f * (k1 + 1)
			denom := f + k1*(1-b+b*(dl/avgLen))
			raw[i] += idf[t] * num / denom
		}
		if raw[i] > peak {
			peak = raw[i]
		}
	}

	scores := make([]float64, len(raw))
	if peak > 0 {
		for i := range raw {
			scores[i] = raw[i] / peak
		}
	}
	return scores
}

func tokenize(text string) []string {
	lower := strings.ToLower(text)
	words := strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	filtered := words[:0]
	for _, w := range words {
		if !stopWords[w] {
			filtered = append(filtered, w)
		}
	}
	return filtered
}

func containsTerm(doc []string, term string) bool {
	for _, w := range doc {
		if w == term {
			return true
		}
	}
	return false
}

func termFrequencies(doc []string, terms []string) map[string]float64 {
	tf := make(map[string]float64, len(terms))
	for _, w := range doc {
		for _, t := range terms {
			if w == t {
				tf[t]++
			}
		}
	}
	return tf
}

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true, "need": true,
	"to": true, "of": true, "in": true, "for": true, "on": true,
	"with": true, "at": true, "by": true, "from": true, "as": true,
	"into": true, "about": true, "like": true, "through": true, "after": true,
	"over": true, "between": true, "out": true, "against": true, "during": true,
	"without": true, "before": true, "under": true, "around": true, "among": true,
	"and": true, "but": true, "or": true, "nor": true, "not": true, "so": true,
	"yet": true, "both": true, "either": true, "neither": true, "each": true,
	"every": true, "all": true, "any": true, "few": true, "more": true,
	"most": true, "other": true, "some": true, "such": true, "no": true,
	"only": true, "own": true, "same": true, "than": true, "too": true,
	"very": true, "just": true, "because": true, "if": true, "when": true,
	"where": true, "how": true, "what": true, "which": true, "who": true,
	"whom": true, "this": true, "that": true, "these": true, "those": true,
	"i": true, "me": true, "my": true, "we": true, "our": true,
	"you": true, "your": true, "he": true, "him": true, "his": true,
	"she": true, "her": true, "it": true, "its": true, "they": true,
	"them": true, "their": true,
}
