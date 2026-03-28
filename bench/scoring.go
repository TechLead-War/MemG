package bench

import (
	"strings"
	"unicode"
)

// normalizeAnswer applies standard NLP normalization for token F1:
// lowercase, remove articles, remove punctuation, collapse whitespace.
func normalizeAnswer(s string) string {
	s = strings.ToLower(s)
	s = strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsSpace(r) {
			return r
		}
		return -1
	}, s)

	words := strings.Fields(s)
	filtered := make([]string, 0, len(words))
	for _, w := range words {
		if w == "a" || w == "an" || w == "the" {
			continue
		}
		filtered = append(filtered, w)
	}
	return strings.Join(filtered, " ")
}

// TokenF1 computes token-level F1 between a predicted and gold answer.
func TokenF1(predicted, gold string) float64 {
	predNorm := normalizeAnswer(predicted)
	goldNorm := normalizeAnswer(gold)

	predTokens := strings.Fields(predNorm)
	goldTokens := strings.Fields(goldNorm)

	if len(predTokens) == 0 && len(goldTokens) == 0 {
		return 1.0
	}
	if len(predTokens) == 0 || len(goldTokens) == 0 {
		return 0.0
	}

	goldCounts := make(map[string]int)
	for _, t := range goldTokens {
		goldCounts[t]++
	}

	common := 0
	for _, t := range predTokens {
		if goldCounts[t] > 0 {
			common++
			goldCounts[t]--
		}
	}

	if common == 0 {
		return 0.0
	}

	precision := float64(common) / float64(len(predTokens))
	recall := float64(common) / float64(len(goldTokens))
	return 2 * precision * recall / (precision + recall)
}

// ExactMatch returns 1.0 if the normalized answers are identical, 0.0 otherwise.
func ExactMatch(predicted, gold string) float64 {
	if normalizeAnswer(predicted) == normalizeAnswer(gold) {
		return 1.0
	}
	return 0.0
}

// ScoreMultiHop handles multi-hop questions (category 1) where the gold
// answer may contain comma-separated parts. Computes F1 for each part
// and the full string, returns the maximum.
func ScoreMultiHop(predicted, gold string) float64 {
	best := TokenF1(predicted, gold)

	parts := strings.Split(gold, ",")
	if len(parts) <= 1 {
		return best
	}

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if f1 := TokenF1(predicted, part); f1 > best {
			best = f1
		}
	}
	return best
}

// ScoreAdversarial returns 1.0 if the model correctly identifies that the
// information is not available (negation), 0.0 if it hallucinates an answer.
func ScoreAdversarial(predicted string) float64 {
	lower := strings.ToLower(predicted)
	negations := []string{
		"not mentioned", "no information", "not available",
		"no mention", "cannot determine", "can't determine",
		"don't have", "do not have", "not provided",
		"not stated", "no evidence", "not found",
		"i don't know", "not in the", "not in my",
		"no record", "not discussed", "not addressed",
		"not specified", "no data", "no relevant",
		"cannot be determined", "isn't mentioned",
		"is not mentioned", "wasn't mentioned",
		"was not mentioned", "not enough information",
	}
	for _, neg := range negations {
		if strings.Contains(lower, neg) {
			return 1.0
		}
	}
	return 0.0
}

// ScoreQA dispatches to the correct scoring function based on category.
func ScoreQA(predicted string, qa QA) float64 {
	switch qa.Category {
	case 1:
		return ScoreMultiHop(predicted, string(qa.Answer))
	case 5:
		return ScoreAdversarial(predicted)
	default:
		gold := string(qa.Answer)
		// Category 3 may have semicolon-delimited answers; use first part.
		if qa.Category == 3 {
			if idx := strings.Index(gold, ";"); idx >= 0 {
				gold = gold[:idx]
			}
		}
		return TokenF1(predicted, gold)
	}
}
