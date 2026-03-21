package memory

import (
	"strings"
	"unicode/utf8"
)

// ContextBudget controls how tokens are allocated across memory categories.
type ContextBudget struct {
	TotalTokens   int // overall budget
	SummaryTokens int // max for summaries
}

// ContextInput holds the raw components to be merged.
type ContextInput struct {
	ConsciousFacts []*ConsciousFact
	RecalledFacts  []*RecalledFact
	Summaries      []*RecalledSummary
	Budget         ContextBudget
}

// BuildContext merges conscious facts, recalled facts, and summaries into a
// single context string under a token budget, with cross-component dedup.
// It prioritizes: conscious > recalled > summaries.
func BuildContext(input ContextInput) string {
	budget := input.Budget.TotalTokens
	if budget <= 0 {
		budget = 4000
	}
	summaryBudget := input.Budget.SummaryTokens
	if summaryBudget <= 0 {
		summaryBudget = 1000
	}

	var b strings.Builder
	seen := make(map[string]struct{})
	tokensUsed := 0

	// 1. Conscious facts (highest priority).
	if len(input.ConsciousFacts) > 0 {
		b.WriteString("User profile:\n")
		for _, f := range input.ConsciousFacts {
			normalized := normalizeForDedup(f.Content)
			if _, ok := seen[normalized]; ok {
				continue
			}
			seen[normalized] = struct{}{}
			line := "- " + f.Content + "\n"
			est := estimateTokens(line)
			if tokensUsed+est > budget {
				break
			}
			b.WriteString(line)
			tokensUsed += est
		}
	}

	// 2. Recalled facts (medium priority) — dedup against conscious.
	if len(input.RecalledFacts) > 0 {
		var factsSection strings.Builder
		factsSection.WriteString("\nRelevant context from memory:\n")
		headerTokens := estimateTokens(factsSection.String())
		sectionTokens := headerTokens
		any := false

		for _, f := range input.RecalledFacts {
			normalized := normalizeForDedup(f.Content)
			if _, ok := seen[normalized]; ok {
				continue
			}
			seen[normalized] = struct{}{}
			line := "- "
			if f.TemporalStatus == "historical" {
				line += "[historical] "
			}
			line += f.Content + "\n"
			est := estimateTokens(line)
			if tokensUsed+sectionTokens+est > budget {
				break
			}
			factsSection.WriteString(line)
			sectionTokens += est
			any = true
		}

		if any {
			b.WriteString(factsSection.String())
			tokensUsed += sectionTokens
		}
	}

	// 3. Summaries (lowest priority, own sub-budget).
	if len(input.Summaries) > 0 {
		effectiveBudget := summaryBudget
		remaining := budget - tokensUsed
		if effectiveBudget > remaining {
			effectiveBudget = remaining
		}

		if effectiveBudget > 0 {
			var sumSection strings.Builder
			sumSection.WriteString("\nRelevant past conversations:\n")
			headerTokens := estimateTokens(sumSection.String())
			sectionTokens := headerTokens
			any := false

			for _, s := range input.Summaries {
				normalized := normalizeForDedup(s.Summary)
				if _, ok := seen[normalized]; ok {
					continue
				}
				seen[normalized] = struct{}{}
				line := "- [" + s.CreatedAt.Format("Jan 2, 2006") + "] " + s.Summary + "\n"
				est := estimateTokens(line)
				if sectionTokens+est > effectiveBudget {
					break
				}
				sumSection.WriteString(line)
				sectionTokens += est
				any = true
			}

			if any {
				b.WriteString(sumSection.String())
			}
		}
	}

	return strings.TrimRight(b.String(), "\n")
}

// normalizeForDedup strips common prefixes and normalizes casing for more
// effective cross-component dedup in BuildContext. This catches variants like
// "The user lives in Seattle" vs "User lives in Seattle".
func normalizeForDedup(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	for _, prefix := range []string{"the user ", "user ", "user's ", "[historical] "} {
		s = strings.TrimPrefix(s, prefix)
	}
	return strings.TrimSpace(s)
}

// estimateTokens provides a cross-language token count approximation.
// Word-based estimation (~1.3 tokens per word) works well for English.
// For CJK text with few spaces, falls back to rune-based estimation.
func estimateTokens(s string) int {
	words := len(strings.Fields(s))
	if words > 0 {
		return (words*13 + 9) / 10 // ceil(words * 1.3)
	}
	// No spaces (likely CJK) — roughly 1 token per 1.5 runes.
	runes := utf8.RuneCountInString(s)
	if runes == 0 {
		return 1
	}
	return (runes*2 + 2) / 3
}
