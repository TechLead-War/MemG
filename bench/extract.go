// Package bench provides the LoCoMo benchmark runner.
//
// Extraction uses the library's DefaultExtractionStage with a custom prompt
// tailored for multi-speaker peer conversations. All extraction, validation,
// embedding, and temporal resolution logic lives in the library — the bench
// only configures the prompt.
package bench

import (
	"context"
	"regexp"
	"strings"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/memory/augment"
	"memg/proxy"

	"github.com/tj/go-naturaldate"
)

// NewExtractionStage creates the bench extraction stage. It reuses the
// library's DefaultExtractionStage with a LoCoMo-tailored prompt that
// handles multi-speaker conversations and aggressive temporal extraction.
func NewExtractionStage(provider llm.Provider, embedder embed.Embedder) augment.Stage {
	return proxy.NewDefaultExtractionStage(provider, embedder).
		WithPromptBuilder(benchPrompt)
}

// benchPrompt builds an extraction prompt tailored for LoCoMo multi-speaker
// conversations with aggressive temporal extraction.
func benchPrompt(sessionCtx string) string {
	var b strings.Builder
	b.WriteString(`You are a knowledge extraction engine. `)
	if sessionCtx != "" {
		b.WriteString(sessionCtx)
		b.WriteByte(' ')
	}
	b.WriteString(`
Extract factual information from this conversation between two people. Return a JSON array. Each fact:
{
  "content": "clear factual statement attributed to the relevant person by name, with absolute dates",
  "type": "identity|event|pattern",
  "significance": 1-10,
  "tag": "category label",
  "slot": "semantic slot name",
  "reference_time": "YYYY-MM-DD if time-bound, empty string otherwise",
  "confidence": 0.0-1.0
}

Rules:
- Extract facts about ALL speakers. Always include the person's name in the fact.
- "identity" = enduring truths (preferences, attributes, relationships, personality)
- "event" = things that happened at a specific time or date
- "pattern" = behavioral tendencies observed across the conversation
- "tag": use one of: skill, preference, relationship, medical, location, work, hobby, personal, financial, or other
- "slot": use: location, job, diet, name, email, relationship, preference, medical, hobby, skill, family, education, or other
- "reference_time" = ISO 8601 date (YYYY-MM-DD) for time-bound events
- "confidence": 1.0 = explicitly stated, 0.7 = strongly implied, 0.4 = inferred
- Significance: 10 = life-critical (medical, allergies), 7-9 = important (job, location, major events), 4-6 = moderate (hobbies, preferences), 1-3 = trivial

TEMPORAL EXTRACTION — CRITICAL:
- Extract ALL events, activities, plans, achievements, and time-bound facts, even minor ones.
- ALWAYS resolve relative dates to absolute dates using the session date.
  - "yesterday" → compute the actual calendar date and use it
  - "last week" → compute the approximate date
  - "two years ago" → compute the year
  - "next month" → compute the actual month and year
- In the "content" field, ALWAYS use absolute dates (e.g. "on January 20, 2023"), NEVER relative ones (e.g. "yesterday").
- In the "reference_time" field, provide the ISO 8601 date (YYYY-MM-DD).
- If a speaker mentions a past event, hobby start date, adoption date, purchase date, trip, class, or any activity with a time reference, extract it as an event fact with the resolved date.

- Extract specific details: names, dates, places, relationships, emotions, sentiments, attitudes, opinions, plans, achievements
- Also extract what people SAY about things (e.g. "Gina described the studio as amazing")
- Do NOT extract greetings or filler like 'hi', 'bye', 'thanks'. DO extract emotions, opinions, and attitudes even if they seem casual — they are facts
- If nothing is worth extracting, return []

Return ONLY the JSON array.`)
	return b.String()
}

// --- Temporal resolution helpers used by the bench runner ---

var sessionDateRe = regexp.MustCompile(`on\s+(\d{1,2}\s+\w+,?\s+\d{4})`)

func parseSessionDate(sessionCtx string) *time.Time {
	if sessionCtx == "" {
		return nil
	}
	m := sessionDateRe.FindStringSubmatch(sessionCtx)
	if len(m) < 2 {
		return nil
	}
	raw := m[1]
	for _, layout := range []string{
		"2 January, 2006",
		"2 January 2006",
		"02 January, 2006",
		"02 January 2006",
	} {
		if t, err := time.Parse(layout, raw); err == nil {
			return &t
		}
	}
	return nil
}

var relativeDatePatterns = regexp.MustCompile(
	`(?i)\b(yesterday|today|tomorrow|last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|night)|` +
		`this\s+(?:week|month|morning)|next\s+(?:week|month|year)|` +
		`(?:a\s+)?(?:few|couple(?:\s+of)?|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?)\s+ago|` +
		`the\s+(?:day\s+before\s+yesterday|other\s+day)|recently|the\s+previous\s+\w+)`)

func inferReferenceTime(content string, sessionDate time.Time) string {
	match := relativeDatePatterns.FindString(content)
	if match == "" {
		return ""
	}
	resolved, err := naturaldate.Parse(match, sessionDate, naturaldate.WithDirection(naturaldate.Past))
	if err != nil {
		return ""
	}
	return resolved.Format("2006-01-02")
}

func embedDateInContent(content, refTime string, sessionDate *time.Time) string {
	t, err := time.Parse("2006-01-02", refTime)
	if err != nil {
		return content
	}
	formatted := t.Format("January 2, 2006")

	if sessionDate != nil {
		loc := relativeDatePatterns.FindStringIndex(content)
		if loc != nil {
			before := content[:loc[0]]
			after := content[loc[1]:]
			return before + "on " + formatted + after
		}
	}

	if strings.Contains(content, refTime) || strings.Contains(content, formatted) {
		return content
	}
	monthYear := t.Format("January 2006")
	if strings.Contains(content, monthYear) {
		return content
	}

	return content + " (on " + formatted + ")"
}

func chatRetry(ctx context.Context, provider llm.Provider, req *llm.Request, maxRetries int) (*llm.Response, error) {
	backoff := time.Second
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err := provider.Chat(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		errMsg := err.Error()
		retryable := strings.Contains(errMsg, "429") ||
			strings.Contains(errMsg, "500") ||
			strings.Contains(errMsg, "502") ||
			strings.Contains(errMsg, "503") ||
			strings.Contains(errMsg, "rate") ||
			strings.Contains(errMsg, "timeout")

		if !retryable || attempt == maxRetries {
			return nil, err
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
		}
		backoff *= 2
	}
	return nil, lastErr
}

