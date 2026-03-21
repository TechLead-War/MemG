package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/memory/augment"
	"memg/store"
)

// extractedFact is the JSON shape returned by the extraction LLM.
type extractedFact struct {
	Content       string   `json:"content"`
	Type          string   `json:"type"`
	Significance  int      `json:"significance"`
	Tag           string   `json:"tag"`
	Slot          string   `json:"slot"`
	ReferenceTime string   `json:"reference_time"`
	Confidence    *float64 `json:"confidence"`
}

// DefaultExtractionStage is the built-in extraction stage that the proxy uses
// when no custom stages are registered. It prompts an LLM to extract structured
// facts from conversation messages and embeds them for vector search.
type DefaultExtractionStage struct {
	provider       llm.Provider
	embedder       embed.Embedder
	embeddingModel string
}

// NewDefaultExtractionStage creates the built-in extraction stage.
func NewDefaultExtractionStage(provider llm.Provider, embedder embed.Embedder) *DefaultExtractionStage {
	return &DefaultExtractionStage{
		provider:       provider,
		embedder:       embedder,
		embeddingModel: embed.ModelNameOf(embedder),
	}
}

// Name returns the human-readable label for this stage.
func (s *DefaultExtractionStage) Name() string { return "default-extraction" }

// Execute analyses the job's messages and returns any extracted knowledge.
func (s *DefaultExtractionStage) Execute(ctx context.Context, job *augment.Job) (*augment.Extraction, error) {
	if len(job.Messages) == 0 {
		return nil, nil
	}

	if isTrivialTurn(job.Messages) {
		return nil, nil
	}

	userMessages := userOnlyMessages(job.Messages)
	if len(userMessages) == 0 {
		return nil, nil
	}

	// Build a transcript from user-grounded messages only. This prevents the
	// assistant's speculation from hardening into durable memory by default.
	var transcript strings.Builder
	for _, msg := range userMessages {
		fmt.Fprintf(&transcript, "%s: %s\n", msg.Role, msg.Content)
	}

	// Build the extraction prompt with today's date for temporal resolution.
	now := time.Now()
	today := now.Format("2006-01-02")
	yesterday := now.AddDate(0, 0, -1).Format("2006-01-02")

	prompt := fmt.Sprintf(`You are a knowledge extraction engine. Today's date is %s.

Extract facts from this conversation. Return a JSON array. Each fact:
{
  "content": "the fact as a clear statement about the user",
  "type": "identity|event|pattern",
  "significance": 1-10,
  "tag": "category label",
  "slot": "semantic slot name (e.g. location, job, diet, name, email, relationship, preference)",
  "reference_time": "ISO date if time-bound, empty string if not",
  "confidence": 0.0-1.0
}

Rules:
- "identity" = enduring truths (preferences, attributes, relationships)
- "event" = things that happened at a specific time
- "pattern" = behavioral tendencies observed across the conversation
- "tag" = a category label. Use one of: skill, preference, relationship, medical, location, work, hobby, personal, financial, or other
- "slot" = the semantic slot this fact fills. Use: location, job, diet, name, email, relationship, preference, medical, hobby, skill, or other
- "reference_time" = ISO 8601 date (YYYY-MM-DD) for time-bound facts, empty string otherwise
- "confidence" = how confident you are (1.0 = explicitly stated by user, 0.5 = inferred, 0.0 = guessing)
- Resolve relative dates: "today" → "%s", "yesterday" → "%s"
- Significance: 10 = life-critical (allergies, medical), 7-9 = important (job, location), 4-6 = moderate, 1-3 = trivial (lunch, weather)
- Skip greetings, filler, "thank you", and trivial exchanges
- If nothing is worth extracting, return []

Return ONLY the JSON array, no other text.`, today, today, yesterday)

	req := &llm.Request{
		System:    prompt,
		Messages:  []*llm.Message{llm.UserMessage(transcript.String())},
		MaxTokens: 2048,
	}

	resp, err := chatWithRetry(ctx, s.provider, req, 3)
	if err != nil {
		return nil, fmt.Errorf("extraction llm call: %w", err)
	}

	extracted, err := parseExtractionResponse(resp.Content)
	if err != nil {
		return nil, fmt.Errorf("extraction parse: %w", err)
	}

	extracted = validateExtraction(extracted)
	if len(extracted) == 0 {
		return nil, nil
	}

	facts := make([]*store.Fact, 0, len(extracted))
	contents := make([]string, 0, len(extracted))

	for _, ef := range extracted {
		factType := resolveFactType(ef.Type)
		significance := clampSignificance(ef.Significance)

		f := &store.Fact{
			Content:        ef.Content,
			Type:           factType,
			Significance:   significance,
			TemporalStatus: store.TemporalCurrent,
			Tag:            strings.ToLower(strings.TrimSpace(ef.Tag)),
			Slot:           strings.ToLower(strings.TrimSpace(ef.Slot)),
			Confidence:     confidenceValue(ef.Confidence),
			EmbeddingModel: s.embeddingModel,
			SourceRole:     llm.RoleUser,
			ExpiresAt:      store.TTLForSignificance(significance),
		}
		if ef.ReferenceTime != "" {
			if t, parseErr := time.Parse("2006-01-02", ef.ReferenceTime); parseErr == nil {
				f.ReferenceTime = &t
			}
		}
		facts = append(facts, f)
		contents = append(contents, ef.Content)
	}

	embeddings, err := s.embedder.Embed(ctx, contents)
	if err != nil {
		return nil, fmt.Errorf("extraction embed: %w", err)
	}

	for i, emb := range embeddings {
		if i < len(facts) {
			facts[i].Embedding = emb
		}
	}

	return &augment.Extraction{Facts: facts}, nil
}

// parseExtractionResponse parses the LLM response into a slice of extracted
// facts. It is resilient to common LLM response quirks:
//   - Direct JSON array
//   - JSON wrapped in markdown code blocks (```json ... ```)
//   - JSON array embedded in surrounding text
func parseExtractionResponse(content string) ([]extractedFact, error) {
	content = strings.TrimSpace(content)

	var facts []extractedFact
	if err := json.Unmarshal([]byte(content), &facts); err == nil {
		return facts, nil
	}

	stripped := stripCodeFences(content)
	if stripped != content {
		if err := json.Unmarshal([]byte(stripped), &facts); err == nil {
			return facts, nil
		}
	}

	start := strings.Index(content, "[")
	end := strings.LastIndex(content, "]")
	if start >= 0 && end > start {
		candidate := content[start : end+1]
		if err := json.Unmarshal([]byte(candidate), &facts); err == nil {
			return facts, nil
		}
	}

	return nil, fmt.Errorf("could not parse JSON array from response: %.100s", content)
}

// stripCodeFences removes markdown code block wrappers (```json ... ``` or ``` ... ```).
func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)

	if strings.HasPrefix(s, "```") {
		if idx := strings.Index(s, "\n"); idx >= 0 {
			s = s[idx+1:]
		} else {
			s = strings.TrimPrefix(s, "```json")
			s = strings.TrimPrefix(s, "```")
		}
	}

	if strings.HasSuffix(s, "```") {
		s = strings.TrimSuffix(s, "```")
	}

	return strings.TrimSpace(s)
}

// resolveFactType maps the LLM's type string to a store.FactType, defaulting
// to identity for unrecognised values.
func resolveFactType(t string) store.FactType {
	switch strings.ToLower(strings.TrimSpace(t)) {
	case "event":
		return store.FactTypeEvent
	case "pattern":
		return store.FactTypePattern
	case "identity":
		return store.FactTypeIdentity
	default:
		return store.FactTypeIdentity
	}
}

// clampSignificance ensures the value is within the valid range [1, 10],
// defaulting to SignificanceMedium for out-of-range or zero values.
func clampSignificance(v int) store.Significance {
	if v < 1 || v > 10 {
		return store.SignificanceMedium
	}
	return store.Significance(v)
}

// validateExtraction filters out invalid facts from the extraction result.
// It drops: empty content, oversized content (>500 chars), invalid dates,
// invalid tags/types, and clamps confidence to [0, 1].
func validateExtraction(facts []extractedFact) []extractedFact {
	valid := make([]extractedFact, 0, len(facts))
	validTags := map[string]bool{
		"skill": true, "preference": true, "relationship": true,
		"medical": true, "location": true, "work": true,
		"hobby": true, "personal": true, "financial": true, "other": true,
	}
	for _, f := range facts {
		content := strings.TrimSpace(f.Content)
		if content == "" {
			continue
		}
		if len(content) > 500 {
			continue
		}
		if f.ReferenceTime != "" {
			if _, err := time.Parse("2006-01-02", f.ReferenceTime); err != nil {
				f.ReferenceTime = "" // Clear invalid date rather than dropping the fact.
			}
		}
		// Clamp confidence to [0, 1] when present. Missing confidence is handled
		// later so explicit 0.0 remains low-confidence instead of being upgraded.
		if f.Confidence != nil {
			if *f.Confidence < 0 {
				v := 0.0
				f.Confidence = &v
			}
			if *f.Confidence > 1 {
				v := 1.0
				f.Confidence = &v
			}
		}
		f.Tag = strings.ToLower(strings.TrimSpace(f.Tag))
		if f.Tag != "" && !validTags[f.Tag] {
			f.Tag = "other"
		}

		f.Content = content
		valid = append(valid, f)
	}
	return valid
}

// isTrivialTurn checks whether the conversation contains only trivial
// content (greetings, acknowledgments) that would not produce meaningful facts.
// This avoids wasting an LLM call on "thanks", "ok", "hello", etc.
func isTrivialTurn(messages []*llm.Message) bool {
	if len(messages) == 0 {
		return true
	}

	trivialPatterns := []string{
		"thanks", "thank you", "ok", "okay", "got it", "sure",
		"hello", "hi", "hey", "bye", "goodbye", "good morning",
		"good night", "good evening", "good afternoon", "yes", "no",
		"yep", "nope", "cool", "great", "awesome", "nice",
		"understood", "roger", "ack", "k", "kk", "lol", "haha",
	}

	// Check only user messages — assistant content doesn't matter for extraction gating.
	for _, msg := range messages {
		if msg.Role != llm.RoleUser {
			continue
		}
		content := strings.ToLower(strings.TrimSpace(msg.Content))
		content = strings.Map(func(r rune) rune {
			if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == ' ' {
				return r
			}
			return -1
		}, content)
		content = strings.TrimSpace(content)

		if content == "" {
			continue
		}

		isTrivial := false
		for _, pattern := range trivialPatterns {
			if content == pattern {
				isTrivial = true
				break
			}
		}

		if !isTrivial {
			return false
		}
	}

	return true
}

func userOnlyMessages(messages []*llm.Message) []*llm.Message {
	out := make([]*llm.Message, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == llm.RoleUser {
			out = append(out, msg)
		}
	}
	return out
}

func confidenceValue(v *float64) float64 {
	if v == nil {
		return 0.8
	}
	return *v
}

// chatWithRetry calls the LLM provider with exponential backoff on transient
// errors (rate limits, server errors). Extraction is background work so
// waiting a few seconds is acceptable.
func chatWithRetry(ctx context.Context, provider llm.Provider, req *llm.Request, maxRetries int) (*llm.Response, error) {
	backoff := time.Second
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err := provider.Chat(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		// Check if this is a retryable error (rate limit or server error).
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

		fmt.Printf("memg proxy: extraction attempt %d failed (%v), retrying in %s\n", attempt+1, err, backoff)

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
		}
		backoff *= 2 // exponential: 1s, 2s, 4s
	}

	return nil, lastErr
}
