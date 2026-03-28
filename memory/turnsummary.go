package memory

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"memg/embed"
	"memg/llm"
	"memg/store"
)

const turnSummaryPrompt = `Summarize these conversation turns. Focus on: decisions made, questions asked, code discussed, data referenced, open items. Keep under 200 tokens.`

const overviewConsolidatePrompt = `Consolidate these summaries into one overview. Keep under 300 tokens.`

// MaintainTurnSummaries checks if messages have fallen off the working memory
// window and generates immutable turn-range summaries for them.
// It also consolidates old summaries when count exceeds 3.
func MaintainTurnSummaries(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	provider llm.Provider,
	conversationID string,
	entityID string,
	messages []*store.Message,
	workingMemoryTurns int,
) error {
	if len(messages) <= workingMemoryTurns {
		return nil
	}

	existing, err := repo.ListTurnSummaries(ctx, conversationID)
	if err != nil {
		return fmt.Errorf("turn summaries: list: %w", err)
	}

	highestEnd := 0
	for _, ts := range existing {
		if !ts.IsOverview && ts.EndTurn > highestEnd {
			highestEnd = ts.EndTurn
		}
	}

	newEnd := len(messages) - workingMemoryTurns
	if newEnd <= highestEnd {
		return nil
	}

	toSummarize := messages[highestEnd:newEnd]
	if len(toSummarize) == 0 {
		return nil
	}

	var transcript strings.Builder
	for _, m := range toSummarize {
		transcript.WriteString(m.Role)
		transcript.WriteString(": ")
		transcript.WriteString(m.Content)
		transcript.WriteByte('\n')
	}

	req := &llm.Request{
		System:    turnSummaryPrompt,
		Messages:  []*llm.Message{llm.UserMessage(transcript.String())},
		MaxTokens: 300,
	}
	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return fmt.Errorf("turn summaries: llm: %w", err)
	}

	summary := strings.TrimSpace(resp.Content)
	if summary == "" {
		return nil
	}

	vectors, err := embedder.Embed(ctx, []string{summary})
	if err != nil {
		return fmt.Errorf("turn summaries: embed: %w", err)
	}
	if len(vectors) == 0 {
		return nil
	}

	ts := &store.TurnSummary{
		ConversationID:   conversationID,
		EntityID:         entityID,
		StartTurn:        highestEnd + 1,
		EndTurn:          newEnd,
		Summary:          summary,
		SummaryEmbedding: vectors[0],
		IsOverview:       false,
	}
	if err := repo.InsertTurnSummary(ctx, ts); err != nil {
		return fmt.Errorf("turn summaries: insert: %w", err)
	}

	return consolidateTurnSummaries(ctx, repo, embedder, provider, conversationID, entityID, existing)
}

func consolidateTurnSummaries(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	provider llm.Provider,
	conversationID, entityID string,
	existingSummaries []*store.TurnSummary,
) error {
	all, err := repo.ListTurnSummaries(ctx, conversationID)
	if err != nil {
		return fmt.Errorf("turn summaries: consolidate list: %w", err)
	}

	var nonOverview []*store.TurnSummary
	var overview *store.TurnSummary
	for _, ts := range all {
		if ts.IsOverview {
			overview = ts
		} else {
			nonOverview = append(nonOverview, ts)
		}
	}

	if len(nonOverview) <= 3 {
		return nil
	}

	sort.Slice(nonOverview, func(i, j int) bool {
		return nonOverview[i].StartTurn < nonOverview[j].StartTurn
	})

	oldest := nonOverview[:2]

	var combined strings.Builder
	if overview != nil {
		combined.WriteString(overview.Summary)
		combined.WriteByte('\n')
	}
	for _, s := range oldest {
		combined.WriteString(s.Summary)
		combined.WriteByte('\n')
	}

	req := &llm.Request{
		System:    overviewConsolidatePrompt,
		Messages:  []*llm.Message{llm.UserMessage(combined.String())},
		MaxTokens: 400,
	}
	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return fmt.Errorf("turn summaries: consolidate llm: %w", err)
	}

	overviewText := strings.TrimSpace(resp.Content)
	if overviewText == "" {
		return nil
	}

	vectors, err := embedder.Embed(ctx, []string{overviewText})
	if err != nil {
		return fmt.Errorf("turn summaries: consolidate embed: %w", err)
	}
	if len(vectors) == 0 {
		return nil
	}

	newOverview := &store.TurnSummary{
		ConversationID:   conversationID,
		EntityID:         entityID,
		StartTurn:        1,
		EndTurn:          oldest[len(oldest)-1].EndTurn,
		Summary:          overviewText,
		SummaryEmbedding: vectors[0],
		IsOverview:       true,
	}
	if err := repo.InsertTurnSummary(ctx, newOverview); err != nil {
		return fmt.Errorf("turn summaries: consolidate insert: %w", err)
	}

	toDelete := make([]string, 0, 3)
	if overview != nil {
		toDelete = append(toDelete, overview.UUID)
	}
	for _, s := range oldest {
		toDelete = append(toDelete, s.UUID)
	}
	if err := repo.DeleteTurnSummaries(ctx, conversationID, toDelete); err != nil {
		return fmt.Errorf("turn summaries: consolidate delete: %w", err)
	}

	return nil
}
