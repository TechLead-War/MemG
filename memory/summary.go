package memory

import (
	"context"
	"fmt"
	"strings"

	"memg/embed"
	"memg/llm"
	"memg/store"
)

const summaryPrompt = `Summarize this conversation. Focus on:
- What was discussed
- What decisions were made
- What is still pending or unresolved
- Any new information learned about the user

Be concise — 2-5 sentences. Only include what is meaningful and worth remembering.
If the conversation contains no meaningful content worth remembering (e.g. just greetings or trivial exchanges), respond with exactly: NONE`

// GenerateAndStoreSummary creates a summary of the conversation's messages,
// embeds it, and stores both on the conversation record. If the conversation
// is trivial (the LLM returns "NONE"), no summary is stored.
func GenerateAndStoreSummary(
	ctx context.Context,
	provider llm.Provider,
	embedder embed.Embedder,
	repo store.Repository,
	conversationUUID string,
) error {
	msgs, err := repo.ReadMessages(ctx, conversationUUID)
	if err != nil {
		return fmt.Errorf("summary: read messages: %w", err)
	}
	if len(msgs) == 0 {
		return nil
	}

	// Limit to the last 100 messages to avoid sending unbounded transcripts
	// to the LLM. Older context is already captured in prior summaries.
	const maxMessages = 100
	if len(msgs) > maxMessages {
		msgs = msgs[len(msgs)-maxMessages:]
	}

	var transcript strings.Builder
	for _, m := range msgs {
		transcript.WriteString(m.Role)
		transcript.WriteString(": ")
		transcript.WriteString(m.Content)
		transcript.WriteByte('\n')
	}

	req := &llm.Request{
		System: summaryPrompt,
		Messages: []*llm.Message{
			llm.UserMessage(transcript.String()),
		},
		MaxTokens: 300,
	}

	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return fmt.Errorf("summary: llm call: %w", err)
	}

	summary := strings.TrimSpace(resp.Content)

	if summary == "" || strings.EqualFold(summary, "NONE") {
		return nil
	}

	vectors, err := embedder.Embed(ctx, []string{summary})
	if err != nil {
		return fmt.Errorf("summary: embed: %w", err)
	}
	if len(vectors) == 0 {
		return nil
	}

	modelName := embed.ModelNameOf(embedder)
	if err := repo.UpdateConversationSummary(ctx, conversationUUID, summary, vectors[0], modelName); err != nil {
		return fmt.Errorf("summary: store: %w", err)
	}

	return nil
}
