package memory

import (
	"context"
	"fmt"
	"strings"

	"memg/llm"
	"memg/store"
)

// SaveExchange persists user messages and the assistant response to the
// conversation log. If no active conversation exists for the session, one is
// created.
func SaveExchange(
	ctx context.Context,
	repo store.Repository,
	sessionUUID, entityUUID string,
	input []*llm.Message,
	resp *llm.Response,
) error {
	if sessionUUID == "" {
		return nil
	}

	conv, err := repo.ActiveConversation(ctx, sessionUUID)
	if err != nil {
		return fmt.Errorf("save exchange: active conversation: %w", err)
	}

	convID := ""
	if conv != nil {
		convID = conv.UUID
	} else {
		convID, err = repo.StartConversation(ctx, sessionUUID, entityUUID)
		if err != nil {
			return fmt.Errorf("save exchange: start conversation: %w", err)
		}
	}

	input = NormalizeConversationMessages(input)
	if convID != "" && len(input) > 0 {
		input, err = DiffIncomingMessages(ctx, repo, convID, input)
		if err != nil {
			return fmt.Errorf("save exchange: diff messages: %w", err)
		}
	}

	for _, m := range input {
		if err := repo.AppendMessage(ctx, convID, &store.Message{
			Role:    m.Role,
			Content: m.Content,
			Kind:    "text",
		}); err != nil {
			return fmt.Errorf("save exchange: append user message: %w", err)
		}
	}

	if resp != nil && strings.TrimSpace(resp.Content) != "" {
		if err := repo.AppendMessage(ctx, convID, &store.Message{
			Role:    resp.Role,
			Content: strings.TrimSpace(resp.Content),
			Kind:    "text",
		}); err != nil {
			return fmt.Errorf("save exchange: append assistant message: %w", err)
		}
	}
	return nil
}

// LoadHistory retrieves past conversation messages for the given session.
func LoadHistory(
	ctx context.Context,
	repo store.Repository,
	sessionUUID string,
) ([]*llm.Message, error) {
	conv, err := repo.ActiveConversation(ctx, sessionUUID)
	if err != nil || conv == nil {
		return nil, err
	}

	msgs, err := repo.ReadMessages(ctx, conv.UUID)
	if err != nil {
		return nil, err
	}

	out := make([]*llm.Message, len(msgs))
	for i, m := range msgs {
		out[i] = &llm.Message{Role: m.Role, Content: m.Content}
	}
	return NormalizeConversationMessages(out), nil
}

// DiffIncomingMessages returns only messages not already in the conversation.
// It uses sequence-based tail comparison instead of set-membership dedup so
// that legitimately repeated messages (same role+content) are not filtered out.
func DiffIncomingMessages(
	ctx context.Context,
	repo store.Repository,
	conversationUUID string,
	incoming []*llm.Message,
) ([]*llm.Message, error) {
	incoming = NormalizeConversationMessages(incoming)

	existing, err := repo.ReadMessages(ctx, conversationUUID)
	if err != nil {
		return nil, err
	}

	normalizedExisting := make([]*llm.Message, 0, len(existing))
	for _, msg := range existing {
		normalizedExisting = append(normalizedExisting, &llm.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	existingMsgs := NormalizeConversationMessages(normalizedExisting)

	if len(existingMsgs) == 0 {
		return incoming, nil
	}

	bestOverlap := overlapLength(existingMsgs, incoming)

	// Return messages after the overlap.
	if bestOverlap >= len(incoming) {
		return nil, nil // All messages already exist.
	}
	return incoming[bestOverlap:], nil
}

// LoadRecentHistory retrieves at most the last `maxTurns` messages from the
// active conversation for the given session. This prevents unbounded history
// growth when sessions are long-lived. When maxTurns > 0, the limiting is
// done in SQL to avoid loading all messages into memory.
func LoadRecentHistory(
	ctx context.Context,
	repo store.Repository,
	sessionUUID string,
	maxTurns int,
) ([]*llm.Message, error) {
	conv, err := repo.ActiveConversation(ctx, sessionUUID)
	if err != nil || conv == nil {
		return nil, err
	}

	var msgs []*store.Message
	if maxTurns > 0 {
		msgs, err = repo.ReadRecentMessages(ctx, conv.UUID, maxTurns)
	} else {
		msgs, err = repo.ReadMessages(ctx, conv.UUID)
	}
	if err != nil {
		return nil, err
	}

	out := make([]*llm.Message, len(msgs))
	for i, m := range msgs {
		out[i] = &llm.Message{Role: m.Role, Content: m.Content}
	}
	return NormalizeConversationMessages(out), nil
}

// NormalizeConversationMessages drops non-conversation roles (for example
// system prompts), trims whitespace, and returns only user/assistant turns.
func NormalizeConversationMessages(messages []*llm.Message) []*llm.Message {
	out := make([]*llm.Message, 0, len(messages))
	for _, msg := range messages {
		if msg == nil {
			continue
		}
		role := strings.TrimSpace(msg.Role)
		if role != llm.RoleUser && role != llm.RoleAssistant {
			continue
		}
		content := strings.TrimSpace(msg.Content)
		if content == "" {
			continue
		}
		out = append(out, &llm.Message{
			Role:    role,
			Content: content,
		})
	}
	return out
}

// MergeHistory prepends only the missing prefix of stored history so that a
// caller can combine persisted working memory with the latest incoming request
// without duplicating turns the client already replayed.
func MergeHistory(history, incoming []*llm.Message) []*llm.Message {
	history = NormalizeConversationMessages(history)
	incoming = NormalizeConversationMessages(incoming)

	if len(history) == 0 {
		return incoming
	}
	if len(incoming) == 0 {
		return history
	}

	overlap := overlapLength(history, incoming)
	merged := make([]*llm.Message, 0, len(history)+len(incoming)-overlap)
	merged = append(merged, history...)
	merged = append(merged, incoming[overlap:]...)
	return merged
}

// MissingHistory returns only the stored history turns that are absent from
// the incoming request.
func MissingHistory(history, incoming []*llm.Message) []*llm.Message {
	history = NormalizeConversationMessages(history)
	incoming = NormalizeConversationMessages(incoming)
	if len(history) == 0 {
		return nil
	}
	overlap := overlapLength(history, incoming)
	if overlap >= len(history) {
		return nil
	}
	missing := make([]*llm.Message, len(history)-overlap)
	copy(missing, history[:len(history)-overlap])
	return missing
}

func overlapLength(existing, incoming []*llm.Message) int {
	existingLen := len(existing)
	maxOverlap := len(incoming)
	if maxOverlap > existingLen {
		maxOverlap = existingLen
	}
	for overlap := maxOverlap; overlap > 0; overlap-- {
		match := true
		for i := 0; i < overlap; i++ {
			e := existing[existingLen-overlap+i]
			m := incoming[i]
			if e.Role != m.Role || e.Content != m.Content {
				match = false
				break
			}
		}
		if match {
			return overlap
		}
	}
	return 0
}
