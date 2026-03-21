package proxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"memg/llm"
	"memg/memory"
)

// anthropicMessage represents a single message in the Anthropic messages array.
type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicContentDelta is the payload of a content_block_delta SSE event.
type anthropicContentDelta struct {
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

// anthropicMessageDelta is the payload of a message_delta SSE event.
type anthropicMessageDelta struct {
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
}

// anthropicResponse represents a non-streaming Anthropic messages response.
type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

// AnthropicFormat handles the Anthropic messages API wire format.
type AnthropicFormat struct{}

func (f *AnthropicFormat) Name() string { return "anthropic" }

// ParseRequest extracts the system prompt and messages from an Anthropic request body.
func (f *AnthropicFormat) ParseRequest(body []byte) (*ParsedRequest, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("anthropic: parse request: %w", err)
	}

	parsed := &ParsedRequest{}

	// Parse top-level system prompt.
	if sysRaw, ok := raw["system"]; ok {
		var systemText string
		if err := json.Unmarshal(sysRaw, &systemText); err == nil && systemText != "" {
			parsed.Messages = append(parsed.Messages, &llm.Message{
				Role:    llm.RoleSystem,
				Content: systemText,
			})
		}
	}

	// Parse messages array.
	if msgsRaw, ok := raw["messages"]; ok {
		var msgs []anthropicMessage
		if err := json.Unmarshal(msgsRaw, &msgs); err != nil {
			return nil, fmt.Errorf("anthropic: parse messages: %w", err)
		}
		for _, m := range msgs {
			parsed.Messages = append(parsed.Messages, &llm.Message{
				Role:    m.Role,
				Content: m.Content,
			})
		}
	}

	// Find the last user message.
	for i := len(parsed.Messages) - 1; i >= 0; i-- {
		if parsed.Messages[i].Role == llm.RoleUser {
			parsed.LastUserText = parsed.Messages[i].Content
			break
		}
	}

	return parsed, nil
}

// InjectHistory prepends only the missing persisted turns to the Anthropic
// messages array.
func (f *AnthropicFormat) InjectHistory(body []byte, history []*llm.Message) ([]byte, error) {
	if len(history) == 0 {
		return body, nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("anthropic: inject history: %w", err)
	}

	msgsRaw, ok := raw["messages"]
	if !ok {
		return nil, fmt.Errorf("anthropic: inject history: no messages field")
	}

	var msgs []anthropicMessage
	if err := json.Unmarshal(msgsRaw, &msgs); err != nil {
		return nil, fmt.Errorf("anthropic: inject history: parse messages: %w", err)
	}

	current := make([]*llm.Message, 0, len(msgs))
	for _, msg := range msgs {
		current = append(current, &llm.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	missing := memory.MissingHistory(history, current)
	if len(missing) == 0 {
		return body, nil
	}

	updated := make([]anthropicMessage, 0, len(msgs)+len(missing))
	for _, msg := range missing {
		updated = append(updated, anthropicMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	updated = append(updated, msgs...)

	encoded, err := json.Marshal(updated)
	if err != nil {
		return nil, fmt.Errorf("anthropic: inject history: marshal messages: %w", err)
	}
	raw["messages"] = encoded
	return json.Marshal(raw)
}

// InjectContext modifies or inserts the top-level "system" field in the
// Anthropic request body. All unknown fields are preserved.
func (f *AnthropicFormat) InjectContext(body []byte, contextText string) ([]byte, error) {
	if contextText == "" {
		return body, nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("anthropic: inject context: %w", err)
	}

	var existingSystem string
	if sysRaw, ok := raw["system"]; ok {
		_ = json.Unmarshal(sysRaw, &existingSystem)
	}

	var newSystem string
	if existingSystem != "" {
		newSystem = contextText + "\n\n" + existingSystem
	} else {
		newSystem = contextText
	}

	sysBytes, err := json.Marshal(newSystem)
	if err != nil {
		return nil, fmt.Errorf("anthropic: inject context: marshal system: %w", err)
	}
	raw["system"] = sysBytes

	return json.Marshal(raw)
}

// IsStreaming returns true if the request has "stream":true.
func (f *AnthropicFormat) IsStreaming(body []byte) bool {
	var partial struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &partial); err != nil {
		return false
	}
	return partial.Stream
}

// AccumulateStreamData parses an Anthropic SSE event and extracts content deltas.
// Anthropic uses typed events: content_block_delta for text chunks,
// message_delta and message_stop for completion signals.
func (f *AnthropicFormat) AccumulateStreamData(eventType, data string) (string, bool) {
	data = strings.TrimSpace(data)

	switch eventType {
	case "content_block_delta":
		var cd anthropicContentDelta
		if err := json.Unmarshal([]byte(data), &cd); err != nil {
			return "", false
		}
		if cd.Delta.Type == "text_delta" {
			return cd.Delta.Text, false
		}
		return "", false

	case "message_delta":
		var md anthropicMessageDelta
		if err := json.Unmarshal([]byte(data), &md); err != nil {
			return "", false
		}
		if md.Delta.StopReason != "" {
			return "", true
		}
		return "", false

	case "message_stop":
		return "", true

	default:
		return "", false
	}
}

// ExtractResponseContent extracts the text content from a non-streaming
// Anthropic response.
func (f *AnthropicFormat) ExtractResponseContent(body []byte) (string, error) {
	var resp anthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", fmt.Errorf("anthropic: extract response: %w", err)
	}
	for _, block := range resp.Content {
		if block.Type == "text" {
			return block.Text, nil
		}
	}
	return "", nil
}
