package proxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"memg/llm"
	"memg/memory"
)

// openaiRequest is used for partial parsing of OpenAI chat completions requests.
type openaiRequest struct {
	Messages []json.RawMessage `json:"messages"`
	Stream   bool              `json:"stream"`
}

// openaiMessage represents a single message in the OpenAI messages array.
type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openaiStreamChunk represents one chunk of an OpenAI streaming response.
type openaiStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

// openaiResponse represents a non-streaming OpenAI chat completions response.
type openaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// OpenAIFormat handles the OpenAI chat completions wire format.
type OpenAIFormat struct{}

func (f *OpenAIFormat) Name() string { return "openai" }

// ParseRequest extracts messages and the last user text from an OpenAI request body.
func (f *OpenAIFormat) ParseRequest(body []byte) (*ParsedRequest, error) {
	var req openaiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("openai: parse request: %w", err)
	}

	parsed := &ParsedRequest{}
	for _, raw := range req.Messages {
		var msg openaiMessage
		if err := json.Unmarshal(raw, &msg); err != nil {
			continue
		}
		parsed.Messages = append(parsed.Messages, &llm.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
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

// InjectHistory prepends only the missing persisted turns before the current
// request messages, keeping the original message payloads intact.
func (f *OpenAIFormat) InjectHistory(body []byte, history []*llm.Message) ([]byte, error) {
	if len(history) == 0 {
		return body, nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("openai: inject history: %w", err)
	}

	messagesRaw, ok := raw["messages"]
	if !ok {
		return nil, fmt.Errorf("openai: inject history: no messages field")
	}

	var messages []json.RawMessage
	if err := json.Unmarshal(messagesRaw, &messages); err != nil {
		return nil, fmt.Errorf("openai: inject history: parse messages: %w", err)
	}

	current := make([]*llm.Message, 0, len(messages))
	insertAt := len(messages)
	for i, rawMsg := range messages {
		var msg openaiMessage
		if err := json.Unmarshal(rawMsg, &msg); err != nil {
			if insertAt == len(messages) {
				insertAt = i
			}
			continue
		}
		if msg.Role == llm.RoleSystem {
			continue
		}
		if insertAt == len(messages) {
			insertAt = i
		}
		current = append(current, &llm.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	missing := memory.MissingHistory(history, current)
	if len(missing) == 0 {
		return body, nil
	}

	injected := make([]json.RawMessage, 0, len(missing))
	for _, msg := range missing {
		rawMsg, err := json.Marshal(openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
		if err != nil {
			return nil, fmt.Errorf("openai: inject history: marshal message: %w", err)
		}
		injected = append(injected, rawMsg)
	}

	updated := make([]json.RawMessage, 0, len(messages)+len(injected))
	updated = append(updated, messages[:insertAt]...)
	updated = append(updated, injected...)
	updated = append(updated, messages[insertAt:]...)

	encoded, err := json.Marshal(updated)
	if err != nil {
		return nil, fmt.Errorf("openai: inject history: marshal messages: %w", err)
	}
	raw["messages"] = encoded
	return json.Marshal(raw)
}

// InjectContext prepends memory context to the system message. If no system
// message exists, one is inserted at position 0. All unknown fields in the
// request body are preserved.
func (f *OpenAIFormat) InjectContext(body []byte, contextText string) ([]byte, error) {
	if contextText == "" {
		return body, nil
	}

	// Use a map to preserve all unknown fields.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("openai: inject context: %w", err)
	}

	messagesRaw, ok := raw["messages"]
	if !ok {
		return nil, fmt.Errorf("openai: inject context: no messages field")
	}

	var messages []json.RawMessage
	if err := json.Unmarshal(messagesRaw, &messages); err != nil {
		return nil, fmt.Errorf("openai: inject context: parse messages: %w", err)
	}

	// Find existing system message.
	systemIdx := -1
	for i, m := range messages {
		var msg openaiMessage
		if err := json.Unmarshal(m, &msg); err != nil {
			continue
		}
		if msg.Role == llm.RoleSystem {
			systemIdx = i
			break
		}
	}

	if systemIdx >= 0 {
		// Prepend context to existing system message content, preserving any
		// extra fields the message might carry.
		var msgMap map[string]json.RawMessage
		if err := json.Unmarshal(messages[systemIdx], &msgMap); err != nil {
			return nil, fmt.Errorf("openai: inject context: parse system msg: %w", err)
		}
		var existingContent string
		if contentRaw, ok := msgMap["content"]; ok {
			_ = json.Unmarshal(contentRaw, &existingContent)
		}
		newContent := contextText + "\n\n" + existingContent
		contentBytes, _ := json.Marshal(newContent)
		msgMap["content"] = contentBytes
		updatedMsg, err := json.Marshal(msgMap)
		if err != nil {
			return nil, fmt.Errorf("openai: inject context: marshal system msg: %w", err)
		}
		messages[systemIdx] = updatedMsg
	} else {
		// Insert a new system message at position 0.
		sysMsg := openaiMessage{
			Role:    llm.RoleSystem,
			Content: contextText,
		}
		sysMsgBytes, err := json.Marshal(sysMsg)
		if err != nil {
			return nil, fmt.Errorf("openai: inject context: marshal new system msg: %w", err)
		}
		messages = append([]json.RawMessage{sysMsgBytes}, messages...)
	}

	updatedMessages, err := json.Marshal(messages)
	if err != nil {
		return nil, fmt.Errorf("openai: inject context: marshal messages: %w", err)
	}
	raw["messages"] = updatedMessages

	return json.Marshal(raw)
}

// IsStreaming returns true if the request has "stream":true.
func (f *OpenAIFormat) IsStreaming(body []byte) bool {
	var req openaiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return req.Stream
}

// AccumulateStreamData parses an OpenAI SSE data line and extracts the content delta.
// OpenAI does not use event types — eventType is ignored.
func (f *OpenAIFormat) AccumulateStreamData(eventType, data string) (string, bool) {
	data = strings.TrimSpace(data)
	if data == "[DONE]" {
		return "", true
	}

	var chunk openaiStreamChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return "", false
	}

	if len(chunk.Choices) == 0 {
		return "", false
	}

	choice := chunk.Choices[0]
	delta := choice.Delta.Content
	done := choice.FinishReason != nil && *choice.FinishReason != ""

	return delta, done
}

// ExtractResponseContent extracts the assistant's message content from a
// non-streaming OpenAI response.
func (f *OpenAIFormat) ExtractResponseContent(body []byte) (string, error) {
	var resp openaiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", fmt.Errorf("openai: extract response: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", nil
	}
	return resp.Choices[0].Message.Content, nil
}
