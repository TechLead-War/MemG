package proxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"memg/llm"
	"memg/memory"
)

// ---------- wire types ----------

type geminiProxyPart struct {
	Text string `json:"text"`
}

type geminiProxyContent struct {
	Role  string            `json:"role"`
	Parts []geminiProxyPart `json:"parts"`
}

type geminiProxySystemInstruction struct {
	Parts []geminiProxyPart `json:"parts"`
}

type geminiProxyResponse struct {
	Candidates []struct {
		Content struct {
			Role  string            `json:"role"`
			Parts []geminiProxyPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
}

// ---------- GeminiFormat ----------

// GeminiFormat handles the Google Gemini API wire format.
// The streaming field is set at detection time because Gemini determines
// streaming from the URL path, not a body field.
type GeminiFormat struct {
	streaming bool
}

func (f *GeminiFormat) Name() string { return "gemini" }

// ParseRequest extracts messages and the last user text from a Gemini request body.
func (f *GeminiFormat) ParseRequest(body []byte) (*ParsedRequest, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("gemini: parse request: %w", err)
	}

	parsed := &ParsedRequest{}

	if sysRaw, ok := raw["systemInstruction"]; ok {
		var si geminiProxySystemInstruction
		if err := json.Unmarshal(sysRaw, &si); err == nil && len(si.Parts) > 0 {
			var sb strings.Builder
			for _, p := range si.Parts {
				sb.WriteString(p.Text)
			}
			if text := sb.String(); text != "" {
				parsed.Messages = append(parsed.Messages, &llm.Message{
					Role:    llm.RoleSystem,
					Content: text,
				})
			}
		}
	}

	if contentsRaw, ok := raw["contents"]; ok {
		var contents []geminiProxyContent
		if err := json.Unmarshal(contentsRaw, &contents); err != nil {
			return nil, fmt.Errorf("gemini: parse contents: %w", err)
		}
		for _, c := range contents {
			role := c.Role
			if role == "model" {
				role = llm.RoleAssistant
			}
			var sb strings.Builder
			for _, p := range c.Parts {
				sb.WriteString(p.Text)
			}
			parsed.Messages = append(parsed.Messages, &llm.Message{
				Role:    role,
				Content: sb.String(),
			})
		}
	}

	for i := len(parsed.Messages) - 1; i >= 0; i-- {
		if parsed.Messages[i].Role == llm.RoleUser {
			parsed.LastUserText = parsed.Messages[i].Content
			break
		}
	}

	return parsed, nil
}

// InjectHistory prepends only the missing persisted turns to the Gemini
// contents array, maintaining the user/model alternation requirement.
func (f *GeminiFormat) InjectHistory(body []byte, history []*llm.Message) ([]byte, error) {
	if len(history) == 0 {
		return body, nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("gemini: inject history: %w", err)
	}

	contentsRaw, ok := raw["contents"]
	if !ok {
		return nil, fmt.Errorf("gemini: inject history: no contents field")
	}

	var contents []geminiProxyContent
	if err := json.Unmarshal(contentsRaw, &contents); err != nil {
		return nil, fmt.Errorf("gemini: inject history: parse contents: %w", err)
	}

	current := make([]*llm.Message, 0, len(contents))
	for _, c := range contents {
		role := c.Role
		if role == "model" {
			role = llm.RoleAssistant
		}
		var sb strings.Builder
		for _, p := range c.Parts {
			sb.WriteString(p.Text)
		}
		current = append(current, &llm.Message{
			Role:    role,
			Content: sb.String(),
		})
	}

	missing := memory.MissingHistory(history, current)
	if len(missing) == 0 {
		return body, nil
	}

	injected := make([]geminiProxyContent, 0, len(missing))
	for _, msg := range missing {
		role := msg.Role
		if role == llm.RoleAssistant {
			role = "model"
		}
		injected = append(injected, geminiProxyContent{
			Role:  role,
			Parts: []geminiProxyPart{{Text: msg.Content}},
		})
	}

	updated := make([]geminiProxyContent, 0, len(contents)+len(injected))
	updated = append(updated, injected...)
	updated = append(updated, contents...)

	encoded, err := json.Marshal(updated)
	if err != nil {
		return nil, fmt.Errorf("gemini: inject history: marshal contents: %w", err)
	}
	raw["contents"] = encoded
	return json.Marshal(raw)
}

// InjectContext prepends memory context to the systemInstruction field.
// If no systemInstruction exists, one is created.
func (f *GeminiFormat) InjectContext(body []byte, contextText string) ([]byte, error) {
	if contextText == "" {
		return body, nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("gemini: inject context: %w", err)
	}

	var existingText string
	if siRaw, ok := raw["systemInstruction"]; ok {
		var si geminiProxySystemInstruction
		if err := json.Unmarshal(siRaw, &si); err == nil && len(si.Parts) > 0 {
			var sb strings.Builder
			for _, p := range si.Parts {
				sb.WriteString(p.Text)
			}
			existingText = sb.String()
		}
	}

	var newText string
	if existingText != "" {
		newText = contextText + "\n\n" + existingText
	} else {
		newText = contextText
	}

	si := geminiProxySystemInstruction{
		Parts: []geminiProxyPart{{Text: newText}},
	}
	siBytes, err := json.Marshal(si)
	if err != nil {
		return nil, fmt.Errorf("gemini: inject context: marshal systemInstruction: %w", err)
	}
	raw["systemInstruction"] = siBytes

	return json.Marshal(raw)
}

// IsStreaming returns whether this is a streaming request. For Gemini, this is
// determined by the URL path (streamGenerateContent vs generateContent), not
// a body field. The flag is set at detection time in DetectFormat.
func (f *GeminiFormat) IsStreaming(_ []byte) bool {
	return f.streaming
}

// AccumulateStreamData parses a Gemini SSE data line and extracts the content delta.
// Gemini does not use event types — eventType is ignored.
func (f *GeminiFormat) AccumulateStreamData(_, data string) (string, bool) {
	data = strings.TrimSpace(data)
	if data == "" {
		return "", false
	}

	var resp geminiProxyResponse
	if err := json.Unmarshal([]byte(data), &resp); err != nil {
		return "", false
	}

	if len(resp.Candidates) == 0 {
		return "", false
	}

	candidate := resp.Candidates[0]
	var delta string
	if len(candidate.Content.Parts) > 0 {
		delta = candidate.Content.Parts[0].Text
	}

	done := candidate.FinishReason != "" && candidate.FinishReason != "MALFORMED_FUNCTION_CALL"

	return delta, done
}

// ExtractResponseContent extracts the text content from a non-streaming
// Gemini response.
func (f *GeminiFormat) ExtractResponseContent(body []byte) (string, error) {
	var resp geminiProxyResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", fmt.Errorf("gemini: extract response: %w", err)
	}
	if len(resp.Candidates) == 0 {
		return "", nil
	}
	candidate := resp.Candidates[0]
	if len(candidate.Content.Parts) == 0 {
		return "", nil
	}
	return candidate.Content.Parts[0].Text, nil
}
