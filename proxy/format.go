// Package proxy implements a transparent reverse proxy that intercepts LLM API
// calls, augments them with recalled memory context, and captures responses
// for asynchronous knowledge extraction.
package proxy

import (
	"memg/llm"
	"strings"
)

// Format abstracts differences between LLM API wire formats.
type Format interface {
	Name() string
	ParseRequest(body []byte) (*ParsedRequest, error)
	InjectHistory(body []byte, history []*llm.Message) ([]byte, error)
	InjectContext(body []byte, contextText string) ([]byte, error)
	IsStreaming(body []byte) bool
	AccumulateStreamData(eventType, data string) (delta string, done bool)
	ExtractResponseContent(body []byte) (string, error)
}

// ParsedRequest holds the parsed contents of an incoming LLM request.
type ParsedRequest struct {
	Messages     []*llm.Message
	LastUserText string
}

// DetectFormat returns the appropriate format based on URL path.
// Returns nil for non-chat endpoints (passthrough).
func DetectFormat(path string) Format {
	if strings.Contains(path, "/chat/completions") {
		return &OpenAIFormat{}
	}
	if strings.Contains(path, "/v1/messages") {
		return &AnthropicFormat{}
	}
	if strings.Contains(path, ":streamGenerateContent") {
		return &GeminiFormat{streaming: true}
	}
	if strings.Contains(path, ":generateContent") {
		return &GeminiFormat{streaming: false}
	}
	return nil
}
