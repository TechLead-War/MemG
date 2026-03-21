// Package llm defines the interface between MemG and language model providers.
package llm

import "context"

// Provider abstracts communication with a language model service.
type Provider interface {
	// Chat sends a request and returns the complete response.
	Chat(ctx context.Context, req *Request) (*Response, error)
	// Stream sends a request and returns an incremental reader.
	Stream(ctx context.Context, req *Request) (*StreamReader, error)
}

// CallOption modifies a single LLM request.
type CallOption func(*Request)

// WithModel overrides the model for a single call.
func WithModel(model string) CallOption {
	return func(r *Request) { r.Model = model }
}

// WithMaxTokens caps the response length.
func WithMaxTokens(n int) CallOption {
	return func(r *Request) { r.MaxTokens = n }
}

// WithTemperature adjusts randomness of the response.
func WithTemperature(t float64) CallOption {
	return func(r *Request) { r.Temperature = &t }
}
