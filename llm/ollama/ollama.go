// Package ollama provides an LLM provider for a local Ollama instance.
package ollama

import (
	"memg/llm"
	"memg/llm/openaicompat"
)

const (
	defaultBaseURL = "http://localhost:11434/v1"
	providerName   = "ollama"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// New creates an Ollama provider from the given config.
// No API key is required. The user must specify a model via config.
func New(cfg llm.ProviderConfig) (*openaicompat.Client, error) {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	var opts []openaicompat.Option
	if cfg.HTTPClient != nil {
		opts = append(opts, openaicompat.WithHTTPClient(cfg.HTTPClient))
	}
	return openaicompat.New("", cfg.Model, baseURL, opts...), nil
}
