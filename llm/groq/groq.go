// Package groq provides an LLM provider for the Groq API.
package groq

import (
	"memg/llm"
	"memg/llm/openaicompat"
)

const (
	defaultBaseURL = "https://api.groq.com/openai/v1"
	defaultModel   = "llama-3.3-70b-versatile"
	envAPIKey      = "GROQ_API_KEY"
	providerName   = "groq"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// New creates a Groq provider from the given config.
func New(cfg llm.ProviderConfig) (*openaicompat.Client, error) {
	apiKey, err := cfg.ResolveAPIKey(envAPIKey)
	if err != nil {
		return nil, err
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	model := cfg.Model
	if model == "" {
		model = defaultModel
	}

	var opts []openaicompat.Option
	if cfg.HTTPClient != nil {
		opts = append(opts, openaicompat.WithHTTPClient(cfg.HTTPClient))
	}
	return openaicompat.New(apiKey, model, baseURL, opts...), nil
}
