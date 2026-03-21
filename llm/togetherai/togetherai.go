// Package togetherai provides an LLM provider for the Together AI API.
package togetherai

import (
	"memg/llm"
	"memg/llm/openaicompat"
)

const (
	defaultBaseURL = "https://api.together.xyz/v1"
	defaultModel   = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
	envAPIKey      = "TOGETHER_API_KEY"
	providerName   = "togetherai"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// New creates a Together AI provider from the given config.
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
