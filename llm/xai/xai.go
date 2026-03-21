// Package xai provides an LLM provider for the xAI (Grok) API.
package xai

import (
	"memg/llm"
	"memg/llm/openaicompat"
)

const (
	defaultBaseURL = "https://api.x.ai/v1"
	defaultModel   = "grok-3"
	envAPIKey      = "XAI_API_KEY"
	providerName   = "xai"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// New creates an xAI provider from the given config.
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
