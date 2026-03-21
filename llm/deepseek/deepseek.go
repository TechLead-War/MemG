// Package deepseek provides an LLM provider for the DeepSeek API.
package deepseek

import (
	"memg/llm"
	"memg/llm/openaicompat"
)

const (
	defaultBaseURL = "https://api.deepseek.com/v1"
	defaultModel   = "deepseek-chat"
	envAPIKey      = "DEEPSEEK_API_KEY"
	providerName   = "deepseek"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// New creates a DeepSeek provider from the given config.
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
