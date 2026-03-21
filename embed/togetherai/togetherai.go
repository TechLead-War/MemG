// Package togetherai provides an embedding provider backed by the Together AI API.
package togetherai

import (
	"memg/embed"
	"memg/embed/openaicompat"
)

const (
	defaultBaseURL   = "https://api.together.xyz/v1"
	defaultModel     = "togethercomputer/m2-bert-80M-8k-retrieval"
	defaultDimension = 768
	envVar           = "TOGETHER_API_KEY"
	providerName     = "togetherai"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// New creates a Together AI embedding client from the given config.
func New(cfg embed.ProviderConfig) (*openaicompat.Client, error) {
	apiKey, err := cfg.ResolveAPIKey(envVar)
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
	dimension := cfg.Dimension
	if dimension == 0 {
		dimension = defaultDimension
	}

	var opts []openaicompat.Option
	if cfg.HTTPClient != nil {
		opts = append(opts, openaicompat.WithHTTPClient(cfg.HTTPClient))
	}
	return openaicompat.New(apiKey, model, dimension, baseURL, opts...), nil
}
