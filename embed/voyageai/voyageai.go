// Package voyageai provides an embedding provider backed by the Voyage AI API.
package voyageai

import (
	"memg/embed"
	"memg/embed/openaicompat"
)

const (
	defaultBaseURL   = "https://api.voyageai.com/v1"
	defaultModel     = "voyage-3"
	defaultDimension = 1024
	envVar           = "VOYAGE_API_KEY"
	providerName     = "voyageai"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// New creates a Voyage AI embedding client from the given config.
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
