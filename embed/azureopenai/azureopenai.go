// Package azureopenai provides an embedding provider backed by Azure OpenAI.
package azureopenai

import (
	"fmt"
	"strings"

	"memg/embed"
	"memg/embed/openaicompat"
)

const (
	envVar       = "AZURE_OPENAI_API_KEY"
	providerName = "azureopenai"
	apiVersion   = "2024-02-01"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// New creates an Azure OpenAI embedding client from the given config.
// BaseURL, Model, and Dimension must be provided by the caller.
func New(cfg embed.ProviderConfig) (*openaicompat.Client, error) {
	apiKey, err := cfg.ResolveAPIKey(envVar)
	if err != nil {
		return nil, err
	}

	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("azureopenai: BaseURL is required (e.g. https://{resource}.openai.azure.com/openai/deployments/{deployment})")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("azureopenai: Model is required")
	}
	if cfg.Dimension == 0 {
		return nil, fmt.Errorf("azureopenai: Dimension is required")
	}

	// Append api-version query parameter to the base URL.
	baseURL := cfg.BaseURL
	if strings.Contains(baseURL, "?") {
		baseURL += "&api-version=" + apiVersion
	} else {
		baseURL += "?api-version=" + apiVersion
	}

	opts := []openaicompat.Option{
		openaicompat.WithAuthHeader("api-key", ""),
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, openaicompat.WithHTTPClient(cfg.HTTPClient))
	}
	return openaicompat.New(apiKey, cfg.Model, cfg.Dimension, baseURL, opts...), nil
}
