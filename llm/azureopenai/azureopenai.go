// Package azureopenai provides an LLM provider for the Azure OpenAI Service.
package azureopenai

import (
	"strings"

	"memg/llm"
	"memg/llm/openaicompat"
)

const (
	envAPIKey    = "AZURE_OPENAI_API_KEY"
	providerName = "azureopenai"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// New creates an Azure OpenAI provider from the given config.
// The user must supply BaseURL (including the deployment path) and Model.
func New(cfg llm.ProviderConfig) (*openaicompat.Client, error) {
	apiKey, err := cfg.ResolveAPIKey(envAPIKey)
	if err != nil {
		return nil, err
	}

	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if !strings.Contains(baseURL, "api-version=") {
		sep := "?"
		if strings.Contains(baseURL, "?") {
			sep = "&"
		}
		baseURL += sep + "api-version=2024-02-01"
	}

	opts := []openaicompat.Option{
		openaicompat.WithAuthHeader("api-key", ""),
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, openaicompat.WithHTTPClient(cfg.HTTPClient))
	}
	return openaicompat.New(apiKey, cfg.Model, baseURL, opts...), nil
}
