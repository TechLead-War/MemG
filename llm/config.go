package llm

import (
	"fmt"
	"net/http"
	"os"
)

// ProviderConfig holds connection settings for an LLM provider.
type ProviderConfig struct {
	APIKey     string       // explicit key, highest priority
	BaseURL    string       // override endpoint
	Model      string       // model identifier
	HTTPClient *http.Client // optional custom HTTP client (e.g. for adding internal headers)
}

// ResolveAPIKey returns the API key from explicit config or the given env var.
func (c *ProviderConfig) ResolveAPIKey(envVar string) (string, error) {
	if c.APIKey != "" {
		return c.APIKey, nil
	}
	if v := os.Getenv(envVar); v != "" {
		return v, nil
	}
	return "", fmt.Errorf("API key required: set config APIKey or env var %s", envVar)
}
