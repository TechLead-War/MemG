package embed

import (
	"fmt"
	"net/http"
	"os"
)

// ProviderConfig holds connection settings for an embedding provider.
type ProviderConfig struct {
	APIKey     string       // explicit key, highest priority
	BaseURL    string       // override endpoint
	Model      string       // model identifier
	Dimension  int          // embedding vector dimension (0 = use provider default)
	HTTPClient *http.Client // optional custom HTTP client (e.g. for adding internal headers)
}

// ResolveAPIKey returns the API key from explicit config or the given env var.
// It returns an error if neither source provides a key.
func (c *ProviderConfig) ResolveAPIKey(envVar string) (string, error) {
	if c.APIKey != "" {
		return c.APIKey, nil
	}
	if v := os.Getenv(envVar); v != "" {
		return v, nil
	}
	return "", fmt.Errorf("API key required: set config APIKey or env var %s", envVar)
}
