// Package ollama provides an embedding provider backed by a local Ollama instance.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"memg/embed"
)

const (
	defaultBaseURL = "http://localhost:11434"
	providerName   = "ollama"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Client implements embed.Embedder for the Ollama embedding API.
type Client struct {
	baseURL    string
	model      string
	dimension  int
	httpClient *http.Client
}

// New creates an Ollama embedding client from the given config.
// Model and Dimension must be provided by the caller.
func New(cfg embed.ProviderConfig) (*Client, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("ollama: Model is required")
	}
	if cfg.Dimension == 0 {
		return nil, fmt.Errorf("ollama: Dimension is required")
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	httpClient := http.DefaultClient
	if cfg.HTTPClient != nil {
		httpClient = cfg.HTTPClient
	}

	return &Client{
		baseURL:    baseURL,
		model:      cfg.Model,
		dimension:  cfg.Dimension,
		httpClient: httpClient,
	}, nil
}

// ModelName returns the configured embedding model identifier.
func (c *Client) ModelName() string { return c.model }

// ollamaRequest is the JSON body sent to the Ollama embed endpoint.
type ollamaRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ollamaResponse is the JSON response from the Ollama embed endpoint.
type ollamaResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed produces one embedding vector per input text.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(ollamaRequest{
		Model: c.model,
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := c.baseURL + "/api/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama: status %d: %s", resp.StatusCode, raw)
	}

	var result ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}

	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama: expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}

	return result.Embeddings, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
