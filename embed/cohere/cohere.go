// Package cohere provides an embedding provider backed by the Cohere API.
package cohere

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
	defaultBaseURL   = "https://api.cohere.com"
	defaultModel     = "embed-english-v3.0"
	defaultDimension = 1024
	envVar           = "COHERE_API_KEY"
	providerName     = "cohere"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Client implements embed.Embedder for the Cohere embedding API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	dimension  int
	httpClient *http.Client
}

// New creates a Cohere embedding client from the given config.
func New(cfg embed.ProviderConfig) (*Client, error) {
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

	httpClient := http.DefaultClient
	if cfg.HTTPClient != nil {
		httpClient = cfg.HTTPClient
	}

	return &Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		model:      model,
		dimension:  dimension,
		httpClient: httpClient,
	}, nil
}

// ModelName returns the configured embedding model identifier.
func (c *Client) ModelName() string { return c.model }

// cohereRequest is the JSON body sent to the Cohere embed endpoint.
type cohereRequest struct {
	Texts          []string `json:"texts"`
	Model          string   `json:"model"`
	InputType      string   `json:"input_type"`
	EmbeddingTypes []string `json:"embedding_types"`
}

// cohereResponse is the JSON response from the Cohere embed endpoint.
type cohereResponse struct {
	Embeddings struct {
		Float [][]float32 `json:"float"`
	} `json:"embeddings"`
}

// Embed produces one embedding vector per input text.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(cohereRequest{
		Texts:          texts,
		Model:          c.model,
		InputType:      "search_document",
		EmbeddingTypes: []string{"float"},
	})
	if err != nil {
		return nil, fmt.Errorf("cohere: marshal request: %w", err)
	}

	url := c.baseURL + "/v2/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("cohere: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cohere: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("cohere: status %d: %s", resp.StatusCode, raw)
	}

	var result cohereResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("cohere: decode response: %w", err)
	}

	if len(result.Embeddings.Float) != len(texts) {
		return nil, fmt.Errorf("cohere: expected %d embeddings, got %d", len(texts), len(result.Embeddings.Float))
	}

	return result.Embeddings.Float, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
