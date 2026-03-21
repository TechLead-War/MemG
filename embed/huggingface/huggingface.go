// Package huggingface provides an embedding provider backed by the
// Hugging Face Inference API.
package huggingface

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
	defaultBaseURL   = "https://api-inference.huggingface.co"
	defaultModel     = "sentence-transformers/all-MiniLM-L6-v2"
	defaultDimension = 384
	envVar           = "HF_API_KEY"
	providerName     = "huggingface"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Client implements embed.Embedder for the Hugging Face Inference API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	dimension  int
	httpClient *http.Client
}

// New creates a Hugging Face embedding client from the given config.
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

// hfRequest is the JSON body sent to the Hugging Face feature-extraction endpoint.
type hfRequest struct {
	Inputs []string `json:"inputs"`
}

// Embed produces one embedding vector per input text.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(hfRequest{Inputs: texts})
	if err != nil {
		return nil, fmt.Errorf("huggingface: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/pipeline/feature-extraction/%s", c.baseURL, c.model)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("huggingface: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("huggingface: status %d: %s", resp.StatusCode, raw)
	}

	// Response is a JSON array of arrays: [[0.1, 0.2, ...], [...]]
	var result [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("huggingface: decode response: %w", err)
	}

	if len(result) != len(texts) {
		return nil, fmt.Errorf("huggingface: expected %d embeddings, got %d", len(texts), len(result))
	}

	return result, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
