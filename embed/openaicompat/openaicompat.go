// Package openaicompat provides a shared embedding client for any
// OpenAI-compatible embedding API (OpenAI, Together AI, Voyage AI, Azure, etc.).
package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const maxBatchSize = 2048

// Client implements embed.Embedder for any OpenAI-compatible embedding API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	dimension  int
	authHeader string // header name, default "Authorization"
	authPrefix string // prefix, default "Bearer "
	httpClient *http.Client
}

// Option configures optional Client behaviour.
type Option func(*Client)

// WithAuthHeader customises the authentication header name and value prefix.
// For example, Azure OpenAI uses header "api-key" with no prefix.
func WithAuthHeader(name, prefix string) Option {
	return func(c *Client) {
		c.authHeader = name
		c.authPrefix = prefix
	}
}

// WithHTTPClient sets a custom HTTP client for the embedding requests.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.httpClient = hc
	}
}

// New creates a Client configured for the given endpoint.
func New(apiKey, model string, dimension int, baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		model:      model,
		dimension:  dimension,
		authHeader: "Authorization",
		authPrefix: "Bearer ",
		httpClient: http.DefaultClient,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// ModelName returns the configured embedding model identifier.
func (c *Client) ModelName() string { return c.model }

// embeddingRequest is the JSON body sent to the API.
type embeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// embeddingResponse is the JSON response from the API.
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// Embed produces one embedding vector per input text. It handles batching
// internally, splitting inputs into groups of up to 2048 texts.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))

	for start := 0; start < len(texts); start += maxBatchSize {
		end := start + maxBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[start:end]

		vecs, err := c.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}

		for i, v := range vecs {
			results[start+i] = v
		}
	}

	return results, nil
}

func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(embeddingRequest{
		Input: texts,
		Model: c.model,
	})
	if err != nil {
		return nil, fmt.Errorf("openaicompat: marshal request: %w", err)
	}

	url := c.baseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set(c.authHeader, c.authPrefix+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openaicompat: status %d: %s", resp.StatusCode, raw)
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("openaicompat: decode response: %w", err)
	}

	// The API may return results out of order; re-order by index.
	vecs := make([][]float32, len(texts))
	for _, d := range result.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("openaicompat: response index %d out of range [0, %d)", d.Index, len(vecs))
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
