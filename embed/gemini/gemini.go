// Package gemini provides an embedding provider backed by the Google Gemini API.
package gemini

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
	defaultBaseURL   = "https://generativelanguage.googleapis.com"
	defaultModel     = "text-embedding-004"
	defaultDimension = 768
	envVar           = "GEMINI_API_KEY"
	providerName     = "gemini"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Client implements embed.Embedder for the Google Gemini embedding API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	dimension  int
	httpClient *http.Client
}

// New creates a Gemini embedding client from the given config.
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

// geminiRequest is the JSON body sent to the Gemini embedding API.
type geminiRequest struct {
	Content geminiContent `json:"content"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

// geminiResponse is the JSON response from the Gemini embedding API.
type geminiResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

// Embed produces one embedding vector per input text. Gemini embeds one text
// per API call, so this method loops over all inputs sequentially.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	for i, text := range texts {
		vec, err := c.embedSingle(ctx, text)
		if err != nil {
			return nil, err
		}
		results[i] = vec
	}
	return results, nil
}

func (c *Client) embedSingle(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(geminiRequest{
		Content: geminiContent{
			Parts: []geminiPart{{Text: text}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:embedContent?key=%s", c.baseURL, c.model, c.apiKey)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: status %d: %s", resp.StatusCode, raw)
	}

	var result geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}
	return result.Embedding.Values, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
