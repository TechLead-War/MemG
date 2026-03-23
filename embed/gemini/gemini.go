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
	maxBatchSize     = 100
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

// ---------- wire types ----------

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

// batchEmbedRequest is the JSON body sent to the batchEmbedContents endpoint.
type batchEmbedRequest struct {
	Requests []embedContentRequest `json:"requests"`
}

type embedContentRequest struct {
	Model   string        `json:"model"`
	Content geminiContent `json:"content"`
}

// batchEmbedResponse is the JSON response from the batchEmbedContents endpoint.
type batchEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

// Embed produces one embedding vector per input text using the Gemini
// batchEmbedContents endpoint, batching up to 100 texts per API call.
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
	modelRef := fmt.Sprintf("models/%s", c.model)

	requests := make([]embedContentRequest, len(texts))
	for i, text := range texts {
		requests[i] = embedContentRequest{
			Model: modelRef,
			Content: geminiContent{
				Parts: []geminiPart{{Text: text}},
			},
		}
	}

	body, err := json.Marshal(batchEmbedRequest{Requests: requests})
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal batch request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/%s:batchEmbedContents?key=%s", c.baseURL, modelRef, c.apiKey)
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

	var result batchEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("gemini: decode batch response: %w", err)
	}

	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("gemini: expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}

	vecs := make([][]float32, len(texts))
	for i, emb := range result.Embeddings {
		vecs[i] = emb.Values
	}
	return vecs, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
