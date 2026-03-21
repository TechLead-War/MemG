package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// RemoteEmbedder calls an external HTTP embedding service (e.g. TEI or an
// OpenAI-compatible endpoint).
type RemoteEmbedder struct {
	url       string
	dimension int
	client    *http.Client
}

// NewRemote creates an embedder that sends texts to the given URL.
// The dimension parameter must match the model's output vector length.
func NewRemote(url string, dimension int) *RemoteEmbedder {
	return &RemoteEmbedder{
		url:       url,
		dimension: dimension,
		client:    http.DefaultClient,
	}
}

type embedPayload struct {
	Inputs []string `json:"inputs"`
}

// Embed sends texts to the remote service and returns the vectors.
func (e *RemoteEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(embedPayload{Inputs: texts})
	if err != nil {
		return nil, fmt.Errorf("remote embed: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("remote embed: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("remote embed: send: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("remote embed: status %d: %s", resp.StatusCode, raw)
	}

	var vectors [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&vectors); err != nil {
		return nil, fmt.Errorf("remote embed: decode: %w", err)
	}
	return vectors, nil
}

// Dimension returns the configured vector length.
func (e *RemoteEmbedder) Dimension() int { return e.dimension }

// ModelName is empty for generic remote embedders because the backing service
// does not expose a stable model identifier through this interface.
func (e *RemoteEmbedder) ModelName() string { return "" }
