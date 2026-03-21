// Package bedrock provides an embedding provider backed by AWS Bedrock.
// It currently supports the Amazon Titan Embed model format.
package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"memg/embed"
)

const (
	defaultModel     = "amazon.titan-embed-text-v2:0"
	defaultDimension = 1024
	defaultRegion    = "us-east-1"
	providerName     = "bedrock"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Client implements embed.Embedder for AWS Bedrock embedding models.
type Client struct {
	accessKey  string
	secretKey  string
	region     string
	model      string
	dimension  int
	httpClient *http.Client
}

// New creates a Bedrock embedding client from the given config.
// AWS credentials are read from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
// environment variables. The region is read from AWS_REGION (default us-east-1).
func New(cfg embed.ProviderConfig) (*Client, error) {
	accessKey := os.Getenv("AWS_ACCESS_KEY_ID")
	if accessKey == "" {
		return nil, fmt.Errorf("bedrock: AWS_ACCESS_KEY_ID environment variable is required")
	}
	secretKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	if secretKey == "" {
		return nil, fmt.Errorf("bedrock: AWS_SECRET_ACCESS_KEY environment variable is required")
	}

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
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
		accessKey:  accessKey,
		secretKey:  secretKey,
		region:     region,
		model:      model,
		dimension:  dimension,
		httpClient: httpClient,
	}, nil
}

// ModelName returns the configured embedding model identifier.
func (c *Client) ModelName() string { return c.model }

// titanRequest is the JSON body sent to the Titan Embed model.
type titanRequest struct {
	InputText string `json:"inputText"`
}

// titanResponse is the JSON response from the Titan Embed model.
type titanResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Embed produces one embedding vector per input text. Bedrock's Titan model
// accepts one text per API call, so this method loops over inputs sequentially.
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
	body, err := json.Marshal(titanRequest{InputText: text})
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke", c.region, c.model)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Sign with AWS Signature V4.
	bodyHash := sha256Hex(body)
	signRequest(req, c.accessKey, c.secretKey, c.region, bodyHash, time.Now())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: status %d: %s", resp.StatusCode, raw)
	}

	var result titanResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("bedrock: decode response: %w", err)
	}
	return result.Embedding, nil
}

// Dimension returns the configured vector length.
func (c *Client) Dimension() int { return c.dimension }
