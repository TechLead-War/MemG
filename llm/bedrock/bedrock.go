// Package bedrock provides an LLM provider for AWS Bedrock.
package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"memg/llm"
)

const (
	defaultModel  = "anthropic.claude-sonnet-4-20250514-v1:0"
	defaultRegion = "us-east-1"
	providerName  = "bedrock"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// Client implements llm.Provider for AWS Bedrock.
type Client struct {
	model      string
	region     string
	accessKey  string
	secretKey  string
	httpClient *http.Client
}

// New creates a Bedrock provider from the given config.
func New(cfg llm.ProviderConfig) (*Client, error) {
	accessKey := os.Getenv("AWS_ACCESS_KEY_ID")
	if accessKey == "" {
		return nil, fmt.Errorf("bedrock: AWS_ACCESS_KEY_ID is required")
	}

	secretKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	if secretKey == "" {
		return nil, fmt.Errorf("bedrock: AWS_SECRET_ACCESS_KEY is required")
	}

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	model := cfg.Model
	if model == "" {
		model = defaultModel
	}

	httpClient := http.DefaultClient
	if cfg.HTTPClient != nil {
		httpClient = cfg.HTTPClient
	}

	return &Client{
		model:      model,
		region:     region,
		accessKey:  accessKey,
		secretKey:  secretKey,
		httpClient: httpClient,
	}, nil
}

// ---------- request / response wire types (Anthropic Messages format) ----------

type bedrockRequest struct {
	Model            string        `json:"model,omitempty"`
	MaxTokens        int           `json:"max_tokens"`
	AnthropicVersion string        `json:"anthropic_version"`
	System           string        `json:"system,omitempty"`
	Messages         []msgWire     `json:"messages"`
	Temperature      *float64      `json:"temperature,omitempty"`
}

type msgWire struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type bedrockResponse struct {
	Content    []contentBlock `json:"content"`
	Role       string         `json:"role"`
	StopReason string         `json:"stop_reason"`
	Usage      bedrockUsage   `json:"usage"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type bedrockUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ---------- Chat ----------

// Chat sends a non-streaming request to AWS Bedrock.
func (c *Client) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	model := c.model
	if req.Model != "" {
		model = req.Model
	}

	body, err := c.buildBody(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}

	endpoint := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke",
		c.region, model)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	payloadHash := hashSHA256(body)
	signRequest(httpReq, payloadHash, c.accessKey, c.secretKey, c.region, time.Now())

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var br bedrockResponse
	if err := json.NewDecoder(resp.Body).Decode(&br); err != nil {
		return nil, fmt.Errorf("bedrock: decode response: %w", err)
	}

	content := ""
	if len(br.Content) > 0 {
		content = br.Content[0].Text
	}

	return &llm.Response{
		Content:      content,
		Role:         br.Role,
		FinishReason: br.StopReason,
		Usage: llm.Usage{
			PromptTokens: br.Usage.InputTokens,
			OutputTokens: br.Usage.OutputTokens,
			TotalTokens:  br.Usage.InputTokens + br.Usage.OutputTokens,
		},
	}, nil
}

// ---------- Stream ----------

// Stream is not supported for Bedrock in this version.
func (c *Client) Stream(_ context.Context, _ *llm.Request) (*llm.StreamReader, error) {
	return nil, fmt.Errorf("bedrock: streaming not supported for Bedrock")
}

// ---------- helpers ----------

func (c *Client) buildBody(req *llm.Request) ([]byte, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	msgs := make([]msgWire, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, msgWire{Role: m.Role, Content: m.Content})
	}

	// Filter out system messages from the messages array since Bedrock/Anthropic
	// expects system as a top-level field.
	var filtered []msgWire
	systemParts := []string{}
	if req.System != "" {
		systemParts = append(systemParts, req.System)
	}
	for _, m := range msgs {
		if strings.EqualFold(m.Role, llm.RoleSystem) {
			systemParts = append(systemParts, m.Content)
		} else {
			filtered = append(filtered, m)
		}
	}

	system := strings.Join(systemParts, "\n\n")

	br := bedrockRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        maxTokens,
		System:           system,
		Messages:         filtered,
		Temperature:      req.Temperature,
	}
	return json.Marshal(br)
}
