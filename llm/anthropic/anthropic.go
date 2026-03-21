// Package anthropic provides an LLM provider for the Anthropic Messages API.
package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"memg/llm"
)

const (
	defaultBaseURL = "https://api.anthropic.com"
	defaultModel   = "claude-sonnet-4-20250514"
	envAPIKey      = "ANTHROPIC_API_KEY"
	providerName   = "anthropic"
	apiVersion     = "2023-06-01"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// Client implements llm.Provider for the Anthropic Messages API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	httpClient *http.Client
}

// New creates an Anthropic provider from the given config.
func New(cfg llm.ProviderConfig) (*Client, error) {
	apiKey, err := cfg.ResolveAPIKey(envAPIKey)
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

	httpClient := http.DefaultClient
	if cfg.HTTPClient != nil {
		httpClient = cfg.HTTPClient
	}

	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		model:      model,
		httpClient: httpClient,
	}, nil
}

// ---------- request / response wire types ----------

type messagesRequest struct {
	Model       string        `json:"model"`
	MaxTokens   int           `json:"max_tokens"`
	System      string        `json:"system,omitempty"`
	Messages    []msgWire     `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
}

type msgWire struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type messagesResponse struct {
	Content    []contentBlock `json:"content"`
	Role       string         `json:"role"`
	StopReason string         `json:"stop_reason"`
	Usage      anthropicUsage `json:"usage"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ---------- Chat ----------

// Chat sends a non-streaming message to the Anthropic API.
func (c *Client) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	body, err := c.buildBody(req, false)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var mr messagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&mr); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	content := ""
	if len(mr.Content) > 0 {
		content = mr.Content[0].Text
	}

	return &llm.Response{
		Content:      content,
		Role:         mr.Role,
		FinishReason: mr.StopReason,
		Usage: llm.Usage{
			PromptTokens: mr.Usage.InputTokens,
			OutputTokens: mr.Usage.OutputTokens,
			TotalTokens:  mr.Usage.InputTokens + mr.Usage.OutputTokens,
		},
	}, nil
}

// ---------- Stream ----------

// Stream sends a streaming message to the Anthropic API.
func (c *Client) Stream(ctx context.Context, req *llm.Request) (*llm.StreamReader, error) {
	body, err := c.buildBody(req, true)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropic: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan llm.StreamEvent)
	go c.readSSE(resp.Body, ch)
	return llm.NewStreamReader(ch), nil
}

// ---------- helpers ----------

func (c *Client) buildBody(req *llm.Request, stream bool) ([]byte, error) {
	model := c.model
	if req.Model != "" {
		model = req.Model
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	msgs := make([]msgWire, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, msgWire{Role: m.Role, Content: m.Content})
	}

	mr := messagesRequest{
		Model:       model,
		MaxTokens:   maxTokens,
		System:      req.System,
		Messages:    msgs,
		Stream:      stream,
		Temperature: req.Temperature,
	}
	return json.Marshal(mr)
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", apiVersion)
}

// SSE event types for Anthropic streaming.
type sseContentDelta struct {
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

type sseMessageDelta struct {
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
}

func (c *Client) readSSE(body io.ReadCloser, ch chan<- llm.StreamEvent) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	var eventType string

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines (event boundary).
		if line == "" {
			eventType = ""
			continue
		}

		// Parse event type.
		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		// Parse data.
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		switch eventType {
		case "content_block_delta":
			var cd sseContentDelta
			if err := json.Unmarshal([]byte(data), &cd); err != nil {
				ch <- llm.StreamEvent{Err: fmt.Errorf("anthropic: decode content delta: %w", err)}
				return
			}
			ch <- llm.StreamEvent{Delta: cd.Delta.Text}

		case "message_delta":
			var md sseMessageDelta
			if err := json.Unmarshal([]byte(data), &md); err != nil {
				ch <- llm.StreamEvent{Err: fmt.Errorf("anthropic: decode message delta: %w", err)}
				return
			}
			ch <- llm.StreamEvent{Done: true, Finish: md.Delta.StopReason}
			return

		case "message_stop":
			return

		default:
			// message_start, content_block_start, ping, etc. — ignore.
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- llm.StreamEvent{Err: fmt.Errorf("anthropic: read stream: %w", err)}
	}
}
