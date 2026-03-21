// Package openaicompat provides a shared client for OpenAI-compatible chat
// completion APIs. It is used by the openai, deepseek, groq, togetherai, xai,
// ollama, and azureopenai provider packages.
package openaicompat

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

// Client implements llm.Provider for any OpenAI-compatible chat completions API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	authHeader string
	authPrefix string
	httpClient *http.Client
}

// Option configures the Client.
type Option func(*Client)

// WithAuthHeader sets a custom authorization header name and value prefix.
func WithAuthHeader(name, prefix string) Option {
	return func(c *Client) {
		c.authHeader = name
		c.authPrefix = prefix
	}
}

// WithHTTPClient sets a custom http.Client for requests.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.httpClient = hc
	}
}

// New creates a Client for an OpenAI-compatible API.
func New(apiKey, model, baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		model:      model,
		authHeader: "Authorization",
		authPrefix: "Bearer ",
		httpClient: http.DefaultClient,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// ---------- request / response wire types ----------

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

type chatChoice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
}

type streamChoice struct {
	Delta        chatMessage `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ---------- Chat ----------

// Chat sends a non-streaming chat completion request.
func (c *Client) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	body, err := c.buildBody(req, false)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: %w", err)
	}
	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openaicompat: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, fmt.Errorf("openaicompat: decode response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("openaicompat: empty choices in response")
	}

	return &llm.Response{
		Content:      cr.Choices[0].Message.Content,
		Role:         cr.Choices[0].Message.Role,
		FinishReason: cr.Choices[0].FinishReason,
		Usage: llm.Usage{
			PromptTokens: cr.Usage.PromptTokens,
			OutputTokens: cr.Usage.CompletionTokens,
			TotalTokens:  cr.Usage.TotalTokens,
		},
	}, nil
}

// ---------- Stream ----------

// Stream sends a streaming chat completion request.
func (c *Client) Stream(ctx context.Context, req *llm.Request) (*llm.StreamReader, error) {
	body, err := c.buildBody(req, true)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: %w", err)
	}
	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openaicompat: HTTP %d: %s", resp.StatusCode, string(respBody))
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

	temperature := 0.7
	if req.Temperature != nil {
		temperature = *req.Temperature
	}

	msgs := make([]chatMessage, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, chatMessage{Role: llm.RoleSystem, Content: req.System})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, chatMessage{Role: m.Role, Content: m.Content})
	}

	cr := chatRequest{
		Model:       model,
		Messages:    msgs,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Stream:      stream,
	}
	return json.Marshal(cr)
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set(c.authHeader, c.authPrefix+c.apiKey)
	}
}

func (c *Client) readSSE(body io.ReadCloser, ch chan<- llm.StreamEvent) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines and comments.
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			return
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			ch <- llm.StreamEvent{Err: fmt.Errorf("openaicompat: decode stream chunk: %w", err)}
			return
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		ev := llm.StreamEvent{Delta: choice.Delta.Content}

		if choice.FinishReason != nil && *choice.FinishReason != "" {
			ev.Done = true
			ev.Finish = *choice.FinishReason
		}

		ch <- ev

		if ev.Done {
			return
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- llm.StreamEvent{Err: fmt.Errorf("openaicompat: read stream: %w", err)}
	}
}
