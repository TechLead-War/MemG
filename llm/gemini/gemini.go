// Package gemini provides an LLM provider for the Google Gemini API.
package gemini

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
	defaultBaseURL = "https://generativelanguage.googleapis.com"
	defaultModel   = "gemini-2.0-flash"
	envAPIKey      = "GEMINI_API_KEY"
	providerName   = "gemini"
)

func init() {
	llm.RegisterProvider(providerName, func(cfg llm.ProviderConfig) (llm.Provider, error) {
		return New(cfg)
	})
}

// Client implements llm.Provider for the Google Gemini API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	httpClient *http.Client
}

// New creates a Gemini provider from the given config.
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

type geminiRequest struct {
	Contents          []geminiContent          `json:"contents"`
	SystemInstruction *geminiSystemInstruction `json:"systemInstruction,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiSystemInstruction struct {
	Parts []geminiPart `json:"parts"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata *geminiUsage      `json:"usageMetadata,omitempty"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// ---------- Chat ----------

// Chat sends a non-streaming request to the Gemini API.
func (c *Client) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	model := c.model
	if req.Model != "" {
		model = req.Model
	}

	body, err := c.buildBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", c.baseURL, model, c.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var gr geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&gr); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}

	if len(gr.Candidates) == 0 {
		return nil, fmt.Errorf("gemini: empty candidates in response")
	}

	candidate := gr.Candidates[0]
	content := ""
	if len(candidate.Content.Parts) > 0 {
		content = candidate.Content.Parts[0].Text
	}

	role := candidate.Content.Role
	if role == "model" {
		role = llm.RoleAssistant
	}

	usage := llm.Usage{}
	if gr.UsageMetadata != nil {
		usage.PromptTokens = gr.UsageMetadata.PromptTokenCount
		usage.OutputTokens = gr.UsageMetadata.CandidatesTokenCount
		usage.TotalTokens = gr.UsageMetadata.TotalTokenCount
	}

	return &llm.Response{
		Content:      content,
		Role:         role,
		FinishReason: candidate.FinishReason,
		Usage:        usage,
	}, nil
}

// ---------- Stream ----------

// Stream sends a streaming request to the Gemini API.
func (c *Client) Stream(ctx context.Context, req *llm.Request) (*llm.StreamReader, error) {
	model := c.model
	if req.Model != "" {
		model = req.Model
	}

	body, err := c.buildBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", c.baseURL, model, c.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("gemini: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan llm.StreamEvent)
	go c.readSSE(resp.Body, ch)
	return llm.NewStreamReader(ch), nil
}

// ---------- helpers ----------

func (c *Client) buildBody(req *llm.Request) ([]byte, error) {
	contents := make([]geminiContent, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := m.Role
		// Gemini uses "model" instead of "assistant".
		if role == llm.RoleAssistant {
			role = "model"
		}
		contents = append(contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: m.Content}},
		})
	}

	gr := geminiRequest{Contents: contents}
	if req.System != "" {
		gr.SystemInstruction = &geminiSystemInstruction{
			Parts: []geminiPart{{Text: req.System}},
		}
	}

	return json.Marshal(gr)
}

func (c *Client) readSSE(body io.ReadCloser, ch chan<- llm.StreamEvent) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()

		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var gr geminiResponse
		if err := json.Unmarshal([]byte(data), &gr); err != nil {
			ch <- llm.StreamEvent{Err: fmt.Errorf("gemini: decode stream chunk: %w", err)}
			return
		}

		if len(gr.Candidates) == 0 {
			continue
		}

		candidate := gr.Candidates[0]
		delta := ""
		if len(candidate.Content.Parts) > 0 {
			delta = candidate.Content.Parts[0].Text
		}

		ev := llm.StreamEvent{Delta: delta}
		if candidate.FinishReason != "" {
			ev.Done = true
			ev.Finish = candidate.FinishReason
		}

		ch <- ev

		if ev.Done {
			return
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- llm.StreamEvent{Err: fmt.Errorf("gemini: read stream: %w", err)}
	}
}
