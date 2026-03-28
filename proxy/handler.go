package proxy

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/memory"
	"memg/memory/augment"
)

// internalHeader is set on LLM calls made by MemG itself (extraction,
// summarization). The proxy skips interception for these requests to
// prevent infinite recursion.
const internalHeader = "X-MemG-Internal"

// ServeHTTP implements http.Handler. It detects the wire format, augments the
// request with recalled memory, forwards it to the upstream LLM, and captures
// the response for background knowledge extraction.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Recursion protection: skip interception for internal MemG calls.
	if r.Header.Get(internalHeader) == "true" {
		s.passthrough(w, r)
		return
	}

	format := DetectFormat(r.URL.Path)
	if format == nil {
		s.passthrough(w, r)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "memg proxy: failed to read request body", http.StatusBadGateway)
		return
	}
	r.Body.Close()

	externalID := s.entities.resolveEntity(r, s.cfg.DefaultEntity)
	if externalID == "" {
		if s.cfg.Debug {
			fmt.Println("memg proxy: no entity resolved, forwarding without memory")
		}
		s.forwardRaw(w, r, body)
		return
	}

	entityUUID, err := s.entities.getOrCreateUUID(r.Context(), externalID)
	if err != nil {
		fmt.Printf("memg proxy: entity resolution failed: %v\n", err)
		// Degrade gracefully — forward without memory.
		s.forwardRaw(w, r, body)
		return
	}

	parsed, err := format.ParseRequest(body)
	if err != nil {
		fmt.Printf("memg proxy: parse request failed (%s): %v\n", format.Name(), err)
		s.forwardRaw(w, r, body)
		return
	}

	// Resolve session with sliding expiry.
	timeout := s.cfg.SessionTimeout
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	sess, isNew, err := s.cfg.Repo.EnsureSession(r.Context(), entityUUID, "", timeout)
	if err != nil {
		fmt.Printf("memg proxy: session resolution failed: %v\n", err)
	}
	sessionUUID := ""
	if sess != nil {
		sessionUUID = sess.UUID
		if isNew {
			s.goBackground(func() { s.summarizeClosedSession(entityUUID, sess.UUID) })
		}
	}

	// Inject stored working memory before forwarding upstream. This lets the
	// proxy recover context even when the client only sends the latest turn.
	modifiedBody := body
	if sessionUUID != "" {
		history, historyErr := memory.LoadRecentHistory(r.Context(), s.cfg.Repo, sessionUUID, s.cfg.WorkingMemoryTurns)
		if historyErr != nil && s.cfg.Debug {
			fmt.Printf("memg proxy: load recent history failed: %v\n", historyErr)
		}
		if len(history) > 0 {
			withHistory, injectErr := format.InjectHistory(modifiedBody, history)
			if injectErr != nil {
				fmt.Printf("memg proxy: inject history failed (%s): %v\n", format.Name(), injectErr)
			} else {
				modifiedBody = withHistory
			}
		}
	}

	contextText := s.recall(r.Context(), entityUUID, parsed.LastUserText)
	if contextText != "" {
		originalBody := modifiedBody
		modifiedBody, err = format.InjectContext(modifiedBody, contextText)
		if err != nil {
			fmt.Printf("memg proxy: inject context failed (%s): %v\n", format.Name(), err)
			modifiedBody = originalBody
		} else if s.cfg.Debug {
			fmt.Printf("memg proxy: injected %d bytes of context (%s)\n", len(contextText), format.Name())
		}
	}

	if format.IsStreaming(body) {
		s.handleStreaming(w, r, modifiedBody, format, entityUUID, sessionUUID, parsed.Messages)
	} else {
		s.handleNonStreaming(w, r, modifiedBody, format, entityUUID, sessionUUID, parsed.Messages)
	}
}

// passthrough forwards the request to the upstream target without any
// modification or interception.
func (s *Server) passthrough(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "memg proxy: failed to read request body", http.StatusBadGateway)
		return
	}
	r.Body.Close()

	resp, err := s.forwardToUpstream(r, body)
	if err != nil {
		http.Error(w, "memg proxy: upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		if s.cfg.Debug {
			fmt.Printf("memg proxy: copy response body: %v\n", err)
		}
	}
}

// forwardRaw sends the original request body to the upstream target and
// writes the response back to the client.
func (s *Server) forwardRaw(w http.ResponseWriter, r *http.Request, body []byte) {
	resp, err := s.forwardToUpstream(r, body)
	if err != nil {
		http.Error(w, "memg proxy: upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		if s.cfg.Debug {
			fmt.Printf("memg proxy: copy response body: %v\n", err)
		}
	}
}

// forwardToUpstream constructs a new HTTP request to the upstream target,
// copying all headers from the original request.
func (s *Server) forwardToUpstream(r *http.Request, body []byte) (*http.Response, error) {
	targetURL := *s.cfg.Target
	targetURL.Path = r.URL.Path
	targetURL.RawQuery = r.URL.RawQuery

	upstream, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("memg proxy: build upstream request: %w", err)
	}

	for key, values := range r.Header {
		for _, v := range values {
			upstream.Header.Add(key, v)
		}
	}
	upstream.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))

	return s.httpClient.Do(upstream)
}

// recall runs both fact and summary recall for the given entity and query,
// then formats the results into a context string suitable for injection.
func (s *Server) recall(ctx context.Context, entityUUID, queryText string) string {
	factLimit := s.cfg.RecallFactsLimit
	if factLimit <= 0 {
		factLimit = 20
	}
	summaryLimit := s.cfg.RecallSummaryLimit
	if summaryLimit <= 0 {
		summaryLimit = 5
	}
	factThreshold := s.cfg.RecallFactThreshold
	if factThreshold <= 0 {
		factThreshold = 0.3
	}
	summaryThreshold := s.cfg.RecallSummaryThreshold
	if summaryThreshold <= 0 {
		summaryThreshold = 0.3
	}

	var ctxInput memory.ContextInput
	ctxInput.Budget = memory.ContextBudget{
		TotalTokens:   s.cfg.MemoryTokenBudget,
		SummaryTokens: s.cfg.SummaryTokenBudget,
	}
	if ctxInput.Budget.TotalTokens <= 0 {
		ctxInput.Budget.TotalTokens = 4000
	}
	if ctxInput.Budget.SummaryTokens <= 0 {
		ctxInput.Budget.SummaryTokens = 1000
	}

	// Conscious mode (cached).
	if s.cfg.ConsciousMode {
		limit := s.cfg.ConsciousLimit
		if limit <= 0 {
			limit = 10
		}
		conscious, err := memory.LoadConsciousContextCached(ctx, s.consciousCache, s.cfg.Repo, entityUUID, limit)
		if err != nil {
			fmt.Printf("memg proxy: conscious recall failed: %v\n", err)
		}
		ctxInput.ConsciousFacts = conscious
	}

	// Query-based recall — embed once and reuse.
	if queryText != "" {
		vectors, err := s.cfg.Embedder.Embed(ctx, []string{queryText})
		if err != nil {
			fmt.Printf("memg proxy: embed query failed: %v\n", err)
		} else if len(vectors) > 0 {
			queryVec := vectors[0]
			queryModel := embed.ModelNameOf(s.cfg.Embedder)

			maxCandidates := s.cfg.MaxRecallCandidates
			if maxCandidates <= 0 {
				maxCandidates = 50
			}
			facts, err := memory.RecallWithVector(
				ctx, s.engine, s.cfg.Repo,
				queryVec, queryModel, queryText, entityUUID, factLimit, factThreshold, maxCandidates,
			)
			if err != nil {
				fmt.Printf("memg proxy: fact recall failed: %v\n", err)
			}
			ctxInput.RecalledFacts = facts

			summaries, err := memory.RecallSummariesWithVector(
				ctx, s.engine, s.cfg.Repo,
				queryVec, queryModel, queryText, entityUUID, summaryLimit, summaryThreshold,
			)
			if err != nil {
				fmt.Printf("memg proxy: summary recall failed: %v\n", err)
			}
			ctxInput.Summaries = summaries
		}
	}

	// Track recall usage for facts injected into the prompt.
	if len(ctxInput.RecalledFacts) > 0 {
		s.goBackground(func() { s.trackRecallUsage(ctxInput.RecalledFacts) })
	}

	return memory.BuildContext(ctxInput)
}

// trackRecallUsage updates recall stats for facts that were injected into the prompt.
func (s *Server) trackRecallUsage(facts []*memory.RecalledFact) {
	if len(facts) == 0 {
		return
	}
	ids := make([]string, len(facts))
	for i, f := range facts {
		ids[i] = f.ID
	}
	ctx := s.baseCtx
	if err := s.cfg.Repo.UpdateRecallUsage(ctx, ids); err != nil {
		if s.cfg.Debug {
			fmt.Printf("memg proxy: update recall usage: %v\n", err)
		}
	}
}

// summarizeClosedSession finds the most recent conversation that is NOT for
// the current session and generates a summary for it.
func (s *Server) summarizeClosedSession(entityUUID, currentSessionUUID string) {
	if s.cfg.Provider == nil || s.cfg.Embedder == nil {
		return
	}

	if s.cfg.Debug {
		fmt.Printf("memg proxy: session rolled over for entity %s, new session %s\n", entityUUID, currentSessionUUID)
	}

	ctx := s.baseCtx
	conv, err := s.cfg.Repo.FindUnsummarizedConversation(ctx, entityUUID, currentSessionUUID)
	if err != nil || conv == nil {
		return
	}

	if conv.Summary != "" {
		return
	}

	if err := memory.GenerateAndStoreSummary(ctx, s.cfg.Provider, s.cfg.Embedder, s.cfg.Repo, conv.UUID); err != nil {
		if s.cfg.Debug {
			fmt.Printf("memg proxy: generate summary: %v\n", err)
		}
	}
}

// afterResponse saves the exchange to the conversation log and enqueues a
// pipeline job for background knowledge extraction. This runs asynchronously
// and does not block the response to the client.
func (s *Server) afterResponse(entityUUID, sessionUUID string, messages []*llm.Message, responseContent string) {
	ctx := s.baseCtx

	resp := &llm.Response{
		Role:    llm.RoleAssistant,
		Content: responseContent,
	}

	if sessionUUID != "" {
		if err := memory.SaveExchange(ctx, s.cfg.Repo, sessionUUID, entityUUID, messages, resp); err != nil {
			fmt.Printf("memg proxy: save exchange failed: %v\n", err)
		}
	}

	if s.cfg.Pipeline != nil {
		allMessages := memory.NormalizeConversationMessages(messages)
		if content := strings.TrimSpace(responseContent); content != "" {
			allMessages = append(allMessages, &llm.Message{
				Role:    llm.RoleAssistant,
				Content: content,
			})
		}
		if len(allMessages) == 0 {
			return
		}

		s.cfg.Pipeline.Enqueue(&augment.Job{
			EntityID: entityUUID,
			Messages: allMessages,
		})
	}
}

// goBackground runs fn in a goroutine, bounded by the server's background
// semaphore. If the semaphore is full, fn runs synchronously to prevent
// unbounded goroutine growth.
func (s *Server) goBackground(fn func()) {
	select {
	case s.bgSem <- struct{}{}:
		go func() {
			defer func() { <-s.bgSem }()
			fn()
		}()
	default:
		fn()
	}
}

// copyResponseHeaders copies headers from an upstream response to the client
// response writer.
func copyResponseHeaders(w http.ResponseWriter, resp *http.Response) {
	for key, values := range resp.Header {
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}
}
