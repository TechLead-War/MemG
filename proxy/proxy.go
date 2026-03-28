package proxy

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/memory"
	"memg/memory/augment"
	"memg/search"
	"memg/store"
)

// Config holds everything the proxy server needs to operate.
type Config struct {
	Target                 *url.URL          // upstream LLM API base URL
	Repo                   store.Repository  // persistence layer
	Embedder               embed.Embedder    // text embedding service
	Provider               llm.Provider      // LLM provider for extraction calls
	Pipeline               *augment.Pipeline // asynchronous knowledge extraction
	DefaultEntity          string            // entity ID for single-entity mode
	SessionTimeout         time.Duration     // sliding session expiry
	WorkingMemoryTurns     int               // max recent turns for working memory
	MemoryTokenBudget      int               // max token budget for merged context
	SummaryTokenBudget     int               // max tokens for summary text in context
	ConsciousMode          bool              // inject top facts by significance on every request
	ConsciousLimit         int               // max facts for conscious mode (default 10)
	RecallFactsLimit       int               // max facts per recall query (default 20)
	RecallSummaryLimit     int               // max summaries per recall query (default 5)
	RecallFactThreshold    float64           // min score for fact recall (default 0.3)
	RecallSummaryThreshold float64           // min score for summary recall (default 0.3)
	MaxRecallCandidates    int               // safety cap on facts loaded per recall pass (default 500)
	ConsciousCacheTTL      time.Duration     // conscious cache entry lifetime (default 30s)
	Debug                  bool              // enable verbose logging
}

// Server is a transparent reverse proxy that intercepts LLM API traffic,
// augments requests with recalled memory, and captures responses for
// asynchronous knowledge extraction.
type Server struct {
	cfg            Config
	engine         search.Engine
	entities       *entityCache
	consciousCache *memory.ConsciousCache
	httpServer     *http.Server
	httpClient     *http.Client
	bgSem          chan struct{}
	baseCtx        context.Context
	cancelBase     context.CancelFunc
}

// NewServer creates a proxy server with the given configuration.
func NewServer(cfg Config) *Server {
	ttl := cfg.ConsciousCacheTTL
	if ttl <= 0 {
		ttl = 30 * time.Second
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &Server{
		cfg:            cfg,
		engine:         search.NewHybrid(),
		entities:       newEntityCache(cfg.Repo),
		consciousCache: memory.NewConsciousCache(ttl),
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
		bgSem:      make(chan struct{}, 64),
		baseCtx:    ctx,
		cancelBase: cancel,
	}
}

// Start binds the HTTP server to the given address and begins serving.
// It blocks until the server is shut down or encounters a fatal error.
func (s *Server) Start(addr string) error {
	if s.cfg.Repo == nil {
		return fmt.Errorf("memg proxy: repository is required")
	}
	if s.cfg.Embedder == nil {
		return fmt.Errorf("memg proxy: embedder is required")
	}
	if s.cfg.Provider == nil {
		return fmt.Errorf("memg proxy: provider is required")
	}
	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: s,
	}
	fmt.Printf("memg proxy: listening on %s → %s\n", addr, s.cfg.Target.String())
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully stops the HTTP server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.cancelBase != nil {
		defer s.cancelBase()
	}
	if s.httpServer == nil {
		return nil
	}
	return s.httpServer.Shutdown(ctx)
}
