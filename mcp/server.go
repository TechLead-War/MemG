// Package mcp implements a Model Context Protocol (MCP) server that exposes
// MemG's memory operations as tools for any MCP-compatible agent framework.
//
// The server uses the Streamable HTTP transport: a single HTTP endpoint
// accepts JSON-RPC 2.0 POST requests and responds with JSON or SSE.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"memg/embed"
	"memg/search"
	"memg/store"
)

// Config holds everything the MCP server needs to operate.
type Config struct {
	Repo     store.Repository // persistence layer
	Embedder embed.Embedder   // text embedding service (required for search and add)
	Debug    bool             // enable verbose logging
}

// Server implements an MCP server over Streamable HTTP transport.
type Server struct {
	cfg      Config
	engine   search.Engine
	entities *entityCache
	http     *http.Server
}

// NewServer creates an MCP server with the given configuration.
func NewServer(cfg Config) *Server {
	return &Server{
		cfg:      cfg,
		engine:   search.NewHybrid(),
		entities: newEntityCache(cfg.Repo),
	}
}

// ServeHTTP handles MCP requests on the configured endpoint.
// It accepts JSON-RPC 2.0 messages via POST and returns JSON responses.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONRPCError(w, nil, errParse, "failed to read request body")
		return
	}
	r.Body.Close()

	// Try to parse as a single message or a batch.
	var msg jsonrpcMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		writeJSONRPCError(w, nil, errParse, "invalid JSON")
		return
	}

	// Notifications (no ID) don't get responses.
	if msg.ID == nil && msg.Method != "" {
		s.handleNotification(msg)
		w.WriteHeader(http.StatusAccepted)
		return
	}

	result, rpcErr := s.dispatch(r.Context(), &msg)
	if rpcErr != nil {
		writeJSONRPCError(w, msg.ID, rpcErr.Code, rpcErr.Message)
		return
	}

	writeJSONRPCResult(w, msg.ID, result)
}

// Start binds the HTTP server to the given address and begins serving.
func (s *Server) Start(addr string) error {
	mux := http.NewServeMux()
	mux.Handle("/mcp", s)
	mux.Handle("/mcp/", s)

	s.http = &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	fmt.Printf("memg mcp: listening on %s\n", addr)
	return s.http.ListenAndServe()
}

// Shutdown gracefully stops the HTTP server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.http == nil {
		return nil
	}
	return s.http.Shutdown(ctx)
}

// Handler returns the server as an http.Handler for embedding in existing muxes.
func (s *Server) Handler() http.Handler {
	return s
}

// entityCache mirrors the proxy's entity cache for resolving external IDs to UUIDs.
type entityCache struct {
	mu    sync.RWMutex
	repo  store.Repository
	cache map[string]string
}

func newEntityCache(repo store.Repository) *entityCache {
	return &entityCache{
		repo:  repo,
		cache: make(map[string]string),
	}
}

func (ec *entityCache) getOrCreateUUID(ctx context.Context, externalID string) (string, error) {
	ec.mu.RLock()
	if uuid, ok := ec.cache[externalID]; ok {
		ec.mu.RUnlock()
		return uuid, nil
	}
	ec.mu.RUnlock()

	uuid, err := ec.repo.UpsertEntity(ctx, externalID)
	if err != nil {
		return "", err
	}

	ec.mu.Lock()
	ec.cache[externalID] = uuid
	ec.mu.Unlock()

	return uuid, nil
}
