// Package memg provides a pluggable memory layer for language model applications.
// It intercepts LLM calls to inject contextual recall from stored knowledge,
// tracks conversations, and asynchronously extracts entities and facts for
// future retrieval.
package memg

import "errors"

// Sentinel errors returned by the MemG API.
var (
	ErrNoRepository   = errors.New("memg: repository is required")
	ErrNoProvider     = errors.New("memg: language model provider not configured")
	ErrNoEmbedder     = errors.New("memg: embedder not configured")
	ErrEntityRequired = errors.New("memg: entity identifier is required for this operation")
	ErrAlreadyClosed  = errors.New("memg: instance already closed")
)
