package embed

import (
	"fmt"
	"sort"
	"sync"
)

// EmbedderFactory creates an Embedder from a ProviderConfig.
type EmbedderFactory func(cfg ProviderConfig) (Embedder, error)

var (
	mu        sync.RWMutex
	factories = make(map[string]EmbedderFactory)
)

// RegisterEmbedder registers a named factory for creating embedders.
// It is intended to be called from provider init() functions.
func RegisterEmbedder(name string, f EmbedderFactory) {
	mu.Lock()
	defer mu.Unlock()
	factories[name] = f
}

// NewEmbedder creates an Embedder by looking up the named factory.
// It returns an error if no factory is registered under that name.
func NewEmbedder(name string, cfg ProviderConfig) (Embedder, error) {
	mu.RLock()
	f, ok := factories[name]
	mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("embed: unknown provider %q", name)
	}
	return f(cfg)
}

// ListEmbedders returns the names of all registered embedding providers, sorted.
func ListEmbedders() []string {
	mu.RLock()
	defer mu.RUnlock()
	names := make([]string, 0, len(factories))
	for name := range factories {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
