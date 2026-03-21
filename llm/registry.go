package llm

import (
	"fmt"
	"sort"
	"sync"
)

// ProviderFactory creates a Provider from a ProviderConfig.
type ProviderFactory func(cfg ProviderConfig) (Provider, error)

var (
	registryMu sync.RWMutex
	registry   = make(map[string]ProviderFactory)
)

// RegisterProvider registers a named provider factory.
func RegisterProvider(name string, f ProviderFactory) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[name] = f
}

// NewProvider creates a provider by name using the given config.
func NewProvider(name string, cfg ProviderConfig) (Provider, error) {
	registryMu.RLock()
	f, ok := registry[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("llm provider %q is not registered", name)
	}
	return f(cfg)
}

// ListProviders returns registered provider names in sorted order.
func ListProviders() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
