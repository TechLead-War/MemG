package memory

import (
	"context"
	"sync"
	"time"

	"memg/store"
)

// ConsciousCache caches conscious context results to avoid repeated
// database queries. It invalidates on significant inserts or status changes.
type ConsciousCache struct {
	mu      sync.RWMutex
	entries map[string]*consciousCacheEntry
	ttl     time.Duration
}

type consciousCacheEntry struct {
	facts     []*ConsciousFact
	fetchedAt time.Time
}

// NewConsciousCache creates a cache with the given TTL.
func NewConsciousCache(ttl time.Duration) *ConsciousCache {
	return &ConsciousCache{
		entries: make(map[string]*consciousCacheEntry),
		ttl:     ttl,
	}
}

// Get retrieves cached conscious facts for the entity, or nil if expired/missing.
func (cc *ConsciousCache) Get(entityUUID string) []*ConsciousFact {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	entry, ok := cc.entries[entityUUID]
	if !ok {
		return nil
	}
	if time.Since(entry.fetchedAt) > cc.ttl {
		return nil
	}
	// Return a copy to prevent mutation.
	result := make([]*ConsciousFact, len(entry.facts))
	copy(result, entry.facts)
	return result
}

// Set stores conscious facts for the entity.
func (cc *ConsciousCache) Set(entityUUID string, facts []*ConsciousFact) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	cc.entries[entityUUID] = &consciousCacheEntry{
		facts:     facts,
		fetchedAt: time.Now(),
	}
}

// Invalidate removes the cache entry for the entity. Call this when
// high-significance facts are inserted, promoted, or change status.
func (cc *ConsciousCache) Invalidate(entityUUID string) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	delete(cc.entries, entityUUID)
}

// InvalidateAll clears all cached entries.
func (cc *ConsciousCache) InvalidateAll() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.entries = make(map[string]*consciousCacheEntry)
}

// LoadConsciousContextCached is like LoadConsciousContext but checks the cache first.
func LoadConsciousContextCached(
	ctx context.Context,
	cache *ConsciousCache,
	repo store.FactFilteredReader,
	entityUUID string,
	limit int,
) ([]*ConsciousFact, error) {
	if cache != nil {
		if cached := cache.Get(entityUUID); cached != nil {
			return cached, nil
		}
	}

	facts, err := LoadConsciousContext(ctx, repo, entityUUID, limit)
	if err != nil {
		return nil, err
	}

	if cache != nil && facts != nil {
		cache.Set(entityUUID, facts)
	}
	return facts, nil
}
