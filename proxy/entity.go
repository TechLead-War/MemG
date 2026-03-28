package proxy

import (
	"context"
	"net/http"
	"sort"
	"sync"
	"time"

	"memg/store"
)

const entityCacheTTL = 5 * time.Minute

type entityCacheEntry struct {
	uuid      string
	fetchedAt time.Time
}

// entityCache provides fast entity resolution with an in-memory cache backed
// by the persistent store. External IDs (from headers or config) are mapped
// to internal UUIDs. Entries expire after entityCacheTTL, and the cache is
// bounded to 10 000 entries to prevent unbounded memory growth.
type entityCache struct {
	mu    sync.RWMutex
	repo  store.Repository
	cache map[string]*entityCacheEntry
}

// newEntityCache creates an entity cache backed by the given repository.
func newEntityCache(repo store.Repository) *entityCache {
	return &entityCache{
		repo:  repo,
		cache: make(map[string]*entityCacheEntry),
	}
}

// resolveEntity determines the entity external ID for a request. It checks
// the X-MemG-Entity header first, falling back to defaultEntity. Returns an
// empty string if no entity can be determined.
func (ec *entityCache) resolveEntity(r *http.Request, defaultEntity string) string {
	if entity := r.Header.Get("X-MemG-Entity"); entity != "" {
		return entity
	}
	return defaultEntity
}

// getOrCreateUUID resolves an external entity ID to an internal UUID, creating
// the entity record if it does not exist. Results are cached in memory with
// TTL-based expiration.
func (ec *entityCache) getOrCreateUUID(ctx context.Context, externalID string) (string, error) {
	// Fast path: check cache under read lock.
	ec.mu.RLock()
	if entry, ok := ec.cache[externalID]; ok {
		if time.Since(entry.fetchedAt) < entityCacheTTL {
			ec.mu.RUnlock()
			return entry.uuid, nil
		}
	}
	ec.mu.RUnlock()

	// Slow path: upsert in the store and cache the result.
	uuid, err := ec.repo.UpsertEntity(ctx, externalID)
	if err != nil {
		return "", err
	}

	ec.mu.Lock()
	if len(ec.cache) > 10000 {
		now := time.Now()
		// First pass: evict stale entries.
		for k, v := range ec.cache {
			if now.Sub(v.fetchedAt) > entityCacheTTL {
				delete(ec.cache, k)
			}
		}
		// Hard cap: if still over 10,000 after stale eviction,
		// drop oldest entries until at the limit.
		if len(ec.cache) > 10000 {
			type aged struct {
				key string
				at  time.Time
			}
			entries := make([]aged, 0, len(ec.cache))
			for k, v := range ec.cache {
				entries = append(entries, aged{key: k, at: v.fetchedAt})
			}
			sort.Slice(entries, func(i, j int) bool {
				return entries[i].at.Before(entries[j].at)
			})
			excess := len(ec.cache) - 10000
			for i := 0; i < excess; i++ {
				delete(ec.cache, entries[i].key)
			}
		}
	}
	ec.cache[externalID] = &entityCacheEntry{
		uuid:      uuid,
		fetchedAt: time.Now(),
	}
	ec.mu.Unlock()

	return uuid, nil
}
