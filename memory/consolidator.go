package memory

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/store"
)

// Consolidator periodically clusters old, low-significance event facts into
// pattern facts, marks the originals as historical, and preserves a summary
// of repeated behavior.
type Consolidator struct {
	repo      store.Repository
	provider  llm.Provider
	embedder  embed.Embedder
	interval  time.Duration
	stopCh    chan struct{}
	done      chan struct{}
	startOnce sync.Once
}

// NewConsolidator creates a consolidator that runs every interval.
// A zero interval disables automatic consolidation.
func NewConsolidator(repo store.Repository, provider llm.Provider, embedder embed.Embedder, interval time.Duration) *Consolidator {
	return &Consolidator{
		repo:     repo,
		provider: provider,
		embedder: embedder,
		interval: interval,
		stopCh:   make(chan struct{}),
		done:     make(chan struct{}),
	}
}

// Start begins the background consolidation loop. It is safe to call Start
// multiple times; subsequent calls are no-ops.
func (c *Consolidator) Start() {
	c.startOnce.Do(func() {
		if c.interval <= 0 || c.provider == nil || c.embedder == nil {
			close(c.done)
			return
		}
		go c.loop()
	})
}

// Stop signals the consolidator to stop and waits for the loop to exit.
func (c *Consolidator) Stop() {
	select {
	case <-c.stopCh:
	default:
		close(c.stopCh)
	}
	<-c.done
}

func (c *Consolidator) loop() {
	defer close(c.done)
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.consolidate()
		}
	}
}

// ConsolidateEntity triggers consolidation for a single entity. This allows
// callers to run consolidation on demand rather than waiting for the periodic
// background sweep.
func (c *Consolidator) ConsolidateEntity(ctx context.Context, entityUUID string) error {
	if c.provider == nil || c.embedder == nil {
		return fmt.Errorf("consolidator: provider and embedder are required")
	}
	return c.consolidateEntity(ctx, entityUUID)
}

// consolidate iterates over all known entities and consolidates old event
// facts for each one individually.
func (c *Consolidator) consolidate() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	entityUUIDs, err := c.repo.ListEntityUUIDs(ctx, 1000)
	if err != nil {
		log.Printf("memg consolidator: list entities: %v", err)
		return
	}

	for _, entityUUID := range entityUUIDs {
		select {
		case <-c.stopCh:
			return
		case <-ctx.Done():
			return
		default:
		}
		if err := c.consolidateEntity(ctx, entityUUID); err != nil {
			log.Printf("memg consolidator: entity %s: %v", entityUUID, err)
		}
	}
}

// consolidateEntity finds clusters of old event facts for a single entity and
// merges them into pattern facts.
func (c *Consolidator) consolidateEntity(ctx context.Context, entityUUID string) error {
	// Find old event facts (older than 30 days) for this entity.
	cutoff := time.Now().Add(-30 * 24 * time.Hour)

	filter := store.FactFilter{
		Types:               []store.FactType{store.FactTypeEvent},
		Statuses:            []store.TemporalStatus{store.TemporalCurrent},
		ExcludeExpired:      true,
		ReferenceTimeBefore: &cutoff,
		MaxSignificance:     store.SignificanceMedium,
	}

	facts, err := c.repo.ListFactsFiltered(ctx, entityUUID, filter, 500)
	if err != nil {
		return fmt.Errorf("load candidates: %w", err)
	}

	if len(facts) < 3 {
		return nil // Not enough facts to consolidate.
	}

	byTag := make(map[string][]*store.Fact)
	for _, f := range facts {
		tag := f.Tag
		if tag == "" {
			tag = "_untagged"
		}
		byTag[tag] = append(byTag[tag], f)
	}

	for tag, group := range byTag {
		if len(group) < 3 {
			continue // Need at least 3 similar facts to form a pattern.
		}

		// Sort by content for deterministic processing.
		sort.Slice(group, func(i, j int) bool {
			return group[i].Content < group[j].Content
		})

		// Ask the LLM to summarize the cluster into a pattern fact.
		var contents strings.Builder
		for _, f := range group {
			contents.WriteString("- ")
			contents.WriteString(f.Content)
			contents.WriteByte('\n')
		}

		prompt := fmt.Sprintf(`Summarize these %d related events into a single behavioral pattern statement.
The pattern should describe a recurring behavior or tendency.
Return ONLY the pattern statement, nothing else.
If these events don't form a meaningful pattern, respond with exactly: NONE

Events:
%s`, len(group), contents.String())

		req := &llm.Request{
			Messages:  []*llm.Message{llm.UserMessage(prompt)},
			MaxTokens: 100,
		}

		resp, err := c.provider.Chat(ctx, req)
		if err != nil {
			log.Printf("memg consolidator: llm call for entity %s tag %q: %v", entityUUID, tag, err)
			continue
		}

		pattern := strings.TrimSpace(resp.Content)
		if pattern == "" || strings.EqualFold(pattern, "NONE") {
			continue
		}

		vectors, err := c.embedder.Embed(ctx, []string{pattern})
		if err != nil || len(vectors) == 0 {
			log.Printf("memg consolidator: embed pattern for entity %s tag %q: %v", entityUUID, tag, err)
			continue
		}

		patternFact := &store.Fact{
			Content:        pattern,
			Embedding:      vectors[0],
			Type:           store.FactTypePattern,
			TemporalStatus: store.TemporalCurrent,
			Significance:   store.SignificanceMedium,
			Tag:            tag,
			ContentKey:     store.DefaultContentKey(pattern),
		}

		if err := c.repo.InsertFact(ctx, entityUUID, patternFact); err != nil {
			log.Printf("memg consolidator: insert pattern for entity %s tag %q: %v", entityUUID, tag, err)
			continue
		}

		// Mark originals as historical only after the pattern is persisted.
		for _, f := range group {
			if err := c.repo.UpdateTemporalStatus(ctx, f.UUID, store.TemporalHistorical); err != nil {
				log.Printf("memg consolidator: mark historical %s: %v", f.UUID, err)
			}
		}

		log.Printf("memg consolidator: created pattern from %d events (entity=%s tag=%s): %s", len(group), entityUUID, tag, pattern)
	}

	return nil
}
