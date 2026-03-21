package augment

import (
	"context"
	"log"
	"sync"
	"time"

	"memg/search"
	"memg/store"
)

const (
	// semanticDedupThreshold is the minimum cosine similarity for two facts
	// to be considered duplicates. 0.92 catches paraphrases while avoiding
	// false positives on merely related facts.
	semanticDedupThreshold = 0.92

	// semanticDedupCandidates is the maximum number of existing facts loaded
	// for embedding comparison during semantic dedup.
	semanticDedupCandidates = 500

	// promotionThreshold is the reinforcement count at which a fact is
	// automatically promoted to SignificanceHigh (never expires, surfaces
	// in conscious mode).
	promotionThreshold = 5
)

// Pipeline accepts jobs, processes them through registered stages, and
// writes the resulting extractions to the backing store.
type Pipeline struct {
	mu     sync.RWMutex
	repo   store.Repository
	stages []Stage
	pool   *Pool

	// OnFactInserted is called after a high-significance or identity fact is
	// inserted or promoted. Can be used to invalidate caches.
	OnFactInserted func(entityUUID string, fact *store.Fact)

	// OnFactStatusChanged is called when a fact's temporal status is changed
	// (e.g., reclassified from current to historical). Can be used to invalidate caches.
	OnFactStatusChanged func(entityUUID string, factUUID string)
}

// NewPipeline creates an augmentation pipeline backed by the given repository.
func NewPipeline(repo store.Repository) *Pipeline {
	return &Pipeline{
		repo: repo,
		pool: NewPool(32),
	}
}

// AddStage appends a processing stage to the pipeline.
func (p *Pipeline) AddStage(s Stage) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.stages = append(p.stages, s)
}

// Enqueue submits a job for asynchronous processing. If the worker queue
// is full, the job is dropped to prevent goroutine pile-up.
func (p *Pipeline) Enqueue(job *Job) {
	p.mu.RLock()
	stages := make([]Stage, len(p.stages))
	copy(stages, p.stages)
	p.mu.RUnlock()

	if len(stages) == 0 || job == nil {
		return
	}

	if !p.pool.Submit(func() {
		p.process(context.Background(), stages, job)
	}) {
		log.Printf("memg augment: pipeline queue full, dropping extraction job")
	}
}

func (p *Pipeline) process(ctx context.Context, stages []Stage, job *Job) {
	for _, s := range stages {
		ext, err := s.Execute(ctx, job)
		if err != nil {
			log.Printf("memg augment: stage %q failed: %v", s.Name(), err)
			continue
		}
		if ext == nil {
			continue
		}
		if len(ext.Facts) > 0 && job.EntityID != "" {
			p.persistFacts(ctx, job.EntityID, ext.Facts, s)
		}
	}
}

// persistFacts handles dedup, evolution, and default-filling for each fact.
func (p *Pipeline) persistFacts(ctx context.Context, entityUUID string, facts []*store.Fact, stage Stage) {
	// Load existing facts once for dedup and conflict detection.
	existingFacts := p.loadEntityFacts(ctx, entityUUID)

	for _, f := range facts {
		applyDefaults(f)

		if f.ExpiresAt == nil && f.Significance < store.SignificanceHigh {
			f.ExpiresAt = store.TTLForSignificance(f.Significance)
		}

		// Step 1: Exact dedup by content key hash.
		existing, err := p.repo.FindFactByKey(ctx, entityUUID, f.ContentKey)
		if err != nil {
			log.Printf("memg augment: find by key: %v", err)
			continue
		}

		if existing != nil {
			p.reinforceAndPromote(ctx, existing, f.ExpiresAt)
			if p.OnFactInserted != nil {
				safeCallback(func() { p.OnFactInserted(entityUUID, existing) })
			}
			continue
		}

		// Step 2: Slot conflict resolution for mutable identity facts.
		// This runs BEFORE semantic dedup so "Seattle" doesn't get merged
		// into "Austin" as "similar enough".
		if f.Slot != "" && f.Type == store.FactTypeIdentity && f.TemporalStatus == store.TemporalCurrent {
			slotConflicts := findSlotConflicts(f.Slot, existingFacts)
			for _, conflict := range slotConflicts {
				if err := p.repo.UpdateTemporalStatus(ctx, conflict.UUID, store.TemporalHistorical); err != nil {
					log.Printf("memg augment: reclassify slot conflict %s: %v", conflict.UUID, err)
				} else {
					// Update in-memory state to prevent stale reads by subsequent facts.
					conflict.TemporalStatus = store.TemporalHistorical
					if p.OnFactStatusChanged != nil {
						safeCallback(func() { p.OnFactStatusChanged(entityUUID, conflict.UUID) })
					}
				}
			}
		}

		// Step 3: Semantic dedup by embedding similarity.
		if len(f.Embedding) > 0 && len(existingFacts) > 0 {
			if similar := findSemanticMatch(f.Embedding, existingFacts); similar != nil {
				p.reinforceAndPromote(ctx, similar, f.ExpiresAt)
				if p.OnFactInserted != nil {
					safeCallback(func() { p.OnFactInserted(entityUUID, similar) })
				}
				continue
			}
		}

		// Step 4: Evolution — stage-provided ConflictDetector.
		if f.Type == store.FactTypeIdentity && f.TemporalStatus == store.TemporalCurrent {
			if cd, ok := stage.(ConflictDetector); ok {
				conflictIDs, err := cd.DetectConflicts(ctx, entityUUID, f, p.repo)
				if err != nil {
					log.Printf("memg augment: detect conflicts: %v", err)
				}
				for _, id := range conflictIDs {
					if err := p.repo.UpdateTemporalStatus(ctx, id, store.TemporalHistorical); err != nil {
						log.Printf("memg augment: reclassify fact %s: %v", id, err)
					} else if p.OnFactStatusChanged != nil {
						conflictID := id // capture loop variable
						safeCallback(func() { p.OnFactStatusChanged(entityUUID, conflictID) })
					}
				}
			}
		}

		// Step 5: Insert the new fact.
		if err := p.repo.InsertFact(ctx, entityUUID, f); err != nil {
			log.Printf("memg augment: insert fact: %v", err)
		} else if p.OnFactInserted != nil && (f.Significance >= store.SignificanceHigh || f.Type == store.FactTypeIdentity) {
			safeCallback(func() { p.OnFactInserted(entityUUID, f) })
		}
	}
}

// findSlotConflicts returns existing current facts that occupy the same slot.
func findSlotConflicts(slot string, existing []*store.Fact) []*store.Fact {
	if slot == "" || existing == nil {
		return nil
	}
	var conflicts []*store.Fact
	for _, f := range existing {
		if f.Slot == slot && f.TemporalStatus == store.TemporalCurrent && f.Type == store.FactTypeIdentity {
			conflicts = append(conflicts, f)
		}
	}
	return conflicts
}

// reinforceAndPromote reinforces an existing fact and promotes it to
// SignificanceHigh if the reinforcement count crosses the threshold.
func (p *Pipeline) reinforceAndPromote(ctx context.Context, existing *store.Fact, newExpiresAt *time.Time) {
	newCount := existing.ReinforcedCount + 1
	shouldPromote := newCount >= promotionThreshold && existing.Significance < store.SignificanceHigh

	// If promoting, clear the TTL — high-significance facts never expire.
	expiresAt := newExpiresAt
	if shouldPromote {
		expiresAt = nil
	}

	if err := p.repo.ReinforceFact(ctx, existing.UUID, expiresAt); err != nil {
		log.Printf("memg augment: reinforce fact: %v", err)
		return
	}

	if shouldPromote {
		if err := p.repo.UpdateSignificance(ctx, existing.UUID, store.SignificanceHigh); err != nil {
			log.Printf("memg augment: promote fact %s: %v", existing.UUID, err)
		} else {
			log.Printf("memg augment: promoted fact %s to high significance (reinforced %d times)", existing.UUID, newCount)
		}
	}
}

// loadEntityFacts loads current, non-expired facts for semantic dedup.
func (p *Pipeline) loadEntityFacts(ctx context.Context, entityUUID string) []*store.Fact {
	filter := store.FactFilter{
		Statuses:       []store.TemporalStatus{store.TemporalCurrent},
		ExcludeExpired: true,
	}
	var (
		facts []*store.Fact
		err   error
	)
	if rr, ok := p.repo.(store.FactRecallReader); ok {
		facts, err = rr.ListFactsForRecall(ctx, entityUUID, filter, 0)
	} else {
		facts, err = p.repo.ListFactsFiltered(ctx, entityUUID, filter, semanticDedupCandidates)
	}
	if err != nil {
		log.Printf("memg augment: load facts for semantic dedup: %v", err)
		return nil
	}
	return facts
}

// findSemanticMatch returns the existing fact most similar to the given
// embedding, or nil if no fact exceeds the similarity threshold.
func findSemanticMatch(embedding []float32, candidates []*store.Fact) *store.Fact {
	var best *store.Fact
	var bestScore float64

	for _, f := range candidates {
		if len(f.Embedding) == 0 {
			continue
		}
		score := search.CosineSimilarity(embedding, f.Embedding)
		if score > bestScore {
			bestScore = score
			best = f
		}
	}

	if bestScore >= semanticDedupThreshold {
		return best
	}
	return nil
}

// applyDefaults fills zero-value metadata fields with sensible defaults.
func applyDefaults(f *store.Fact) {
	if f.Type == "" {
		f.Type = store.FactTypeIdentity
	}
	if f.TemporalStatus == "" {
		f.TemporalStatus = store.TemporalCurrent
	}
	if f.Significance == 0 {
		f.Significance = store.SignificanceMedium
	}
	if f.ContentKey == "" {
		f.ContentKey = store.DefaultContentKey(f.Content)
	}
}

// safeCallback invokes fn inside a recover guard so that a panicking callback
// does not crash the pool worker or drop remaining facts in the batch.
func safeCallback(fn func()) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("memg augment: callback panic: %v", r)
		}
	}()
	fn()
}

// Shutdown drains the worker pool and blocks until all in-flight jobs finish.
func (p *Pipeline) Shutdown() {
	p.pool.Shutdown()
}
