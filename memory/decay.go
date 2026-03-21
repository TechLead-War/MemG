package memory

import (
	"context"
	"log"
	"time"

	"memg/store"
)

// Pruner periodically removes expired facts from the store.
type Pruner struct {
	repo     store.Repository
	interval time.Duration
	stopCh   chan struct{}
	done     chan struct{}
}

// NewPruner creates a pruner that checks for expired facts every interval.
// A zero interval disables automatic pruning.
func NewPruner(repo store.Repository, interval time.Duration) *Pruner {
	return &Pruner{
		repo:     repo,
		interval: interval,
		stopCh:   make(chan struct{}),
		done:     make(chan struct{}),
	}
}

// Start begins the background pruning loop. It is safe to call Start
// only once; subsequent calls have no effect.
func (p *Pruner) Start() {
	if p.interval <= 0 {
		close(p.done)
		return
	}
	go p.loop()
}

// Stop signals the pruner to stop and blocks until the loop exits.
func (p *Pruner) Stop() {
	select {
	case <-p.stopCh:
		// Already stopped.
	default:
		close(p.stopCh)
	}
	<-p.done
}

func (p *Pruner) loop() {
	defer close(p.done)
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()

	for {
		select {
		case <-p.stopCh:
			return
		case <-ticker.C:
			p.prune()
		}
	}
}

func (p *Pruner) prune() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Prune in batches to avoid long-running transactions. The repository
	// implementation uses a LIMIT clause (via subquery) so each call removes
	// at most 1000 facts.
	const batchSize = 1000
	var totalPruned int64

	for {
		n, err := p.repo.PruneExpiredFacts(ctx, "", time.Now().UTC())
		if err != nil {
			log.Printf("memg pruner: %v", err)
			return
		}
		totalPruned += n
		if n < batchSize {
			break // All expired facts pruned.
		}
	}

	if totalPruned > 0 {
		log.Printf("memg pruner: removed %d expired facts", totalPruned)
	}

	summaryCutoff := time.Now().UTC().Add(-90 * 24 * time.Hour)
	sn, err := p.repo.PruneStaleSummaries(ctx, summaryCutoff)
	if err != nil {
		log.Printf("memg pruner: summary prune: %v", err)
	} else if sn > 0 {
		log.Printf("memg pruner: cleared %d stale summaries", sn)
	}
}
