// Package augment provides an asynchronous pipeline that extracts structured
// knowledge from conversation messages and persists it to the backing store.
package augment

import (
	"context"

	"memg/graph"
	"memg/llm"
	"memg/store"
)

// Job describes a unit of work for the augmentation pipeline.
type Job struct {
	EntityID  string
	ProcessID string
	Messages  []*llm.Message
}

// Extraction holds the structured knowledge produced by a Stage.
//
// Stages should set Type, TemporalStatus, Significance, and ContentKey on
// each Fact when they have domain knowledge to do so. The pipeline applies
// defaults for any zero-value fields:
//   - Type: FactTypeIdentity
//   - TemporalStatus: TemporalCurrent
//   - Significance: SignificanceMedium
//   - ContentKey: auto-computed from Content
type Extraction struct {
	Facts   []*store.Fact
	Triples []*graph.Triple
}

// Stage is a single processing step in the augmentation pipeline.
type Stage interface {
	// Name returns a human-readable label for this stage.
	Name() string
	// Execute analyses the job and returns any extracted knowledge.
	Execute(ctx context.Context, job *Job) (*Extraction, error)
}

// ConflictDetector is an optional interface that stages can implement to
// identify existing facts that a new extraction supersedes. When the pipeline
// detects that a stage implements this interface, it calls DetectConflicts
// before inserting identity facts and reclassifies any returned fact UUIDs
// from current to historical.
type ConflictDetector interface {
	// DetectConflicts returns UUIDs of existing facts that newFact supersedes.
	DetectConflicts(ctx context.Context, entityUUID string, newFact *store.Fact, repo store.Repository) ([]string, error)
}
