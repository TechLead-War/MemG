package embed

import (
	"context"
	"fmt"
	"math"
)

const probeText = "memg-healthcheck"

// Probe verifies that an embedder can produce one valid vector with the
// expected dimensionality. It is intended for startup/readiness checks.
func Probe(ctx context.Context, emb Embedder) error {
	if emb == nil {
		return fmt.Errorf("embedder is nil")
	}

	vectors, err := emb.Embed(ctx, []string{probeText})
	if err != nil {
		return fmt.Errorf("embedder probe failed: %w", err)
	}
	if len(vectors) != 1 {
		return fmt.Errorf("embedder probe returned %d vectors, want 1", len(vectors))
	}

	vec := vectors[0]
	if len(vec) == 0 {
		return fmt.Errorf("embedder probe returned an empty vector")
	}
	if dim := emb.Dimension(); dim > 0 && len(vec) != dim {
		return fmt.Errorf("embedder probe dimension mismatch: got %d, want %d", len(vec), dim)
	}
	for i, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("embedder probe returned invalid value at index %d", i)
		}
	}

	return nil
}
