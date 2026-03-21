package memory

import "context"

// QueryTransform is the result of transforming a raw chat query into a
// retrieval-optimized form.
type QueryTransform struct {
	// RewrittenQuery is the retrieval-optimized version of the original query.
	// Empty means use the original query as-is.
	RewrittenQuery string
}

// QueryTransformer rewrites follow-up chat queries into standalone retrieval
// queries. For example, "what about that?" in the context of a prior food
// discussion might become "user food preferences and dietary restrictions".
type QueryTransformer interface {
	TransformQuery(ctx context.Context, query string, recentHistory []string) (*QueryTransform, error)
}
