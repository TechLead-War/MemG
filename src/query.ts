/**
 * Query transformation interfaces.
 * Matches the Go memory/query.go types.
 */

/** Result of transforming a raw chat query into a retrieval-optimized form. */
export interface QueryTransform {
  /** The retrieval-optimized version of the original query. Empty means use the original. */
  rewrittenQuery: string;
}

/**
 * Rewrites follow-up chat queries into standalone retrieval queries.
 * For example, "what about that?" in the context of a prior food discussion
 * might become "user food preferences and dietary restrictions".
 */
export interface QueryTransformer {
  transformQuery(query: string, recentHistory: string[]): Promise<QueryTransform>;
}
