/**
 * Decay: prune expired facts and stale summaries.
 * Matches the Go memory/decay.go Pruner logic.
 */

import type { Store } from './store.js';

/**
 * Prune expired facts and stale conversation summaries.
 *
 * 1. Calls store.pruneExpiredFacts with current time (batched internally by store).
 * 2. Calls store.pruneStaleSummaries with the given max age (default 90 days).
 * 3. Returns counts of pruned facts and cleared summaries.
 */
export async function pruneExpiredAndStale(
  store: Store,
  summaryMaxAgeDays?: number
): Promise<{ factsPruned: number; summariesCleared: number }> {
  const now = new Date().toISOString();
  const maxAge = summaryMaxAgeDays ?? 90;

  // Prune expired facts. The store implementation batches internally.
  // Pass empty entityUuid to prune across all entities (matching Go behavior
  // which passes "" for the global sweep).
  const factsPruned = await store.pruneExpiredFacts('', now);
  const summariesCleared = await store.pruneStaleSummaries(maxAge);

  return { factsPruned, summariesCleared };
}
