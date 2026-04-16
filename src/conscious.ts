/**
 * Conscious context: loads the most important facts for always-on injection.
 * Matches the Go memory/conscious.go logic.
 */

import type { Store } from './store.js';
import type { ConsciousFact, FactFilter } from './types.js';

/**
 * Load the top facts by significance for an entity. These are the user's most
 * important attributes that should always be present regardless of query relevance.
 *
 * When the store implements listFactsMetadata, the lighter metadata-only query
 * is used (skips embedding decoding).
 *
 * Scoring:
 * - Identity facts with significance < 10: apply staleness decay after 30 days
 *   - staleness = 0 if confirmed < 30 days ago
 *   - staleness = (days - 30) / 90 if 30-120 days
 *   - staleness = 0.5 if > 120 days (capped)
 *   - score = significance * (1 - staleness)
 * - All other facts: score = significance
 *
 * Returns the top `limit` facts sorted by score descending.
 */
export async function loadConsciousContext(
  store: Store,
  entityUuid: string,
  limit?: number
): Promise<ConsciousFact[]> {
  if (!limit || limit <= 0) limit = 10;

  const filter: FactFilter = {
    statuses: ['current'],
    excludeExpired: true,
  };
  let fetchLimit = limit * 5;
  if (fetchLimit < 50) fetchLimit = 50;

  let facts;
  if (typeof store.listFactsMetadata === 'function') {
    facts = await store.listFactsMetadata(entityUuid, filter, fetchLimit);
  } else {
    facts = await store.listFactsFiltered(entityUuid, filter, fetchLimit);
  }
  if (!facts || facts.length === 0) return [];

  const now = Date.now();
  const scored = facts.map((f) => {
    let base = f.significance;

    if (f.factType === 'identity' && f.significance < 10) {
      let lastConfirmed = f.createdAt ? new Date(f.createdAt).getTime() : 0;
      if (f.reinforcedAt) {
        const ra = new Date(f.reinforcedAt).getTime();
        if (ra > lastConfirmed) lastConfirmed = ra;
      }
      if (f.lastRecalledAt) {
        const lr = new Date(f.lastRecalledAt).getTime();
        if (lr > lastConfirmed) lastConfirmed = lr;
      }
      const daysSinceConfirmed = (now - lastConfirmed) / (1000 * 60 * 60 * 24);
      if (daysSinceConfirmed > 30) {
        let staleness = (daysSinceConfirmed - 30) / 90;
        if (staleness > 0.5) staleness = 0.5;
        base *= 1 - staleness;
      }
    }

    return { fact: f, score: base };
  });

  scored.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    return a.fact.uuid.localeCompare(b.fact.uuid);
  });

  const result = scored.slice(0, limit);
  return result.map((s) => ({
    id: s.fact.uuid,
    content: s.fact.content,
    significance: s.fact.significance,
    tag: s.fact.tag,
    confidence: s.fact.confidence,
    verbatim: s.fact.verbatim,
    emotionalValence: s.fact.emotionalValence,
    factType: s.fact.factType,
  }));
}
