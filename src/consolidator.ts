/**
 * Consolidator: clusters old, low-significance event facts into pattern facts.
 * Matches the Go memory/consolidator.go logic.
 */

import { randomUUID } from 'crypto';
import type { Embedder } from './embedder.js';
import { cosineSimilarity } from './search.js';
import { defaultContentKey, type Store } from './store.js';
import type { Fact, FactFilter } from './types.js';

/** Configurable thresholds for entity consolidation. */
export interface ConsolidateOptions {
  /** Minimum age in days for facts to be eligible. Default: 30. */
  minAgeDays?: number;
  /** Maximum significance for eligible facts. Default: 5. */
  maxSignificance?: number;
  /** Minimum facts per tag group to trigger consolidation. Default: 3. */
  minGroupSize?: number;
  /** Maximum facts to scan. Default: 500. */
  scanLimit?: number;
}

export interface ConsolidatorConfig {
  store: Store;
  embedder: Embedder;
  llmChat: (prompt: string) => Promise<string>;
  intervalMs?: number;
}

/**
 * Consolidate old event facts for a single entity into pattern facts.
 *
 * Logic (matches Go consolidateEntity):
 * 1. Query event facts older than `minAgeDays`, current status, significance <= `maxSignificance`
 * 2. Group by tag
 * 3. For groups with `minGroupSize`+ facts, ask LLM to produce a behavioral pattern
 * 4. If a semantically similar pattern already exists (>0.88), reinforce it instead of duplicating
 * 5. Otherwise embed the pattern, insert as pattern fact
 * 6. Mark originals as historical
 * 7. Return count of patterns created
 */
export async function consolidateEntity(
  store: Store,
  embedder: Embedder,
  llmChat: (prompt: string) => Promise<string>,
  entityUuid: string,
  opts?: ConsolidateOptions
): Promise<number> {
  const minAgeDays = opts?.minAgeDays ?? 30;
  const maxSignificance = opts?.maxSignificance ?? 5;
  const minGroupSize = opts?.minGroupSize ?? 3;
  const scanLimit = opts?.scanLimit ?? 500;

  const cutoff = new Date(Date.now() - minAgeDays * 24 * 60 * 60 * 1000).toISOString();

  const filter: FactFilter = {
    types: ['event'],
    statuses: ['current'],
    excludeExpired: true,
    referenceTimeBefore: cutoff,
    maxSignificance: maxSignificance,
  };

  const facts = await store.listFactsFiltered(entityUuid, filter, scanLimit);
  if (facts.length < minGroupSize) return 0;

  // Pre-load existing pattern facts for duplicate detection.
  const existingPatterns = await store.listFactsFiltered(
    entityUuid,
    { types: ['pattern'], statuses: ['current'] },
    scanLimit
  );

  const byTag = new Map<string, Fact[]>();
  for (const f of facts) {
    const tag = f.tag || '_untagged';
    const group = byTag.get(tag) ?? [];
    group.push(f);
    byTag.set(tag, group);
  }

  let patternsCreated = 0;

  for (const [tag, group] of byTag) {
    if (group.length < minGroupSize) continue;

    group.sort((a, b) => a.content.localeCompare(b.content));

    const contents = group.map((f) => `- ${f.content}`).join('\n');

    const prompt = `Summarize these ${group.length} related events into a single behavioral pattern statement.
The pattern should describe a recurring behavior or tendency.
Return ONLY the pattern statement, nothing else.
If these events don't form a meaningful pattern, respond with exactly: NONE

Events:
${contents}`;

    let pattern: string;
    try {
      pattern = await llmChat(prompt);
    } catch (err) {
      console.warn('[memg] consolidator: LLM call failed:', err);
      continue;
    }

    pattern = pattern.trim();
    if (!pattern || pattern.toUpperCase() === 'NONE') continue;

    let vectors: number[][];
    try {
      vectors = await embedder.embed([pattern]);
    } catch (err) {
      console.warn('[memg] consolidator: embed failed:', err);
      continue;
    }
    if (vectors.length === 0) continue;

    const newEmbedding = vectors[0];

    // Check for duplicate pattern: if an existing pattern is semantically
    // very similar (>0.88), reinforce it instead of creating a duplicate.
    let duplicateFound = false;
    for (const existing of existingPatterns) {
      if (!existing.embedding || existing.embedding.length === 0) continue;
      if (existing.embedding.length !== newEmbedding.length) continue;
      const sim = cosineSimilarity(newEmbedding, existing.embedding);
      if (sim > 0.88) {
        try {
          await store.reinforceFact(existing.uuid, null);
        } catch (err) {
          console.warn('[memg] consolidator: reinforce existing pattern failed:', err);
        }
        duplicateFound = true;
        break;
      }
    }

    if (duplicateFound) {
      // Still mark originals as historical even when reinforcing an existing pattern.
      for (const f of group) {
        try {
          await store.updateTemporalStatus(f.uuid, 'historical', 'consolidated');
        } catch (err) {
          console.warn('[memg] consolidator: updateTemporalStatus failed:', err);
        }
      }
      continue;
    }

    const patternFact: Fact = {
      uuid: randomUUID(),
      content: pattern,
      embedding: newEmbedding,
      factType: 'pattern',
      temporalStatus: 'current',
      significance: maxSignificance,
      contentKey: defaultContentKey(pattern),
      tag,
      slot: '',
      confidence: 1.0,
      embeddingModel: embedder.modelName(),
      sourceRole: '',
      reinforcedCount: 0,
      recallCount: 0,
    };

    try {
      await store.insertFact(entityUuid, patternFact);
    } catch (err) {
      console.warn('[memg] consolidator: insertFact failed:', err);
      continue;
    }

    // Add the newly inserted pattern to the existing patterns list so
    // subsequent tag groups can detect duplicates against it.
    existingPatterns.push(patternFact);

    for (const f of group) {
      try {
        await store.updateTemporalStatus(f.uuid, 'historical', patternFact.uuid);
      } catch (err) {
        console.warn('[memg] consolidator: updateTemporalStatus failed:', err);
      }
    }

    patternsCreated++;
  }

  return patternsCreated;
}
