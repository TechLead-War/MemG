/**
 * Consolidator: clusters old, low-significance event facts into pattern facts.
 * Matches the Go memory/consolidator.go logic.
 */

import { randomUUID } from 'crypto';
import type { Embedder } from './embedder.js';
import { defaultContentKey, type Store } from './store.js';
import type { Fact, FactFilter } from './types.js';

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
 * 1. Query event facts older than 30 days, current status, significance <= 5
 * 2. Group by tag
 * 3. For groups with 3+ facts, ask LLM to produce a behavioral pattern
 * 4. Embed the pattern, insert as pattern fact
 * 5. Mark originals as historical
 * 6. Return count of patterns created
 */
export async function consolidateEntity(
  store: Store,
  embedder: Embedder,
  llmChat: (prompt: string) => Promise<string>,
  entityUuid: string
): Promise<number> {
  const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();

  const filter: FactFilter = {
    types: ['event'],
    statuses: ['current'],
    excludeExpired: true,
    referenceTimeBefore: cutoff,
    maxSignificance: 5,
  };

  const facts = await store.listFactsFiltered(entityUuid, filter, 500);
  if (facts.length < 3) return 0;

  const byTag = new Map<string, Fact[]>();
  for (const f of facts) {
    const tag = f.tag || '_untagged';
    const group = byTag.get(tag) ?? [];
    group.push(f);
    byTag.set(tag, group);
  }

  let patternsCreated = 0;

  for (const [tag, group] of byTag) {
    if (group.length < 3) continue;

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
    } catch {
      continue;
    }

    pattern = pattern.trim();
    if (!pattern || pattern.toUpperCase() === 'NONE') continue;

    let vectors: number[][];
    try {
      vectors = await embedder.embed([pattern]);
    } catch {
      continue;
    }
    if (vectors.length === 0) continue;

    const patternFact: Fact = {
      uuid: randomUUID(),
      content: pattern,
      embedding: vectors[0],
      factType: 'pattern',
      temporalStatus: 'current',
      significance: 5,
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
    } catch {
      continue;
    }

    for (const f of group) {
      try {
        await store.updateTemporalStatus(f.uuid, 'historical');
      } catch {
        // best-effort
      }
    }

    patternsCreated++;
  }

  return patternsCreated;
}
