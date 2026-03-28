/**
 * Re-embedding: backfill missing embeddings and re-embed with a new model.
 * Matches the Go memory/reembed.go logic.
 */

import type { Embedder } from './embedder.js';
import type { Store } from './store.js';
import type { FactFilter } from './types.js';

/**
 * Backfill embeddings for facts that were stored without them (e.g., embedder
 * was down at insertion time).
 *
 * Returns the number of facts updated.
 */
export async function backfillMissingEmbeddings(
  store: Store,
  embedder: Embedder,
  entityUuid: string,
  limit?: number
): Promise<number> {
  const effectiveLimit = limit && limit > 0 ? limit : 50;

  const filter: FactFilter = {
    unembeddedOnly: true,
    excludeExpired: true,
  };
  const facts = await store.listFactsFiltered(entityUuid, filter, effectiveLimit);
  if (!facts || facts.length === 0) return 0;

  const contents = facts.map((f) => f.content);
  let vectors: number[][];
  try {
    vectors = await embedder.embed(contents);
  } catch {
    return 0;
  }

  const modelName = embedder.modelName();
  let updated = 0;
  for (let i = 0; i < facts.length && i < vectors.length; i++) {
    try {
      await store.updateFactEmbedding(facts[i].uuid, vectors[i], modelName);
      updated++;
    } catch {
      // best-effort
    }
  }
  return updated;
}

/**
 * Re-embed all facts for an entity using the provided embedder. This is
 * needed when the embedding model changes -- old facts have incompatible
 * vectors that produce zero similarity during recall.
 *
 * Processes facts in batches and updates each fact's embedding and
 * embedding_model in place. Returns the number of facts updated.
 */
export async function reEmbedFacts(
  store: Store,
  embedder: Embedder,
  entityUuid: string,
  modelName: string,
  batchSize?: number
): Promise<number> {
  const effectiveBatch = batchSize && batchSize > 0 ? batchSize : 50;

  const filter: FactFilter = { excludeExpired: true };
  const facts = await store.listFactsFiltered(entityUuid, filter, 100000);
  if (!facts || facts.length === 0) return 0;

  let updated = 0;
  for (let i = 0; i < facts.length; i += effectiveBatch) {
    const end = Math.min(i + effectiveBatch, facts.length);
    const batch = facts.slice(i, end);

    const contents = batch.map((f) => f.content);
    let vectors: number[][];
    try {
      vectors = await embedder.embed(contents);
    } catch {
      break;
    }

    for (let j = 0; j < batch.length && j < vectors.length; j++) {
      try {
        await store.updateFactEmbedding(batch[j].uuid, vectors[j], modelName);
        updated++;
      } catch {
        // best-effort
      }
    }
  }

  return updated;
}
