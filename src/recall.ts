/**
 * Recall: retrieve relevant facts and summaries using hybrid search.
 * Matches the Go memory.Recall and memory.RecallSummaries logic.
 */

import type { Embedder } from './embedder.js';
import {
  HybridEngine,
  dimensionMatch,
  type SearchCandidate,
  type RankResult,
} from './search.js';
import type { Store } from './store.js';
import type {
  Fact,
  FactFilter,
  RecalledFact,
  RecalledSummary,
} from './types.js';

const _backfillInProgress = new Set<string>();

/**
 * Recall facts for an entity using hybrid search.
 */
export async function recallFacts(
  store: Store,
  engine: HybridEngine,
  embedder: Embedder,
  queryText: string,
  entityUuid: string,
  limit: number,
  threshold: number,
  maxCandidates: number,
  filter?: FactFilter
): Promise<RecalledFact[]> {
  let vectors: number[][];
  try {
    vectors = await embedder.embed([queryText]);
  } catch (err) {
    console.warn('[memg] recall: embed query failed:', err);
    return [];
  }
  if (vectors.length === 0 || vectors[0].length === 0) return [];

  const queryVec = vectors[0];
  const queryModel = embedder.modelName();

  return recallFactsWithVector(
    store,
    engine,
    queryVec,
    queryModel,
    queryText,
    entityUuid,
    limit,
    threshold,
    maxCandidates,
    filter
  );
}

/**
 * Recall facts using a pre-computed query vector.
 */
export async function recallFactsWithVector(
  store: Store,
  engine: HybridEngine,
  queryVec: number[],
  queryModel: string,
  queryText: string,
  entityUuid: string,
  limit: number,
  threshold: number,
  maxCandidates: number,
  filter?: FactFilter,
  embedder?: Embedder
): Promise<RecalledFact[]> {
  if (queryVec.length === 0) return [];

  const effectiveFilter: FactFilter = {
    ...filter,
    excludeExpired: true,
  };

  // maxCandidates <= 0 means "load all facts" — no artificial ceiling.
  // Previously defaulted to 50 which silently dropped 50-80% of facts.

  const facts = await store.listFactsForRecall(entityUuid, effectiveFilter, maxCandidates);
  if (facts.length === 0) return [];

  // Detect facts with NULL embeddings for background backfill.
  const hasUnembedded = facts.some((f) => !f.embedding || f.embedding.length === 0);

  const candidates = buildRecallCandidates(queryVec, queryModel, facts);
  const results = engine.rank(queryVec, queryText, candidates, limit, threshold);

  // Trigger background backfill for facts stored during embedding outages.
  if (hasUnembedded && embedder && store.listUnembeddedFacts && store.updateFactEmbedding && !_backfillInProgress.has(entityUuid)) {
    _backfillInProgress.add(entityUuid);
    void (async () => {
      try {
        const unembedded = await store.listUnembeddedFacts(entityUuid, 50);
        if (unembedded.length === 0) return;
        const contents = unembedded.map((f) => f.content);
        const vectors = await embedder.embed(contents);
        const modelName = embedder.modelName();
        for (let i = 0; i < Math.min(vectors.length, unembedded.length); i++) {
          await store.updateFactEmbedding(unembedded[i].uuid, vectors[i], modelName);
        }
      } catch (e) {
        console.warn('memg: background backfill failed:', e);
      } finally {
        _backfillInProgress.delete(entityUuid);
      }
    })();
  }

  // Build a lookup from fact UUID to original Fact for fields not carried through RankResult.
  const factById = new Map<string, Fact>();
  for (const f of facts) {
    factById.set(f.uuid, f);
  }

  return results.map((r) => {
    const orig = factById.get(r.id);
    if (!orig) {
      console.warn(`[memg] recall: fact ${r.id} not found in lookup map, new fields will be empty`);
    }
    return {
      id: r.id,
      content: r.content,
      score: r.score,
      temporalStatus: r.temporalStatus,
      significance: r.significance,
      createdAt: r.createdAt,
      confidence: r.confidence,
      emotionalValence: orig?.emotionalValence,
      emotionalWeight: r.emotionalWeight,
      verbatim: orig?.verbatim,
      threadStatus: orig?.threadStatus ?? undefined,
      startedAt: orig?.startedAt,
      factType: orig?.factType,
      tag: r.tag,
      referenceTime: orig?.referenceTime,
    };
  });
}

/**
 * Recall conversation summaries using hybrid search.
 */
export async function recallSummaries(
  store: Store,
  engine: HybridEngine,
  embedder: Embedder,
  queryText: string,
  entityUuid: string,
  limit: number,
  threshold: number
): Promise<RecalledSummary[]> {
  let vectors: number[][];
  try {
    vectors = await embedder.embed([queryText]);
  } catch (err) {
    console.warn('[memg] recall: embed summaries query failed:', err);
    return [];
  }
  if (vectors.length === 0 || vectors[0].length === 0) return [];

  return recallSummariesWithVector(
    store,
    engine,
    vectors[0],
    queryText,
    entityUuid,
    limit,
    threshold,
    embedder.modelName()
  );
}

/**
 * Recall summaries using a pre-computed query vector.
 */
export async function recallSummariesWithVector(
  store: Store,
  engine: HybridEngine,
  queryVec: number[],
  queryText: string,
  entityUuid: string,
  limit: number,
  threshold: number,
  queryModel: string = ''
): Promise<RecalledSummary[]> {
  if (queryVec.length === 0) return [];

  const convs = await store.listConversationSummaries(entityUuid, 100);
  if (convs.length === 0) return [];

  const candidates: SearchCandidate[] = convs.map((c) => {
    let embedding = c.summaryEmbedding;
    if (queryVec.length > 0 && !dimensionMatch(queryVec, embedding)) {
      embedding = undefined;
    } else if (
      queryModel &&
      c.summaryEmbeddingModel &&
      c.summaryEmbeddingModel !== queryModel
    ) {
      embedding = undefined;
    }
    return {
      id: c.uuid,
      content: c.summary,
      embedding,
      createdAt: c.createdAt,
      temporalStatus: 'current',
      significance: 0,
      confidence: 1.0,
    };
  });

  const results = engine.rank(queryVec, queryText, candidates, limit, threshold);

  return results.map((r) => ({
    conversationId: r.id,
    summary: r.content,
    score: r.score,
    createdAt: r.createdAt,
  }));
}

function buildRecallCandidates(
  queryVec: number[],
  queryModel: string,
  facts: Fact[]
): SearchCandidate[] {
  return facts.map((f) => {
    let confidence = f.confidence;
    if (confidence === 0) confidence = 1.0;

    let embedding = f.embedding;
    if (queryVec.length > 0 && !dimensionMatch(queryVec, embedding)) {
      embedding = undefined;
    } else if (
      queryModel &&
      f.embeddingModel &&
      f.embeddingModel !== queryModel
    ) {
      embedding = undefined;
    }

    return {
      id: f.uuid,
      content: f.content,
      embedding,
      createdAt: f.createdAt,
      temporalStatus: f.temporalStatus,
      significance: f.significance,
      confidence,
      emotionalWeight: f.emotionalWeight,
      threadStatus: f.threadStatus ?? undefined,
      tag: f.tag,
      pinned: f.pinned,
      engagementScore: f.engagementScore,
      referenceTime: f.referenceTime,
    };
  });
}
