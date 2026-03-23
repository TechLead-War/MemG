/**
 * Recall: retrieve relevant facts and summaries using hybrid search.
 * Matches the Go memory.Recall and memory.RecallSummaries logic.
 */

import type { Embedder } from './embedder';
import {
  HybridEngine,
  dimensionMatch,
  type SearchCandidate,
  type RankResult,
} from './search';
import type { Store } from './store';
import type {
  Fact,
  FactFilter,
  RecalledFact,
  RecalledSummary,
} from './types';

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
  } catch {
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
  filter?: FactFilter
): Promise<RecalledFact[]> {
  if (queryVec.length === 0) return [];

  const effectiveFilter: FactFilter = {
    ...filter,
    excludeExpired: true,
  };

  if (maxCandidates <= 0) maxCandidates = 10000;

  const facts = await store.listFactsForRecall(entityUuid, effectiveFilter, maxCandidates);
  if (facts.length === 0) return [];

  const candidates = buildRecallCandidates(queryVec, queryModel, facts);
  const results = engine.rank(queryVec, queryText, candidates, limit, threshold);

  return results.map((r) => ({
    id: r.id,
    content: r.content,
    score: r.score,
    temporalStatus: r.temporalStatus,
    significance: r.significance,
    createdAt: r.createdAt,
  }));
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
  } catch {
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
    threshold
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
  threshold: number
): Promise<RecalledSummary[]> {
  if (queryVec.length === 0) return [];

  const convs = await store.listConversationSummaries(entityUuid, 0);
  if (convs.length === 0) return [];

  const candidates: SearchCandidate[] = convs.map((c) => {
    let embedding = c.summaryEmbedding;
    if (queryVec.length > 0 && !dimensionMatch(queryVec, embedding)) {
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
    };
  });
}
