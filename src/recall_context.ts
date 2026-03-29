/**
 * RecallAndBuildContext: single entry point for the full memory recall pipeline.
 * Matches Go memory/recall_context.go.
 */

import type { Embedder } from './embedder.js';
import type { Store } from './store.js';
import { HybridEngine } from './search.js';
import { recallFactsWithVector, recallSummariesWithVector } from './recall.js';
import { loadConsciousContext } from './conscious.js';
import { buildContext } from './context.js';
import type { TurnSummary, Artifact, RecalledSummary, ConsciousFact, RecalledFact } from './types.js';

export interface RecallConfig {
  recallLimit?: number;
  recallThreshold?: number;
  maxCandidates?: number;
  memoryTokenBudget?: number;
  summaryTokenBudget?: number;
  consciousLimit?: number;
  summaryLimit?: number;
  conversationId?: string;
}

/**
 * Single entry point for the full memory recall pipeline.
 *
 * 1. Embeds the query (once, reused for all recall passes)
 * 2. Loads conscious facts (user profile, always present)
 * 3. Recalls relevant facts via hybrid search
 * 4. Recalls relevant conversation summaries
 * 5. Loads turn summaries for the active conversation
 * 6. Loads relevant artifacts
 * 7. Assembles everything via buildContext with token budgeting
 *
 * Callers should not orchestrate these steps manually.
 */
export async function recallAndBuildContext(
  store: Store,
  embedder: Embedder,
  entityUuid: string,
  query: string,
  cfg?: RecallConfig
): Promise<string> {
  const recallLimit = cfg?.recallLimit ?? 50;
  const recallThreshold = cfg?.recallThreshold ?? 0.05;
  const maxCandidates = cfg?.maxCandidates ?? 50;
  const memoryTokenBudget = cfg?.memoryTokenBudget ?? 4000;
  const summaryTokenBudget = cfg?.summaryTokenBudget ?? 1000;
  const consciousLimit = cfg?.consciousLimit ?? 10;
  const summaryLimit = cfg?.summaryLimit ?? 5;

  const engine = new HybridEngine();

  // Step 1: Embed query once.
  let vectors: number[][];
  try {
    vectors = await embedder.embed([query]);
  } catch {
    return '';
  }
  if (vectors.length === 0 || vectors[0].length === 0) return '';
  const queryVec = vectors[0];
  const queryModel = embedder.modelName();

  // Step 2: Conscious facts.
  let consciousFacts: ConsciousFact[] = [];
  try {
    consciousFacts = await loadConsciousContext(store, entityUuid, consciousLimit);
  } catch (e) {
    console.warn('memg: conscious context load failed:', e);
  }

  // Step 3: Recalled facts.
  let recalledFacts: RecalledFact[] = [];
  try {
    recalledFacts = await recallFactsWithVector(
      store, engine, queryVec, queryModel, query, entityUuid,
      recallLimit, recallThreshold, maxCandidates,
      undefined, embedder
    );
  } catch (e) {
    console.warn('memg: recall facts failed:', e);
  }

  // Step 4: Recalled summaries.
  let recalledSummaries: RecalledSummary[] = [];
  try {
    recalledSummaries = await recallSummariesWithVector(
      store, engine, queryVec, query, entityUuid,
      summaryLimit, recallThreshold, queryModel
    );
  } catch (e) {
    console.warn('memg: recall summaries failed:', e);
  }

  // Step 5: Turn summaries.
  let turnSummaries: TurnSummary[] = [];
  if (cfg?.conversationId) {
    try {
      turnSummaries = await store.listTurnSummaries(cfg.conversationId);
    } catch (e) {
      console.warn('memg: turn summaries load failed:', e);
    }
  }

  // Step 6: Artifacts.
  let artifacts: Artifact[] = [];
  if (cfg?.conversationId) {
    try {
      artifacts = await store.listActiveArtifacts(entityUuid, cfg.conversationId);
    } catch (e) {
      console.warn('memg: artifacts load failed:', e);
    }
  }

  // Step 7: Build context.
  return buildContext({
    consciousFacts,
    recalledFacts,
    summaries: recalledSummaries.map((s) => ({
      conversationId: s.conversationId,
      summary: s.summary,
      score: s.score,
      createdAt: s.createdAt,
    })),
    turnSummaries,
    artifacts,
    budget: {
      totalTokens: memoryTokenBudget,
      summaryTokens: summaryTokenBudget,
    },
  });
}
