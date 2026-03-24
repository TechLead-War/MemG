/**
 * Recall: retrieve relevant facts and summaries using hybrid search.
 * Matches the Go memory.Recall and memory.RecallSummaries logic.
 */
import type { Embedder } from './embedder.js';
import { HybridEngine } from './search.js';
import type { Store } from './store.js';
import type { FactFilter, RecalledFact, RecalledSummary } from './types.js';
/**
 * Recall facts for an entity using hybrid search.
 */
export declare function recallFacts(store: Store, engine: HybridEngine, embedder: Embedder, queryText: string, entityUuid: string, limit: number, threshold: number, maxCandidates: number, filter?: FactFilter): Promise<RecalledFact[]>;
/**
 * Recall facts using a pre-computed query vector.
 */
export declare function recallFactsWithVector(store: Store, engine: HybridEngine, queryVec: number[], queryModel: string, queryText: string, entityUuid: string, limit: number, threshold: number, maxCandidates: number, filter?: FactFilter): Promise<RecalledFact[]>;
/**
 * Recall conversation summaries using hybrid search.
 */
export declare function recallSummaries(store: Store, engine: HybridEngine, embedder: Embedder, queryText: string, entityUuid: string, limit: number, threshold: number): Promise<RecalledSummary[]>;
/**
 * Recall summaries using a pre-computed query vector.
 */
export declare function recallSummariesWithVector(store: Store, engine: HybridEngine, queryVec: number[], queryText: string, entityUuid: string, limit: number, threshold: number): Promise<RecalledSummary[]>;
