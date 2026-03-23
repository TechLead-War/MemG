/**
 * Hybrid vector + lexical search engine.
 * Exact port of the Go search package.
 */
/** A candidate for ranking. */
export interface SearchCandidate {
    id: string;
    content: string;
    embedding?: number[];
    createdAt?: string;
    temporalStatus: string;
    significance: number;
    confidence: number;
}
/** Cosine similarity between two vectors. */
export declare function cosineSimilarity(a: number[], b: number[]): number;
/** Check if two vectors have compatible dimensions. */
export declare function dimensionMatch(a: number[] | undefined, b: number[] | undefined): boolean;
export interface RankResult {
    id: string;
    content: string;
    score: number;
    createdAt?: string;
    temporalStatus: string;
    significance: number;
}
/**
 * Hybrid search engine. Ranks candidates using weighted combination of
 * cosine similarity and BM25 lexical relevance.
 */
export declare class HybridEngine {
    rank(query: number[], queryText: string, candidates: SearchCandidate[], limit: number, threshold: number): RankResult[];
}
