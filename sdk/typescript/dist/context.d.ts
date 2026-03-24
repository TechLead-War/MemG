/**
 * Context builder: merges conscious facts, recalled facts, and summaries
 * into a single context string under a token budget. Matches Go's BuildContext.
 */
import type { ConsciousFact, RecalledFact, RecalledSummary } from './types.js';
export interface ContextBudget {
    totalTokens: number;
    summaryTokens: number;
}
export interface ContextInput {
    consciousFacts: ConsciousFact[];
    recalledFacts: RecalledFact[];
    summaries: RecalledSummary[];
    budget: ContextBudget;
}
/**
 * Token estimation: ceil(wordCount * 1.3).
 * For CJK text with few spaces, falls back to rune-based estimation.
 */
export declare function estimateTokens(s: string): number;
/**
 * Build context string. Priority: conscious > recalled > summaries.
 * Dedup by normalized content across all components.
 */
export declare function buildContext(input: ContextInput): string;
