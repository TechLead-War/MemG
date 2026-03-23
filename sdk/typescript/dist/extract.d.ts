/**
 * Extraction pipeline: extract structured facts from conversation messages.
 * Matches the Go proxy/extract.go logic exactly.
 */
import type { Embedder } from './embedder';
import { type Store } from './store';
/** Message shape for extraction. */
export interface ExtractionMessage {
    role: string;
    content: string;
}
/**
 * Check if a set of messages is trivial (greetings, acknowledgments).
 */
export declare function isTrivialTurn(messages: ExtractionMessage[]): boolean;
/**
 * Run the full extraction pipeline:
 * 1. Trivial turn detection
 * 2. LLM extraction call
 * 3. Parse and validate
 * 4. Embed extracted facts
 * 5. Dedup (content key + semantic)
 * 6. Slot conflict resolution
 * 7. TTL assignment
 * 8. Persist
 */
export declare function runExtraction(store: Store, embedder: Embedder | null, messages: ExtractionMessage[], entityUuid: string, apiKey: string, llmModel: string, llmProvider: string): Promise<{
    inserted: number;
    reinforced: number;
}>;
