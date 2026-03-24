/**
 * SQLite-backed repository for the native MemG engine.
 * Uses better-sqlite3 (synchronous API) for fast, in-process persistence.
 */
import type { Fact, FactConversation, FactEntity, FactFilter, FactMessage, FactSession } from './types.js';
/** A value that may be synchronous or a Promise. */
type MaybeAsync<T> = T | Promise<T>;
/**
 * Store interface for pluggable persistence backends.
 *
 * Methods return `T | Promise<T>` so both synchronous stores (SQLite)
 * and asynchronous stores (Postgres, MySQL) can implement this interface.
 * All callsites use `await` — this is a no-op for sync values.
 *
 * Built-in implementations: `MemGStore` (SQLite), `PostgresStore`, `MySQLStore`.
 */
export interface Store {
    upsertEntity(externalId: string): MaybeAsync<string>;
    lookupEntity(externalId: string): MaybeAsync<FactEntity | null>;
    insertFact(entityUuid: string, fact: Fact): MaybeAsync<void>;
    insertFacts(entityUuid: string, facts: Fact[]): MaybeAsync<void>;
    listFacts(entityUuid: string, limit: number): MaybeAsync<Fact[]>;
    listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): MaybeAsync<Fact[]>;
    listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): MaybeAsync<Fact[]>;
    findFactByKey(entityUuid: string, contentKey: string): MaybeAsync<Fact | null>;
    reinforceFact(factUuid: string, newExpiresAt: string | null): MaybeAsync<void>;
    updateTemporalStatus(factUuid: string, status: string): MaybeAsync<void>;
    updateSignificance(factUuid: string, sig: number): MaybeAsync<void>;
    deleteFact(entityUuid: string, factUuid: string): MaybeAsync<void>;
    deleteEntityFacts(entityUuid: string): MaybeAsync<number>;
    pruneExpiredFacts(entityUuid: string, now: string): MaybeAsync<number>;
    updateRecallUsage(factUuids: string[]): MaybeAsync<void>;
    listUnembeddedFacts(entityUuid: string, limit: number): MaybeAsync<Fact[]>;
    updateFactEmbedding(factUuid: string, embedding: number[], model: string): MaybeAsync<void>;
    ensureSession(entityUuid: string, processUuid: string, timeoutMs: number): MaybeAsync<{
        session: FactSession;
        isNew: boolean;
    }>;
    startConversation(sessionUuid: string, entityUuid: string): MaybeAsync<string>;
    activeConversation(sessionUuid: string): MaybeAsync<FactConversation | null>;
    appendMessage(conversationUuid: string, msg: FactMessage): MaybeAsync<void>;
    readMessages(conversationUuid: string): MaybeAsync<FactMessage[]>;
    readRecentMessages(conversationUuid: string, limit: number): MaybeAsync<FactMessage[]>;
    listConversationSummaries(entityUuid: string, limit: number): MaybeAsync<FactConversation[]>;
    updateConversationSummary(conversationUuid: string, summary: string, embedding: number[]): MaybeAsync<void>;
    findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): MaybeAsync<FactConversation | null>;
    close(): MaybeAsync<void>;
}
export declare class MemGStore implements Store {
    private db;
    constructor(dbPath: string);
    private createSchema;
    upsertEntity(externalId: string): string;
    lookupEntity(externalId: string): FactEntity | null;
    insertFact(entityUuid: string, fact: Fact): void;
    insertFacts(entityUuid: string, facts: Fact[]): void;
    listFacts(entityUuid: string, limit: number): Fact[];
    listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): Fact[];
    listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): Fact[];
    findFactByKey(entityUuid: string, contentKey: string): Fact | null;
    reinforceFact(factUuid: string, newExpiresAt: string | null): void;
    updateTemporalStatus(factUuid: string, status: string): void;
    updateSignificance(factUuid: string, sig: number): void;
    deleteFact(entityUuid: string, factUuid: string): void;
    deleteEntityFacts(entityUuid: string): number;
    pruneExpiredFacts(entityUuid: string, now: string): number;
    updateRecallUsage(factUuids: string[]): void;
    ensureSession(entityUuid: string, processUuid: string, timeoutMs: number): {
        session: FactSession;
        isNew: boolean;
    };
    startConversation(sessionUuid: string, entityUuid: string): string;
    activeConversation(sessionUuid: string): FactConversation | null;
    appendMessage(conversationUuid: string, msg: FactMessage): void;
    readMessages(conversationUuid: string): FactMessage[];
    readRecentMessages(conversationUuid: string, limit: number): FactMessage[];
    listConversationSummaries(entityUuid: string, limit: number): FactConversation[];
    updateConversationSummary(conversationUuid: string, summary: string, embedding: number[]): void;
    findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): FactConversation | null;
    listUnembeddedFacts(entityUuid: string, limit: number): Fact[];
    updateFactEmbedding(factUuid: string, embedding: number[], model: string): void;
    close(): void;
}
/**
 * Compute a content key from fact content. Matches Go's DefaultContentKey:
 * lowercase, strip punctuation, collapse whitespace, SHA-256, first 16 hex chars.
 */
export declare function defaultContentKey(content: string): string;
export {};
