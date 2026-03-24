/**
 * PostgreSQL-backed repository for the native MemG engine.
 * Uses the `pg` npm package (async API) for networked persistence.
 */
import type { Fact, FactConversation, FactEntity, FactFilter, FactMessage, FactSession } from './types.js';
import type { Store } from './store.js';
export declare class PostgresStore implements Store {
    private pool;
    private initialized;
    private constructor();
    static create(connectionString: string): Promise<PostgresStore>;
    private createSchema;
    private ready;
    upsertEntity(externalId: string): Promise<string>;
    lookupEntity(externalId: string): Promise<FactEntity | null>;
    insertFact(entityUuid: string, fact: Fact): Promise<void>;
    insertFacts(entityUuid: string, facts: Fact[]): Promise<void>;
    listFacts(entityUuid: string, limit: number): Promise<Fact[]>;
    listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]>;
    listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]>;
    findFactByKey(entityUuid: string, contentKey: string): Promise<Fact | null>;
    reinforceFact(factUuid: string, newExpiresAt: string | null): Promise<void>;
    updateTemporalStatus(factUuid: string, status: string): Promise<void>;
    updateSignificance(factUuid: string, sig: number): Promise<void>;
    deleteFact(entityUuid: string, factUuid: string): Promise<void>;
    deleteEntityFacts(entityUuid: string): Promise<number>;
    pruneExpiredFacts(entityUuid: string, now: string): Promise<number>;
    updateRecallUsage(factUuids: string[]): Promise<void>;
    ensureSession(entityUuid: string, processUuid: string, timeoutMs: number): Promise<{
        session: FactSession;
        isNew: boolean;
    }>;
    startConversation(sessionUuid: string, entityUuid: string): Promise<string>;
    activeConversation(sessionUuid: string): Promise<FactConversation | null>;
    appendMessage(conversationUuid: string, msg: FactMessage): Promise<void>;
    readMessages(conversationUuid: string): Promise<FactMessage[]>;
    readRecentMessages(conversationUuid: string, limit: number): Promise<FactMessage[]>;
    listConversationSummaries(entityUuid: string, limit: number): Promise<FactConversation[]>;
    updateConversationSummary(conversationUuid: string, summary: string, embedding: number[]): Promise<void>;
    findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): Promise<FactConversation | null>;
    listUnembeddedFacts(entityUuid: string, limit: number): Promise<Fact[]>;
    updateFactEmbedding(factUuid: string, embedding: number[], model: string): Promise<void>;
    close(): Promise<void>;
}
