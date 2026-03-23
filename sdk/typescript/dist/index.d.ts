/**
 * MemG TypeScript SDK
 *
 * A memory layer for LLM applications. Provides:
 * - Native in-process engine (SQLite + local embeddings + hybrid search + extraction)
 * - Proxy-mode wrapping (redirects LLM traffic through MemG proxy)
 * - Client-mode interception (local memory injection via MCP)
 * - Direct memory operations via MCP (add, search, list, delete)
 *
 * @example Native mode (recommended — no Go server needed)
 * ```typescript
 * import { MemG } from 'memg';
 * import OpenAI from 'openai';
 *
 * const openai = MemG.wrap(new OpenAI(), {
 *   entity: 'user-123',
 *   mode: 'native',
 *   nativeConfig: { openaiApiKey: process.env.OPENAI_API_KEY },
 * });
 * const response = await openai.chat.completions.create({ ... });
 * ```
 *
 * @example Direct memory operations
 * ```typescript
 * import { MemG } from 'memg';
 *
 * const m = new MemG({ dbPath: './my-memory.db' });
 * await m.init();
 * await m.add('user-123', 'likes coffee');
 * const results = await m.search('user-123', 'beverage preferences');
 * m.close();
 * ```
 */
import type { Embedder } from './embedder';
import { type ExtractionMessage } from './extract';
import { HybridEngine } from './search';
import { type Store } from './store';
import type { AddResult, MemGOptions, Memory, MemoryInput, NativeConfig, SearchResult, WrapOptions } from './types';
export { MemGClient } from './client';
export { wrapOpenAIProxy, wrapAnthropicProxy } from './proxy';
export { wrapOpenAIClient, wrapAnthropicClient, wrapGeminiClient } from './intercept';
export { detectProvider } from './providers';
export { MemGStore, defaultContentKey, type Store } from './store';
export { PostgresStore } from './postgres_store';
export { MySQLStore } from './mysql_store';
export { HybridEngine, cosineSimilarity, dimensionMatch } from './search';
export { buildContext, estimateTokens } from './context';
export { runExtraction, isTrivialTurn } from './extract';
export { recallFacts, recallSummaries } from './recall';
export { TransformersEmbedder, OpenAIEmbedder, GeminiEmbedder } from './embedder';
export type { Embedder } from './embedder';
export type { Memory, MemoryInput, AddResult, SearchResult, WrapOptions, MemGOptions, NativeConfig, Fact, FactEntity, FactSession, FactConversation, FactMessage, FactFilter, RecalledFact, RecalledSummary, ConsciousFact, } from './types';
/**
 * Main MemG class providing memory operations and client wrapping.
 * Supports native in-process engine, proxy mode, and MCP client mode.
 */
export declare class MemG {
    private store;
    private embedder;
    private engine;
    private config;
    private _mcpClient;
    private _proxyUrl;
    private _initialized;
    constructor(opts?: NativeConfig & MemGOptions);
    /**
     * Initialize the native engine. Must be called before using native operations.
     * Creates the SQLite database and sets up the embedder.
     */
    init(): Promise<void>;
    /**
     * Wrap an LLM client with MemG memory capabilities.
     *
     * @param client - An OpenAI or Anthropic SDK client instance.
     * @param opts - Wrapping options (mode, entity, URLs, extract, nativeConfig).
     * @returns The wrapped client with the same type as the input.
     */
    static wrap<T>(client: T, opts?: WrapOptions): T;
    /**
     * Store memories for an entity.
     */
    add(entityId: string, contentOrMemories: string | string[] | MemoryInput[], opts?: {
        type?: Memory['type'];
        significance?: Memory['significance'];
        tag?: string;
    }): Promise<AddResult>;
    /**
     * Search memories for an entity using semantic hybrid search.
     */
    search(entityId: string, query: string, limit?: number): Promise<SearchResult>;
    /**
     * List stored memories for an entity.
     */
    list(entityId: string, opts?: {
        limit?: number;
        type?: string;
        tag?: string;
    }): Promise<SearchResult>;
    /**
     * Delete a specific memory by its ID.
     */
    delete(entityId: string, memoryId: string): Promise<boolean>;
    /**
     * Delete all memories for an entity.
     */
    deleteAll(entityId: string): Promise<number>;
    /**
     * Memory-augmented chat. Handles the full loop:
     * recall → inject context → call LLM → save exchange → extract facts.
     *
     * No external LLM client needed — calls the API directly via fetch.
     *
     * @param messages - Conversation messages (OpenAI format).
     * @param entityId - Entity identifier for memory scoping.
     * @param opts - Optional overrides for model, provider, maxTokens.
     * @returns Object with content, role, and memoryContext.
     */
    chat(messages: Array<{
        role: string;
        content: string;
    }>, entityId: string, opts?: {
        model?: string;
        provider?: string;
        maxTokens?: number;
    }): Promise<{
        content: string;
        role: string;
        memoryContext: string;
    }>;
    private resolveChatApiKey;
    /**
     * Get the underlying store for advanced operations.
     */
    getStore(): Store | null;
    /**
     * Get the underlying embedder.
     */
    getEmbedder(): Embedder | null;
    /**
     * Get the search engine.
     */
    getEngine(): HybridEngine;
    /**
     * Get the resolved config.
     */
    getConfig(): Required<NativeConfig>;
    /**
     * Build memory context for injection into LLM prompts.
     */
    buildMemoryContext(entityId: string, queryText: string): Promise<string>;
    /**
     * Run the extraction pipeline on messages.
     * Fire-and-forget: errors are logged, not thrown.
     */
    extractFromMessages(entityId: string, messages: ExtractionMessage[]): Promise<void>;
    /**
     * Resolve the API key for the extraction LLM provider.
     */
    private resolveExtractionApiKey;
    /**
     * Close the database connection.
     */
    close(): void;
    /**
     * Load conscious facts (highest significance, always injected).
     * Matches Go LoadConsciousContext.
     */
    private loadConsciousFacts;
    private ensureInitialized;
}
