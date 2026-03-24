import type { AddResult, MemoryInput, SearchResult } from './types.js';
/**
 * MCP JSON-RPC 2.0 client for communicating with the MemG server.
 *
 * Uses native `fetch` (Node 18+). No external dependencies.
 */
export declare class MemGClient {
    private _url;
    private _nextId;
    /**
     * Create a new MCP client.
     * @param mcpUrl - Base URL of the MemG MCP server (default: http://localhost:8686).
     */
    constructor(mcpUrl?: string);
    /**
     * Store new memories for an entity.
     * Duplicate content is automatically deduplicated and reinforced server-side.
     */
    add(entityId: string, memories: MemoryInput[]): Promise<AddResult>;
    /**
     * Search memories for an entity using semantic hybrid search.
     * @param entityId - External entity identifier.
     * @param query - Natural language search query.
     * @param limit - Maximum results to return (default: 10).
     */
    search(entityId: string, query: string, limit?: number): Promise<SearchResult>;
    /**
     * List stored memories for an entity, optionally filtered.
     * @param entityId - External entity identifier.
     * @param opts - Optional filters: limit, type, tag.
     */
    list(entityId: string, opts?: {
        limit?: number;
        type?: string;
        tag?: string;
    }): Promise<SearchResult>;
    /**
     * Delete a specific memory by its ID.
     * @returns true if the deletion succeeded.
     */
    delete(entityId: string, memoryId: string): Promise<boolean>;
    /**
     * Delete all memories for an entity. This action is irreversible.
     * @returns The number of memories deleted.
     */
    deleteAll(entityId: string): Promise<number>;
    /**
     * Extract structured memories from conversation messages using the server's
     * LLM-powered extraction pipeline. Produces typed, tagged, and embedded facts
     * with the same quality as proxy mode.
     *
     * Requires the MCP server to be started with --llm-provider.
     *
     * @param entityId - External entity identifier.
     * @param messages - Conversation messages to extract knowledge from.
     * @returns The number of facts extracted and stored.
     */
    extractFromMessages(entityId: string, messages: Array<{
        role: string;
        content: string;
    }>): Promise<number>;
    /**
     * Send a JSON-RPC 2.0 tools/call request and return the parsed result.
     */
    private _call;
}
