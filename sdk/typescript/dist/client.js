"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MemGClient = void 0;
/**
 * MCP JSON-RPC 2.0 client for communicating with the MemG server.
 *
 * Uses native `fetch` (Node 18+). No external dependencies.
 */
class MemGClient {
    /**
     * Create a new MCP client.
     * @param mcpUrl - Base URL of the MemG MCP server (default: http://localhost:8686).
     */
    constructor(mcpUrl = 'http://localhost:8686') {
        // Ensure the URL ends with /mcp for the JSON-RPC endpoint.
        this._url = mcpUrl.replace(/\/+$/, '') + '/mcp';
        this._nextId = 1;
    }
    /**
     * Store new memories for an entity.
     * Duplicate content is automatically deduplicated and reinforced server-side.
     */
    async add(entityId, memories) {
        const result = await this._call('add_memories', {
            entity_id: entityId,
            memories,
        });
        return { inserted: result.inserted, reinforced: result.reinforced };
    }
    /**
     * Search memories for an entity using semantic hybrid search.
     * @param entityId - External entity identifier.
     * @param query - Natural language search query.
     * @param limit - Maximum results to return (default: 10).
     */
    async search(entityId, query, limit = 10) {
        const result = await this._call('search_memories', {
            entity_id: entityId,
            query,
            limit,
        });
        return {
            memories: result.memories.map(normalizeMemory),
            count: result.count,
        };
    }
    /**
     * List stored memories for an entity, optionally filtered.
     * @param entityId - External entity identifier.
     * @param opts - Optional filters: limit, type, tag.
     */
    async list(entityId, opts) {
        const args = { entity_id: entityId };
        if (opts?.limit !== undefined)
            args.limit = opts.limit;
        if (opts?.type !== undefined)
            args.type = opts.type;
        if (opts?.tag !== undefined)
            args.tag = opts.tag;
        const result = await this._call('list_memories', args);
        return {
            memories: result.memories.map(normalizeMemory),
            count: result.count,
        };
    }
    /**
     * Delete a specific memory by its ID.
     * @returns true if the deletion succeeded.
     */
    async delete(entityId, memoryId) {
        const result = await this._call('delete_memory', {
            entity_id: entityId,
            memory_id: memoryId,
        });
        return result.deleted;
    }
    /**
     * Delete all memories for an entity. This action is irreversible.
     * @returns The number of memories deleted.
     */
    async deleteAll(entityId) {
        const result = await this._call('delete_all_memories', {
            entity_id: entityId,
        });
        return result.deleted;
    }
    /**
     * Send a JSON-RPC 2.0 tools/call request and return the parsed result.
     */
    async _call(toolName, args) {
        const id = this._nextId++;
        const body = JSON.stringify({
            jsonrpc: '2.0',
            id,
            method: 'tools/call',
            params: {
                name: toolName,
                arguments: args,
            },
        });
        const response = await fetch(this._url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
        });
        if (!response.ok) {
            throw new Error(`MemG MCP request failed: HTTP ${response.status} ${response.statusText}`);
        }
        const rpcResponse = (await response.json());
        // JSON-RPC level error.
        if (rpcResponse.error) {
            throw new Error(`MemG MCP error [${rpcResponse.error.code}]: ${rpcResponse.error.message}`);
        }
        const result = rpcResponse.result;
        if (!result || !result.content || result.content.length === 0) {
            throw new Error('MemG MCP returned empty result');
        }
        // Tool-level error (HTTP 200 but isError: true).
        if (result.isError) {
            throw new Error(`MemG tool error: ${result.content[0].text}`);
        }
        return JSON.parse(result.content[0].text);
    }
}
exports.MemGClient = MemGClient;
/** Normalize snake_case server response to camelCase Memory interface. */
function normalizeMemory(raw) {
    return {
        id: raw.id,
        content: raw.content,
        type: raw.type ?? 'identity',
        temporalStatus: raw.temporal_status ?? 'current',
        significance: raw.significance ?? 'medium',
        createdAt: raw.created_at,
        tag: raw.tag,
        score: raw.score,
        reinforcedCount: raw.reinforced_count,
    };
}
//# sourceMappingURL=client.js.map