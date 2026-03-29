import type { AddResult, Memory, MemoryInput, SearchResult } from './types.js';

/**
 * MCP JSON-RPC 2.0 client for communicating with the MemG server.
 *
 * Uses native `fetch` (Node 18+). No external dependencies.
 */
export class MemGClient {
  private _url: string;
  private _nextId: number;

  /**
   * Create a new MCP client.
   * @param mcpUrl - Base URL of the MemG MCP server (default: http://localhost:8686).
   */
  constructor(mcpUrl: string = 'http://localhost:8686') {
    // Ensure the URL ends with /mcp for the JSON-RPC endpoint.
    this._url = mcpUrl.replace(/\/+$/, '') + '/mcp';
    this._nextId = 1;
  }

  /**
   * Store new memories for an entity.
   * Duplicate content is automatically deduplicated and reinforced server-side.
   */
  async add(entityId: string, memories: MemoryInput[]): Promise<AddResult> {
    const result = await this._call<{ inserted: number; reinforced: number }>('add_memories', {
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
  async search(entityId: string, query: string, limit: number = 10): Promise<SearchResult> {
    const result = await this._call<{ memories: RawMemory[]; count: number }>('search_memories', {
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
  async list(
    entityId: string,
    opts?: { limit?: number; type?: string; tag?: string }
  ): Promise<SearchResult> {
    const args: Record<string, unknown> = { entity_id: entityId };
    if (opts?.limit !== undefined) args.limit = opts.limit;
    if (opts?.type !== undefined) args.type = opts.type;
    if (opts?.tag !== undefined) args.tag = opts.tag;

    const result = await this._call<{ memories: RawMemory[]; count: number }>('list_memories', args);
    return {
      memories: result.memories.map(normalizeMemory),
      count: result.count,
    };
  }

  /**
   * Delete a specific memory by its ID.
   * @returns true if the deletion succeeded.
   */
  async delete(entityId: string, memoryId: string): Promise<boolean> {
    const result = await this._call<{ deleted: boolean }>('delete_memory', {
      entity_id: entityId,
      memory_id: memoryId,
    });
    return result.deleted;
  }

  /**
   * Delete all memories for an entity. This action is irreversible.
   * @returns The number of memories deleted.
   */
  async deleteAll(entityId: string): Promise<number> {
    const result = await this._call<{ deleted: number }>('delete_all_memories', {
      entity_id: entityId,
    });
    return result.deleted;
  }

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
  async extractFromMessages(
    entityId: string,
    messages: Array<{ role: string; content: string }>
  ): Promise<number> {
    const result = await this._call<{ extracted: number }>('extract_from_messages', {
      entity_id: entityId,
      messages,
    });
    return result.extracted;
  }

  /**
   * Send a JSON-RPC 2.0 tools/call request and return the parsed result.
   */
  private async _call<T>(toolName: string, args: Record<string, unknown>): Promise<T> {
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

    const rpcResponse = (await response.json()) as JsonRpcResponse;

    // JSON-RPC level error.
    if (rpcResponse.error) {
      throw new Error(`MemG MCP error [${rpcResponse.error.code}]: ${rpcResponse.error.message}`);
    }

    const result = rpcResponse.result as ToolResult | undefined;
    if (!result || !result.content || result.content.length === 0) {
      throw new Error('MemG MCP returned empty result');
    }

    // Tool-level error (HTTP 200 but isError: true).
    if (result.isError) {
      throw new Error(`MemG tool error: ${result.content[0].text}`);
    }

    return JSON.parse(result.content[0].text) as T;
  }
}

// --- Internal types ---

interface JsonRpcResponse {
  jsonrpc: string;
  id: number | string | null;
  result?: unknown;
  error?: { code: number; message: string };
}

interface ToolResult {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
}

interface RawMemory {
  id: string;
  content: string;
  type?: string;
  temporal_status?: string;
  significance?: string;
  created_at?: string;
  tag?: string;
  score?: number;
  reinforced_count?: number;
}

/** Normalize snake_case server response to camelCase Memory interface. */
function normalizeMemory(raw: RawMemory): Memory {
  return {
    id: raw.id,
    content: raw.content,
    type: (raw.type as Memory['type']) ?? 'identity',
    temporalStatus: (raw.temporal_status as Memory['temporalStatus']) ?? 'current',
    significance: (raw.significance as Memory['significance']) ?? 'medium',
    createdAt: raw.created_at,
    tag: raw.tag,
    score: raw.score,
    reinforcedCount: raw.reinforced_count,
  };
}
