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

import { MemGClient } from './client';
import { resolveConfig, DEFAULT_CONFIG } from './config';
import { buildContext, type ContextInput } from './context';
import type { Embedder } from './embedder';
import { runExtraction, type ExtractionMessage } from './extract';
import { detectProvider } from './providers';
import { recallFacts, recallSummaries } from './recall';
import { HybridEngine } from './search';
import { MemGStore, defaultContentKey, type Store } from './store';
import type {
  AddResult,
  ConsciousFact,
  Fact,
  MemGOptions,
  Memory,
  MemoryInput,
  NativeConfig,
  RecalledFact,
  RecalledSummary,
  SearchResult,
  WrapOptions,
} from './types';

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
export type {
  Memory,
  MemoryInput,
  AddResult,
  SearchResult,
  WrapOptions,
  MemGOptions,
  NativeConfig,
  Fact,
  FactEntity,
  FactSession,
  FactConversation,
  FactMessage,
  FactFilter,
  RecalledFact,
  RecalledSummary,
  ConsciousFact,
} from './types';

/**
 * Main MemG class providing memory operations and client wrapping.
 * Supports native in-process engine, proxy mode, and MCP client mode.
 */
export class MemG {
  private store: Store | null = null;
  private embedder: Embedder | null = null;
  private engine: HybridEngine;
  private config: Required<NativeConfig>;
  private _mcpClient: MemGClient | null = null;
  private _proxyUrl: string;
  private _initialized = false;

  constructor(opts?: NativeConfig & MemGOptions) {
    this.config = resolveConfig(opts);
    this.engine = new HybridEngine();
    this._proxyUrl = (opts as any)?.proxyUrl ?? 'http://localhost:8787/v1';
  }

  /**
   * Initialize the native engine. Must be called before using native operations.
   * Creates the SQLite database and sets up the embedder.
   */
  async init(): Promise<void> {
    if (this._initialized) return;

    if (this.config.store) {
      this.store = this.config.store;
    } else if (this.config.storeProvider === 'postgres') {
      if (!this.config.storeUrl) throw new Error('storeUrl required for postgres');
      const { PostgresStore } = require('./postgres_store');
      this.store = await PostgresStore.create(this.config.storeUrl);
    } else if (this.config.storeProvider === 'mysql') {
      if (!this.config.storeUrl) throw new Error('storeUrl required for mysql');
      const { MySQLStore } = require('./mysql_store');
      this.store = await MySQLStore.create(this.config.storeUrl);
    } else {
      this.store = new MemGStore(this.config.dbPath);
    }

    if (this.config.embedProvider === 'gemini' && this.config.geminiApiKey) {
      const { GeminiEmbedder } = require('./embedder') as typeof import('./embedder');
      this.embedder = new GeminiEmbedder(
        this.config.geminiApiKey,
        this.config.embedModel || 'text-embedding-004',
        this.config.embedDimension || 768
      );
    } else if (this.config.embedProvider === 'openai' && this.config.openaiApiKey) {
      const { OpenAIEmbedder } = require('./embedder') as typeof import('./embedder');
      this.embedder = new OpenAIEmbedder(
        this.config.openaiApiKey,
        this.config.embedModel || 'text-embedding-3-small',
        this.config.embedDimension || 1536
      );
    } else if (this.config.embedProvider === 'sentence-transformers') {
      try {
        const { TransformersEmbedder } = require('./embedder') as typeof import('./embedder');
        this.embedder = await TransformersEmbedder.create(
          this.config.embedModel || 'Xenova/all-MiniLM-L6-v2'
        );
      } catch (err) {
        console.warn(
          '[memg] @huggingface/transformers not available, recall will use lexical-only search. Install it: npm i @huggingface/transformers'
        );
      }
    }

    this._initialized = true;
  }

  /**
   * Wrap an LLM client with MemG memory capabilities.
   *
   * @param client - An OpenAI or Anthropic SDK client instance.
   * @param opts - Wrapping options (mode, entity, URLs, extract, nativeConfig).
   * @returns The wrapped client with the same type as the input.
   */
  static wrap<T>(client: T, opts?: WrapOptions): T {
    const provider = detectProvider(client);

    const mergedOpts: WrapOptions = {
      mode: 'native',
      proxyUrl: 'http://localhost:8787/v1',
      mcpUrl: 'http://localhost:8686',
      extract: true,
      ...opts,
    };

    if (provider === 'openai') {
      const { wrap } = require('./providers/openai');
      return wrap(client, mergedOpts) as T;
    }

    if (provider === 'anthropic') {
      const { wrap } = require('./providers/anthropic');
      return wrap(client, mergedOpts) as T;
    }

    if (provider === 'gemini') {
      const { wrap } = require('./providers/gemini');
      return wrap(client, mergedOpts) as T;
    }

    throw new Error(
      `Unsupported client type: ${(client as any)?.constructor?.name}. Supported: OpenAI, Anthropic, Gemini`
    );
  }

  /**
   * Store memories for an entity.
   */
  async add(
    entityId: string,
    contentOrMemories: string | string[] | MemoryInput[],
    opts?: { type?: Memory['type']; significance?: Memory['significance']; tag?: string }
  ): Promise<AddResult> {
    this.ensureInitialized();

    let memories: MemoryInput[];
    if (typeof contentOrMemories === 'string') {
      memories = [{
        content: contentOrMemories,
        type: opts?.type,
        significance: opts?.significance,
        tag: opts?.tag,
      }];
    } else if (Array.isArray(contentOrMemories)) {
      if (contentOrMemories.length === 0) return { inserted: 0, reinforced: 0 };
      if (typeof contentOrMemories[0] === 'string') {
        memories = (contentOrMemories as string[]).map((content) => ({
          content,
          type: opts?.type,
          significance: opts?.significance,
          tag: opts?.tag,
        }));
      } else {
        memories = contentOrMemories as MemoryInput[];
      }
    } else {
      throw new Error('contentOrMemories must be a string, string[], or MemoryInput[]');
    }

    const entityUuid = await this.store!.upsertEntity(entityId);
    let inserted = 0;
    let reinforced = 0;

    for (const mem of memories) {
      const contentKey = defaultContentKey(mem.content);
      const existing = await this.store!.findFactByKey(entityUuid, contentKey);

      if (existing) {
        await this.store!.reinforceFact(existing.uuid, existing.expiresAt ?? null);
        reinforced++;
        continue;
      }

      const sig = significanceToNumber(mem.significance);
      const fact: Fact = {
        uuid: '',
        content: mem.content,
        factType: mem.type ?? 'identity',
        temporalStatus: 'current',
        significance: sig,
        contentKey,
        tag: mem.tag ?? '',
        slot: '',
        confidence: 1.0,
        embeddingModel: this.embedder?.modelName() ?? '',
        sourceRole: 'user',
        reinforcedCount: 0,
        recallCount: 0,
        expiresAt: ttlForSig(sig) ?? undefined,
      };

      // Embed the content.
      if (this.embedder) {
        try {
          const [vec] = await this.embedder.embed([mem.content]);
          fact.embedding = vec;
        } catch {
          // Skip embedding on failure.
        }
      }

      await this.store!.insertFact(entityUuid, fact);
      inserted++;
    }

    return { inserted, reinforced };
  }

  /**
   * Search memories for an entity using semantic hybrid search.
   */
  async search(entityId: string, query: string, limit?: number): Promise<SearchResult> {
    this.ensureInitialized();

    const entity = await this.store!.lookupEntity(entityId);
    if (!entity) return { memories: [], count: 0 };

    if (!this.embedder) {
      console.warn('[memg] no embedder available, search requires embeddings');
      return { memories: [], count: 0 };
    }

    const recalled = await recallFacts(
      this.store!,
      this.engine,
      this.embedder,
      query,
      entity.uuid,
      limit ?? this.config.recallLimit,
      this.config.recallThreshold,
      this.config.maxRecallCandidates
    );

    const memories: Memory[] = recalled.map((r) => ({
      id: r.id,
      content: r.content,
      type: 'identity' as const,
      temporalStatus: r.temporalStatus as 'current' | 'historical',
      significance: significanceToLabel(r.significance),
      createdAt: r.createdAt,
      score: r.score,
    }));

    // Update recall usage.
    if (recalled.length > 0) {
      await this.store!.updateRecallUsage(recalled.map((r) => r.id));
    }

    return { memories, count: memories.length };
  }

  /**
   * List stored memories for an entity.
   */
  async list(
    entityId: string,
    opts?: { limit?: number; type?: string; tag?: string }
  ): Promise<SearchResult> {
    this.ensureInitialized();

    const entity = await this.store!.lookupEntity(entityId);
    if (!entity) return { memories: [], count: 0 };

    const filter: import('./types').FactFilter = {};
    if (opts?.type) filter.types = [opts.type];
    if (opts?.tag) filter.tags = [opts.tag];

    const facts = await this.store!.listFactsFiltered(
      entity.uuid,
      filter,
      opts?.limit ?? 100
    );

    const memories: Memory[] = facts.map((f) => ({
      id: f.uuid,
      content: f.content,
      type: f.factType as Memory['type'],
      temporalStatus: f.temporalStatus as Memory['temporalStatus'],
      significance: significanceToLabel(f.significance),
      createdAt: f.createdAt,
      tag: f.tag || undefined,
      reinforcedCount: f.reinforcedCount,
    }));

    return { memories, count: memories.length };
  }

  /**
   * Delete a specific memory by its ID.
   */
  async delete(entityId: string, memoryId: string): Promise<boolean> {
    this.ensureInitialized();

    const entity = await this.store!.lookupEntity(entityId);
    if (!entity) return false;

    await this.store!.deleteFact(entity.uuid, memoryId);
    return true;
  }

  /**
   * Delete all memories for an entity.
   */
  async deleteAll(entityId: string): Promise<number> {
    this.ensureInitialized();

    const entity = await this.store!.lookupEntity(entityId);
    if (!entity) return 0;

    return await this.store!.deleteEntityFacts(entity.uuid);
  }

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
  async chat(
    messages: Array<{ role: string; content: string }>,
    entityId: string,
    opts?: { model?: string; provider?: string; maxTokens?: number }
  ): Promise<{ content: string; role: string; memoryContext: string }> {
    this.ensureInitialized();

    const provider = opts?.provider ?? this.config.llmProvider;
    const model = opts?.model ?? this.config.llmModel;
    const maxTokens = opts?.maxTokens ?? 4096;

    // Build memory context.
    const memoryContext = await this.buildMemoryContext(entityId, extractLastUser(messages) ?? '');

    // Inject context into messages.
    let augmented = [...messages];
    if (memoryContext) {
      const sysIdx = augmented.findIndex((m) => m.role === 'system');
      if (sysIdx >= 0) {
        augmented[sysIdx] = { ...augmented[sysIdx], content: augmented[sysIdx].content + '\n\n' + memoryContext };
      } else {
        augmented.unshift({ role: 'system', content: memoryContext });
      }
    }

    // Call the LLM.
    const apiKey = this.resolveChatApiKey(provider);
    if (!apiKey) {
      throw new Error(`No API key for provider "${provider}". Set openaiApiKey, geminiApiKey, or ANTHROPIC_API_KEY.`);
    }
    const content = await chatCallLLM(apiKey, model, augmented, provider, maxTokens);

    // Background extraction (fire-and-forget).
    const extractMessages = messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }));
    if (content) {
      extractMessages.push({ role: 'assistant', content });
    }
    this.extractFromMessages(entityId, extractMessages).catch(() => {});

    return { content, role: 'assistant', memoryContext };
  }

  private resolveChatApiKey(provider: string): string {
    if (provider === 'gemini') return this.config.geminiApiKey;
    if (provider === 'anthropic') return process.env.ANTHROPIC_API_KEY ?? '';
    return this.config.openaiApiKey;
  }

  /**
   * Get the underlying store for advanced operations.
   */
  getStore(): Store | null {
    return this.store;
  }

  /**
   * Get the underlying embedder.
   */
  getEmbedder(): Embedder | null {
    return this.embedder;
  }

  /**
   * Get the search engine.
   */
  getEngine(): HybridEngine {
    return this.engine;
  }

  /**
   * Get the resolved config.
   */
  getConfig(): Required<NativeConfig> {
    return this.config;
  }

  /**
   * Build memory context for injection into LLM prompts.
   */
  async buildMemoryContext(
    entityId: string,
    queryText: string
  ): Promise<string> {
    this.ensureInitialized();

    const entity = await this.store!.lookupEntity(entityId);
    if (!entity) return '';

    let consciousFacts: ConsciousFact[] = [];
    let recalledFactsList: RecalledFact[] = [];
    let summaries: RecalledSummary[] = [];

    // Load conscious facts if enabled.
    if (this.config.consciousMode) {
      consciousFacts = await this.loadConsciousFacts(entity.uuid, this.config.consciousLimit);
    }

    // Recall facts and summaries if embedder is available.
    if (this.embedder && queryText) {
      try {
        const [recalled, recalledSums] = await Promise.all([
          recallFacts(
            this.store!,
            this.engine,
            this.embedder,
            queryText,
            entity.uuid,
            this.config.recallLimit,
            this.config.recallThreshold,
            this.config.maxRecallCandidates
          ),
          recallSummaries(
            this.store!,
            this.engine,
            this.embedder,
            queryText,
            entity.uuid,
            10,
            this.config.recallThreshold
          ),
        ]);
        recalledFactsList = recalled;
        summaries = recalledSums;

        // Track recall usage.
        if (recalled.length > 0) {
          await this.store!.updateRecallUsage(recalled.map((r) => r.id));
        }
      } catch (err) {
        console.warn('[memg] recall failed:', err);
      }
    }

    return buildContext({
      consciousFacts,
      recalledFacts: recalledFactsList,
      summaries,
      budget: {
        totalTokens: this.config.memoryTokenBudget,
        summaryTokens: this.config.summaryTokenBudget,
      },
    });
  }

  /**
   * Run the extraction pipeline on messages.
   * Fire-and-forget: errors are logged, not thrown.
   */
  async extractFromMessages(
    entityId: string,
    messages: ExtractionMessage[]
  ): Promise<void> {
    if (!this.config.extract) return;

    // Resolve the API key for the configured LLM provider.
    const apiKey = this.resolveExtractionApiKey();
    if (!apiKey) return;

    try {
      const entityUuid = await this.store!.upsertEntity(entityId);
      await runExtraction(
        this.store!,
        this.embedder,
        messages,
        entityUuid,
        apiKey,
        this.config.llmModel,
        this.config.llmProvider
      );
    } catch (err) {
      console.warn('[memg] extraction failed:', err);
    }
  }

  /**
   * Resolve the API key for the extraction LLM provider.
   */
  private resolveExtractionApiKey(): string {
    const provider = this.config.llmProvider;
    if (provider === 'gemini') return this.config.geminiApiKey;
    if (provider === 'anthropic') return process.env.ANTHROPIC_API_KEY ?? '';
    return this.config.openaiApiKey; // openai and openai-compatible
  }

  /**
   * Close the database connection.
   */
  close(): void {
    if (this.store) {
      this.store.close();
      this.store = null;
    }
    this._initialized = false;
  }

  /**
   * Load conscious facts (highest significance, always injected).
   * Matches Go LoadConsciousContext.
   */
  private async loadConsciousFacts(entityUuid: string, limit: number): Promise<ConsciousFact[]> {
    if (limit <= 0) limit = 10;

    const filter = {
      statuses: ['current'],
      excludeExpired: true,
    };
    let fetchLimit = limit * 5;
    if (fetchLimit < 50) fetchLimit = 50;

    const facts = await this.store!.listFactsFiltered(entityUuid, filter, fetchLimit);
    if (facts.length === 0) return [];

    // Score and sort by significance (with staleness demotion for mutable facts).
    const now = Date.now();
    const scored = facts.map((f) => {
      let base = f.significance;

      if (f.factType === 'identity' && f.significance < 10) {
        let lastConfirmed = f.createdAt ? new Date(f.createdAt).getTime() : 0;
        if (f.reinforcedAt) {
          const ra = new Date(f.reinforcedAt).getTime();
          if (ra > lastConfirmed) lastConfirmed = ra;
        }
        if (f.lastRecalledAt) {
          const lr = new Date(f.lastRecalledAt).getTime();
          if (lr > lastConfirmed) lastConfirmed = lr;
        }
        const daysSinceConfirmed = (now - lastConfirmed) / (1000 * 60 * 60 * 24);
        if (daysSinceConfirmed > 30) {
          let staleness = (daysSinceConfirmed - 30) / 90;
          if (staleness > 0.5) staleness = 0.5;
          base *= 1 - staleness;
        }
      }

      return { fact: f, score: base };
    });

    scored.sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score;
      return a.fact.uuid.localeCompare(b.fact.uuid);
    });

    const result = scored.slice(0, limit);
    return result.map((s) => ({
      id: s.fact.uuid,
      content: s.fact.content,
      significance: s.fact.significance,
      tag: s.fact.tag,
    }));
  }

  private ensureInitialized(): void {
    if (!this._initialized || !this.store) {
      throw new Error('MemG not initialized. Call await memg.init() first.');
    }
  }
}

function significanceToNumber(s: string | undefined): number {
  switch (s) {
    case 'high':
      return 10;
    case 'medium':
      return 5;
    case 'low':
      return 1;
    default:
      return 5;
  }
}

function significanceToLabel(n: number): 'low' | 'medium' | 'high' {
  if (n >= 10) return 'high';
  if (n >= 5) return 'medium';
  return 'low';
}

function ttlForSig(sig: number): string | null {
  if (sig >= 10) return null;
  const now = Date.now();
  if (sig >= 5) return new Date(now + 30 * 24 * 60 * 60 * 1000).toISOString();
  return new Date(now + 7 * 24 * 60 * 60 * 1000).toISOString();
}

function extractLastUser(messages: Array<{ role: string; content: string }>): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return messages[i].content;
  }
  return null;
}

async function chatCallLLM(
  apiKey: string,
  model: string,
  messages: Array<{ role: string; content: string }>,
  provider: string,
  maxTokens: number
): Promise<string> {
  let url: string;
  let headers: Record<string, string>;
  let body: string;

  if (provider === 'gemini') {
    const system = messages.find((m) => m.role === 'system')?.content ?? '';
    const contents = messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({ role: m.role === 'assistant' ? 'model' : m.role, parts: [{ text: m.content }] }));

    url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    headers = { 'Content-Type': 'application/json' };
    const payload: any = { contents, generationConfig: { maxOutputTokens: maxTokens } };
    if (system) payload.systemInstruction = { parts: [{ text: system }] };
    body = JSON.stringify(payload);
  } else if (provider === 'anthropic') {
    const system = messages.find((m) => m.role === 'system')?.content ?? '';
    const chatMsgs = messages.filter((m) => m.role !== 'system');

    url = 'https://api.anthropic.com/v1/messages';
    headers = { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' };
    const payload: any = { model, max_tokens: maxTokens, messages: chatMsgs };
    if (system) payload.system = system;
    body = JSON.stringify(payload);
  } else {
    url = 'https://api.openai.com/v1/chat/completions';
    headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` };
    body = JSON.stringify({ model, max_tokens: maxTokens, messages });
  }

  const resp = await fetch(url, { method: 'POST', headers, body });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`LLM API error: HTTP ${resp.status} ${text}`);
  }

  const data = await resp.json() as any;

  if (provider === 'gemini') {
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
  }
  if (provider === 'anthropic') {
    const block = data.content?.find((b: any) => b.type === 'text');
    return block?.text ?? '';
  }
  return data.choices?.[0]?.message?.content ?? '';
}
