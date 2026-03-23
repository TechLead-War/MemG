"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.MemG = exports.GeminiEmbedder = exports.OpenAIEmbedder = exports.TransformersEmbedder = exports.recallSummaries = exports.recallFacts = exports.isTrivialTurn = exports.runExtraction = exports.estimateTokens = exports.buildContext = exports.dimensionMatch = exports.cosineSimilarity = exports.HybridEngine = exports.MySQLStore = exports.PostgresStore = exports.defaultContentKey = exports.MemGStore = exports.detectProvider = exports.wrapGeminiClient = exports.wrapAnthropicClient = exports.wrapOpenAIClient = exports.wrapAnthropicProxy = exports.wrapOpenAIProxy = exports.MemGClient = void 0;
const config_1 = require("./config");
const context_1 = require("./context");
const extract_1 = require("./extract");
const providers_1 = require("./providers");
const recall_1 = require("./recall");
const search_1 = require("./search");
const store_1 = require("./store");
var client_1 = require("./client");
Object.defineProperty(exports, "MemGClient", { enumerable: true, get: function () { return client_1.MemGClient; } });
var proxy_1 = require("./proxy");
Object.defineProperty(exports, "wrapOpenAIProxy", { enumerable: true, get: function () { return proxy_1.wrapOpenAIProxy; } });
Object.defineProperty(exports, "wrapAnthropicProxy", { enumerable: true, get: function () { return proxy_1.wrapAnthropicProxy; } });
var intercept_1 = require("./intercept");
Object.defineProperty(exports, "wrapOpenAIClient", { enumerable: true, get: function () { return intercept_1.wrapOpenAIClient; } });
Object.defineProperty(exports, "wrapAnthropicClient", { enumerable: true, get: function () { return intercept_1.wrapAnthropicClient; } });
Object.defineProperty(exports, "wrapGeminiClient", { enumerable: true, get: function () { return intercept_1.wrapGeminiClient; } });
var providers_2 = require("./providers");
Object.defineProperty(exports, "detectProvider", { enumerable: true, get: function () { return providers_2.detectProvider; } });
var store_2 = require("./store");
Object.defineProperty(exports, "MemGStore", { enumerable: true, get: function () { return store_2.MemGStore; } });
Object.defineProperty(exports, "defaultContentKey", { enumerable: true, get: function () { return store_2.defaultContentKey; } });
var postgres_store_1 = require("./postgres_store");
Object.defineProperty(exports, "PostgresStore", { enumerable: true, get: function () { return postgres_store_1.PostgresStore; } });
var mysql_store_1 = require("./mysql_store");
Object.defineProperty(exports, "MySQLStore", { enumerable: true, get: function () { return mysql_store_1.MySQLStore; } });
var search_2 = require("./search");
Object.defineProperty(exports, "HybridEngine", { enumerable: true, get: function () { return search_2.HybridEngine; } });
Object.defineProperty(exports, "cosineSimilarity", { enumerable: true, get: function () { return search_2.cosineSimilarity; } });
Object.defineProperty(exports, "dimensionMatch", { enumerable: true, get: function () { return search_2.dimensionMatch; } });
var context_2 = require("./context");
Object.defineProperty(exports, "buildContext", { enumerable: true, get: function () { return context_2.buildContext; } });
Object.defineProperty(exports, "estimateTokens", { enumerable: true, get: function () { return context_2.estimateTokens; } });
var extract_2 = require("./extract");
Object.defineProperty(exports, "runExtraction", { enumerable: true, get: function () { return extract_2.runExtraction; } });
Object.defineProperty(exports, "isTrivialTurn", { enumerable: true, get: function () { return extract_2.isTrivialTurn; } });
var recall_2 = require("./recall");
Object.defineProperty(exports, "recallFacts", { enumerable: true, get: function () { return recall_2.recallFacts; } });
Object.defineProperty(exports, "recallSummaries", { enumerable: true, get: function () { return recall_2.recallSummaries; } });
var embedder_1 = require("./embedder");
Object.defineProperty(exports, "TransformersEmbedder", { enumerable: true, get: function () { return embedder_1.TransformersEmbedder; } });
Object.defineProperty(exports, "OpenAIEmbedder", { enumerable: true, get: function () { return embedder_1.OpenAIEmbedder; } });
Object.defineProperty(exports, "GeminiEmbedder", { enumerable: true, get: function () { return embedder_1.GeminiEmbedder; } });
/**
 * Main MemG class providing memory operations and client wrapping.
 * Supports native in-process engine, proxy mode, and MCP client mode.
 */
class MemG {
    constructor(opts) {
        this.store = null;
        this.embedder = null;
        this._mcpClient = null;
        this._initialized = false;
        this.config = (0, config_1.resolveConfig)(opts);
        this.engine = new search_1.HybridEngine();
        this._proxyUrl = opts?.proxyUrl ?? 'http://localhost:8787/v1';
    }
    /**
     * Initialize the native engine. Must be called before using native operations.
     * Creates the SQLite database and sets up the embedder.
     */
    async init() {
        if (this._initialized)
            return;
        if (this.config.store) {
            this.store = this.config.store;
        }
        else if (this.config.storeProvider === 'postgres') {
            if (!this.config.storeUrl)
                throw new Error('storeUrl required for postgres');
            const { PostgresStore } = require('./postgres_store');
            this.store = await PostgresStore.create(this.config.storeUrl);
        }
        else if (this.config.storeProvider === 'mysql') {
            if (!this.config.storeUrl)
                throw new Error('storeUrl required for mysql');
            const { MySQLStore } = require('./mysql_store');
            this.store = await MySQLStore.create(this.config.storeUrl);
        }
        else {
            this.store = new store_1.MemGStore(this.config.dbPath);
        }
        if (this.config.embedProvider === 'gemini' && this.config.geminiApiKey) {
            const { GeminiEmbedder } = require('./embedder');
            this.embedder = new GeminiEmbedder(this.config.geminiApiKey, this.config.embedModel || 'text-embedding-004', this.config.embedDimension || 768);
        }
        else if (this.config.embedProvider === 'openai' && this.config.openaiApiKey) {
            const { OpenAIEmbedder } = require('./embedder');
            this.embedder = new OpenAIEmbedder(this.config.openaiApiKey, this.config.embedModel || 'text-embedding-3-small', this.config.embedDimension || 1536);
        }
        else if (this.config.embedProvider === 'sentence-transformers') {
            try {
                const { TransformersEmbedder } = require('./embedder');
                this.embedder = await TransformersEmbedder.create(this.config.embedModel || 'Xenova/all-MiniLM-L6-v2');
            }
            catch (err) {
                console.warn('[memg] @huggingface/transformers not available, recall will use lexical-only search. Install it: npm i @huggingface/transformers');
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
    static wrap(client, opts) {
        const provider = (0, providers_1.detectProvider)(client);
        const mergedOpts = {
            mode: 'native',
            proxyUrl: 'http://localhost:8787/v1',
            mcpUrl: 'http://localhost:8686',
            extract: true,
            ...opts,
        };
        if (provider === 'openai') {
            const { wrap } = require('./providers/openai');
            return wrap(client, mergedOpts);
        }
        if (provider === 'anthropic') {
            const { wrap } = require('./providers/anthropic');
            return wrap(client, mergedOpts);
        }
        if (provider === 'gemini') {
            const { wrap } = require('./providers/gemini');
            return wrap(client, mergedOpts);
        }
        throw new Error(`Unsupported client type: ${client?.constructor?.name}. Supported: OpenAI, Anthropic, Gemini`);
    }
    /**
     * Store memories for an entity.
     */
    async add(entityId, contentOrMemories, opts) {
        this.ensureInitialized();
        let memories;
        if (typeof contentOrMemories === 'string') {
            memories = [{
                    content: contentOrMemories,
                    type: opts?.type,
                    significance: opts?.significance,
                    tag: opts?.tag,
                }];
        }
        else if (Array.isArray(contentOrMemories)) {
            if (contentOrMemories.length === 0)
                return { inserted: 0, reinforced: 0 };
            if (typeof contentOrMemories[0] === 'string') {
                memories = contentOrMemories.map((content) => ({
                    content,
                    type: opts?.type,
                    significance: opts?.significance,
                    tag: opts?.tag,
                }));
            }
            else {
                memories = contentOrMemories;
            }
        }
        else {
            throw new Error('contentOrMemories must be a string, string[], or MemoryInput[]');
        }
        const entityUuid = await this.store.upsertEntity(entityId);
        let inserted = 0;
        let reinforced = 0;
        for (const mem of memories) {
            const contentKey = (0, store_1.defaultContentKey)(mem.content);
            const existing = await this.store.findFactByKey(entityUuid, contentKey);
            if (existing) {
                await this.store.reinforceFact(existing.uuid, existing.expiresAt ?? null);
                reinforced++;
                continue;
            }
            const sig = significanceToNumber(mem.significance);
            const fact = {
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
                }
                catch {
                    // Skip embedding on failure.
                }
            }
            await this.store.insertFact(entityUuid, fact);
            inserted++;
        }
        return { inserted, reinforced };
    }
    /**
     * Search memories for an entity using semantic hybrid search.
     */
    async search(entityId, query, limit) {
        this.ensureInitialized();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return { memories: [], count: 0 };
        if (!this.embedder) {
            console.warn('[memg] no embedder available, search requires embeddings');
            return { memories: [], count: 0 };
        }
        const recalled = await (0, recall_1.recallFacts)(this.store, this.engine, this.embedder, query, entity.uuid, limit ?? this.config.recallLimit, this.config.recallThreshold, this.config.maxRecallCandidates);
        const memories = recalled.map((r) => ({
            id: r.id,
            content: r.content,
            type: 'identity',
            temporalStatus: r.temporalStatus,
            significance: significanceToLabel(r.significance),
            createdAt: r.createdAt,
            score: r.score,
        }));
        // Update recall usage.
        if (recalled.length > 0) {
            await this.store.updateRecallUsage(recalled.map((r) => r.id));
        }
        return { memories, count: memories.length };
    }
    /**
     * List stored memories for an entity.
     */
    async list(entityId, opts) {
        this.ensureInitialized();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return { memories: [], count: 0 };
        const filter = {};
        if (opts?.type)
            filter.types = [opts.type];
        if (opts?.tag)
            filter.tags = [opts.tag];
        const facts = await this.store.listFactsFiltered(entity.uuid, filter, opts?.limit ?? 100);
        const memories = facts.map((f) => ({
            id: f.uuid,
            content: f.content,
            type: f.factType,
            temporalStatus: f.temporalStatus,
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
    async delete(entityId, memoryId) {
        this.ensureInitialized();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return false;
        await this.store.deleteFact(entity.uuid, memoryId);
        return true;
    }
    /**
     * Delete all memories for an entity.
     */
    async deleteAll(entityId) {
        this.ensureInitialized();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return 0;
        return await this.store.deleteEntityFacts(entity.uuid);
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
    async chat(messages, entityId, opts) {
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
            }
            else {
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
        this.extractFromMessages(entityId, extractMessages).catch(() => { });
        return { content, role: 'assistant', memoryContext };
    }
    resolveChatApiKey(provider) {
        if (provider === 'gemini')
            return this.config.geminiApiKey;
        if (provider === 'anthropic')
            return process.env.ANTHROPIC_API_KEY ?? '';
        return this.config.openaiApiKey;
    }
    /**
     * Get the underlying store for advanced operations.
     */
    getStore() {
        return this.store;
    }
    /**
     * Get the underlying embedder.
     */
    getEmbedder() {
        return this.embedder;
    }
    /**
     * Get the search engine.
     */
    getEngine() {
        return this.engine;
    }
    /**
     * Get the resolved config.
     */
    getConfig() {
        return this.config;
    }
    /**
     * Build memory context for injection into LLM prompts.
     */
    async buildMemoryContext(entityId, queryText) {
        this.ensureInitialized();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return '';
        let consciousFacts = [];
        let recalledFactsList = [];
        let summaries = [];
        // Load conscious facts if enabled.
        if (this.config.consciousMode) {
            consciousFacts = await this.loadConsciousFacts(entity.uuid, this.config.consciousLimit);
        }
        // Recall facts and summaries if embedder is available.
        if (this.embedder && queryText) {
            try {
                const [recalled, recalledSums] = await Promise.all([
                    (0, recall_1.recallFacts)(this.store, this.engine, this.embedder, queryText, entity.uuid, this.config.recallLimit, this.config.recallThreshold, this.config.maxRecallCandidates),
                    (0, recall_1.recallSummaries)(this.store, this.engine, this.embedder, queryText, entity.uuid, 10, this.config.recallThreshold),
                ]);
                recalledFactsList = recalled;
                summaries = recalledSums;
                // Track recall usage.
                if (recalled.length > 0) {
                    await this.store.updateRecallUsage(recalled.map((r) => r.id));
                }
            }
            catch (err) {
                console.warn('[memg] recall failed:', err);
            }
        }
        return (0, context_1.buildContext)({
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
    async extractFromMessages(entityId, messages) {
        if (!this.config.extract)
            return;
        // Resolve the API key for the configured LLM provider.
        const apiKey = this.resolveExtractionApiKey();
        if (!apiKey)
            return;
        try {
            const entityUuid = await this.store.upsertEntity(entityId);
            await (0, extract_1.runExtraction)(this.store, this.embedder, messages, entityUuid, apiKey, this.config.llmModel, this.config.llmProvider);
        }
        catch (err) {
            console.warn('[memg] extraction failed:', err);
        }
    }
    /**
     * Resolve the API key for the extraction LLM provider.
     */
    resolveExtractionApiKey() {
        const provider = this.config.llmProvider;
        if (provider === 'gemini')
            return this.config.geminiApiKey;
        if (provider === 'anthropic')
            return process.env.ANTHROPIC_API_KEY ?? '';
        return this.config.openaiApiKey; // openai and openai-compatible
    }
    /**
     * Close the database connection.
     */
    close() {
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
    async loadConsciousFacts(entityUuid, limit) {
        if (limit <= 0)
            limit = 10;
        const filter = {
            statuses: ['current'],
            excludeExpired: true,
        };
        let fetchLimit = limit * 5;
        if (fetchLimit < 50)
            fetchLimit = 50;
        const facts = await this.store.listFactsFiltered(entityUuid, filter, fetchLimit);
        if (facts.length === 0)
            return [];
        // Score and sort by significance (with staleness demotion for mutable facts).
        const now = Date.now();
        const scored = facts.map((f) => {
            let base = f.significance;
            if (f.factType === 'identity' && f.significance < 10) {
                let lastConfirmed = f.createdAt ? new Date(f.createdAt).getTime() : 0;
                if (f.reinforcedAt) {
                    const ra = new Date(f.reinforcedAt).getTime();
                    if (ra > lastConfirmed)
                        lastConfirmed = ra;
                }
                if (f.lastRecalledAt) {
                    const lr = new Date(f.lastRecalledAt).getTime();
                    if (lr > lastConfirmed)
                        lastConfirmed = lr;
                }
                const daysSinceConfirmed = (now - lastConfirmed) / (1000 * 60 * 60 * 24);
                if (daysSinceConfirmed > 30) {
                    let staleness = (daysSinceConfirmed - 30) / 90;
                    if (staleness > 0.5)
                        staleness = 0.5;
                    base *= 1 - staleness;
                }
            }
            return { fact: f, score: base };
        });
        scored.sort((a, b) => {
            if (a.score !== b.score)
                return b.score - a.score;
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
    ensureInitialized() {
        if (!this._initialized || !this.store) {
            throw new Error('MemG not initialized. Call await memg.init() first.');
        }
    }
}
exports.MemG = MemG;
function significanceToNumber(s) {
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
function significanceToLabel(n) {
    if (n >= 10)
        return 'high';
    if (n >= 5)
        return 'medium';
    return 'low';
}
function ttlForSig(sig) {
    if (sig >= 10)
        return null;
    const now = Date.now();
    if (sig >= 5)
        return new Date(now + 30 * 24 * 60 * 60 * 1000).toISOString();
    return new Date(now + 7 * 24 * 60 * 60 * 1000).toISOString();
}
function extractLastUser(messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user')
            return messages[i].content;
    }
    return null;
}
async function chatCallLLM(apiKey, model, messages, provider, maxTokens) {
    let url;
    let headers;
    let body;
    if (provider === 'gemini') {
        const system = messages.find((m) => m.role === 'system')?.content ?? '';
        const contents = messages
            .filter((m) => m.role !== 'system')
            .map((m) => ({ role: m.role === 'assistant' ? 'model' : m.role, parts: [{ text: m.content }] }));
        url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
        headers = { 'Content-Type': 'application/json' };
        const payload = { contents, generationConfig: { maxOutputTokens: maxTokens } };
        if (system)
            payload.systemInstruction = { parts: [{ text: system }] };
        body = JSON.stringify(payload);
    }
    else if (provider === 'anthropic') {
        const system = messages.find((m) => m.role === 'system')?.content ?? '';
        const chatMsgs = messages.filter((m) => m.role !== 'system');
        url = 'https://api.anthropic.com/v1/messages';
        headers = { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' };
        const payload = { model, max_tokens: maxTokens, messages: chatMsgs };
        if (system)
            payload.system = system;
        body = JSON.stringify(payload);
    }
    else {
        url = 'https://api.openai.com/v1/chat/completions';
        headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` };
        body = JSON.stringify({ model, max_tokens: maxTokens, messages });
    }
    const resp = await fetch(url, { method: 'POST', headers, body });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`LLM API error: HTTP ${resp.status} ${text}`);
    }
    const data = await resp.json();
    if (provider === 'gemini') {
        return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
    }
    if (provider === 'anthropic') {
        const block = data.content?.find((b) => b.type === 'text');
        return block?.text ?? '';
    }
    return data.choices?.[0]?.message?.content ?? '';
}
//# sourceMappingURL=index.js.map