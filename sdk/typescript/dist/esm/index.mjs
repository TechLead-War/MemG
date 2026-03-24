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
import { resolveConfig } from './config.mjs';
import { buildContext } from './context.mjs';
import { runExtraction } from './extract.mjs';
import { detectProvider } from './providers/index.mjs';
import { recallFacts, recallSummaries } from './recall.mjs';
import { HybridEngine } from './search.mjs';
import { ensureSession, ensureConversation, saveUserMessage, saveAssistantMessage, loadRecentHistory, } from './session.mjs';
import { MemGStore, defaultContentKey } from './store.mjs';
import { wrap as wrapOpenAIProvider } from './providers/openai.mjs';
import { wrap as wrapAnthropicProvider } from './providers/anthropic.mjs';
import { wrap as wrapGeminiProvider } from './providers/gemini.mjs';
export { MemGClient } from './client.mjs';
export { wrapOpenAIProxy, wrapAnthropicProxy } from './proxy.mjs';
export { wrapOpenAIClient, wrapAnthropicClient, wrapGeminiClient } from './intercept.mjs';
export { detectProvider } from './providers/index.mjs';
export { MemGStore, defaultContentKey } from './store.mjs';
export { PostgresStore } from './postgres_store.mjs';
export { MySQLStore } from './mysql_store.mjs';
export { HybridEngine, cosineSimilarity, dimensionMatch } from './search.mjs';
export { buildContext, estimateTokens } from './context.mjs';
export { runExtraction, isTrivialTurn } from './extract.mjs';
export { recallFacts, recallSummaries } from './recall.mjs';
export { TransformersEmbedder, OpenAIEmbedder, GeminiEmbedder } from './embedder.mjs';
const EMBEDDER_PROBE_TIMEOUT_MS = 15000;
/**
 * Main MemG class providing memory operations and client wrapping.
 * Supports native in-process engine, proxy mode, and MCP client mode.
 */
export class MemG {
    constructor(opts) {
        this.store = null;
        this.embedder = null;
        this._mcpClient = null;
        this._initialized = false;
        this._consciousCache = new Map();
        this._lastPruneAt = 0;
        this.config = resolveConfig(opts);
        this.engine = new HybridEngine();
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
            const { PostgresStore } = await import('./postgres_store.mjs');
            this.store = await PostgresStore.create(this.config.storeUrl);
        }
        else if (this.config.storeProvider === 'mysql') {
            if (!this.config.storeUrl)
                throw new Error('storeUrl required for mysql');
            const { MySQLStore } = await import('./mysql_store.mjs');
            this.store = await MySQLStore.create(this.config.storeUrl);
        }
        else {
            this.store = new MemGStore(this.config.dbPath);
        }
        if (this.config.embedProvider === 'gemini') {
            if (!this.config.geminiApiKey) {
                throw new Error('MemG native mode requires geminiApiKey when embedProvider="gemini".');
            }
            const { GeminiEmbedder } = await import('./embedder.mjs');
            this.embedder = new GeminiEmbedder(this.config.geminiApiKey, this.config.embedModel || 'text-embedding-004', this.config.embedDimension || 768);
        }
        else if (this.config.embedProvider === 'openai') {
            if (!this.config.openaiApiKey) {
                throw new Error('MemG native mode requires openaiApiKey when embedProvider="openai".');
            }
            const { OpenAIEmbedder } = await import('./embedder.mjs');
            this.embedder = new OpenAIEmbedder(this.config.openaiApiKey, this.config.embedModel || 'text-embedding-3-small', this.config.embedDimension || 1536);
        }
        else if (this.config.embedProvider === 'sentence-transformers') {
            const { TransformersEmbedder } = await import('./embedder.mjs');
            this.embedder = await TransformersEmbedder.create(this.config.embedModel || 'Xenova/all-MiniLM-L6-v2');
        }
        else {
            throw new Error(`Unsupported embedProvider "${this.config.embedProvider}".`);
        }
        if (!this.embedder) {
            throw new Error('MemG native mode requires a working embedder.');
        }
        await this.probeEmbedder(this.embedder);
        this.config.embedDimension = this.embedder.dimension();
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
        const provider = detectProvider(client);
        const mergedOpts = {
            mode: 'native',
            proxyUrl: 'http://localhost:8787/v1',
            mcpUrl: 'http://localhost:8686',
            extract: true,
            ...opts,
        };
        if (provider === 'openai') {
            return wrapOpenAIProvider(client, mergedOpts);
        }
        if (provider === 'anthropic') {
            return wrapAnthropicProvider(client, mergedOpts);
        }
        if (provider === 'gemini') {
            return wrapGeminiProvider(client, mergedOpts);
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
            const contentKey = defaultContentKey(mem.content);
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
        const embedder = this.requireEmbedder();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return { memories: [], count: 0 };
        const recalled = await recallFacts(this.store, this.engine, embedder, query, entity.uuid, limit ?? this.config.recallLimit, this.config.recallThreshold, this.config.maxRecallCandidates);
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
        const entityUuid = await this.store.upsertEntity(entityId);
        let sessionUuid = '';
        let conversationUuid = '';
        if (this.config.sessionTimeout > 0) {
            try {
                const { session, isNew } = await ensureSession(this.store, entityUuid, 'default', this.config.sessionTimeout);
                sessionUuid = session.uuid;
                conversationUuid = await ensureConversation(this.store, sessionUuid, entityUuid);
                if (isNew) {
                    this.summarizeClosedConversation(entityUuid, sessionUuid).catch(() => { });
                }
            }
            catch {
            }
        }
        let retrievalMessages = [...messages];
        if (sessionUuid) {
            try {
                const history = await loadRecentHistory(this.store, sessionUuid, this.config.workingMemoryTurns);
                if (history.length > 0) {
                    const historyMsgs = history.map((h) => ({ role: h.role, content: h.content }));
                    retrievalMessages = [...historyMsgs, ...messages];
                }
            }
            catch {
            }
        }
        const queryText = extractLastUser(retrievalMessages) ?? '';
        const memoryContext = await this.buildMemoryContext(entityId, queryText);
        let augmented = [...messages];
        if (sessionUuid && retrievalMessages.length > messages.length) {
            const historyOnly = retrievalMessages.slice(0, retrievalMessages.length - messages.length);
            augmented = [...historyOnly, ...messages];
        }
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
        if (conversationUuid) {
            const userText = extractLastUser(messages) ?? '';
            Promise.resolve().then(async () => {
                try {
                    if (userText)
                        await saveUserMessage(this.store, conversationUuid, userText);
                    if (content)
                        await saveAssistantMessage(this.store, conversationUuid, content);
                }
                catch { /* never block on save failure */ }
            });
        }
        const extractMessages = retrievalMessages
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
        const embedder = this.requireEmbedder();
        const entity = await this.store.lookupEntity(entityId);
        if (!entity)
            return '';
        this.backfillMissingEmbeddings(entity.uuid).catch(() => { });
        this.pruneIfDue(entity.uuid);
        let consciousFacts = [];
        let recalledFactsList = [];
        let summaries = [];
        if (this.config.consciousMode) {
            consciousFacts = await this.loadConsciousFactsCached(entity.uuid, this.config.consciousLimit);
        }
        if (queryText) {
            const [recalled, recalledSums] = await Promise.all([
                recallFacts(this.store, this.engine, embedder, queryText, entity.uuid, this.config.recallLimit, this.config.recallThreshold, this.config.maxRecallCandidates),
                recallSummaries(this.store, this.engine, embedder, queryText, entity.uuid, 10, this.config.recallThreshold),
            ]);
            recalledFactsList = recalled;
            summaries = recalledSums;
            // Track recall usage.
            if (recalled.length > 0) {
                await this.store.updateRecallUsage(recalled.map((r) => r.id));
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
    async extractFromMessages(entityId, messages) {
        if (!this.config.extract)
            return;
        // Resolve the API key for the configured LLM provider.
        const apiKey = this.resolveExtractionApiKey();
        if (!apiKey)
            return;
        try {
            const entityUuid = await this.store.upsertEntity(entityId);
            await runExtraction(this.store, this.embedder, messages, entityUuid, apiKey, this.config.llmModel, this.config.llmProvider);
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
    /**
     * Backfill embeddings for facts that were stored without them (e.g., embedder was down).
     * Runs in the background — does not block recall.
     */
    async backfillMissingEmbeddings(entityUuid) {
        const unembedded = await this.store.listUnembeddedFacts(entityUuid, 50);
        if (unembedded.length === 0)
            return;
        const contents = unembedded.map((f) => f.content);
        let embeddings;
        try {
            embeddings = await this.embedder.embed(contents);
        }
        catch {
            return;
        }
        const model = this.embedder.modelName();
        for (let i = 0; i < embeddings.length && i < unembedded.length; i++) {
            try {
                await this.store.updateFactEmbedding(unembedded[i].uuid, embeddings[i], model);
            }
            catch {
            }
        }
    }
    /**
     * Summarize the most recent unsummarized conversation when a new session starts.
     * Matches Go's summarizeClosedSession behavior.
     */
    async summarizeClosedConversation(entityUuid, currentSessionUuid) {
        const apiKey = this.resolveExtractionApiKey();
        if (!apiKey)
            return;
        const conv = await this.store.findUnsummarizedConversation(entityUuid, currentSessionUuid);
        if (!conv || conv.summary)
            return;
        await this.generateAndStoreSummary(conv.uuid, apiKey);
    }
    /**
     * Generate a summary for a conversation and store it with its embedding.
     */
    async generateAndStoreSummary(conversationUuid, apiKey) {
        const messages = await this.store.readMessages(conversationUuid);
        if (messages.length === 0)
            return;
        const transcript = messages.map((m) => `${m.role}: ${m.content}`).join('\n');
        const summaryPrompt = `Summarize this conversation. Focus on:
- What was discussed
- What decisions were made
- What is still pending or unresolved
- Any new information learned about the user

Be concise — 2-5 sentences. Only include what is meaningful and worth remembering.
If the conversation contains no meaningful content worth remembering (e.g. just greetings or trivial exchanges), respond with exactly: NONE`;
        const { callLLM } = await import('./extract.mjs');
        let summary;
        try {
            summary = await callLLM(apiKey, this.config.llmModel, summaryPrompt, transcript, this.config.llmProvider);
        }
        catch {
            return;
        }
        summary = summary.trim();
        if (!summary || summary.toUpperCase() === 'NONE')
            return;
        let embedding = [];
        if (this.embedder) {
            try {
                const [vec] = await this.embedder.embed([summary]);
                embedding = vec;
            }
            catch {
            }
        }
        await this.store.updateConversationSummary(conversationUuid, summary, embedding);
    }
    async loadConsciousFactsCached(entityUuid, limit) {
        const now = Date.now();
        const cached = this._consciousCache.get(entityUuid);
        if (cached && cached.expiresAt > now)
            return cached.facts;
        const facts = await this.loadConsciousFacts(entityUuid, limit);
        this._consciousCache.set(entityUuid, { facts, expiresAt: now + 30000 });
        return facts;
    }
    pruneIfDue(entityUuid) {
        const now = Date.now();
        if (now - this._lastPruneAt < 300000)
            return; // 5 min interval
        this._lastPruneAt = now;
        const nowISO = new Date().toISOString();
        Promise.resolve(this.store.pruneExpiredFacts(entityUuid, nowISO)).catch(() => { });
    }
    ensureInitialized() {
        if (!this._initialized || !this.store) {
            throw new Error('MemG not initialized. Call await memg.init() first.');
        }
    }
    requireEmbedder() {
        if (!this.embedder) {
            throw new Error('MemG native mode requires a healthy embedder for memory recall.');
        }
        return this.embedder;
    }
    async probeEmbedder(embedder) {
        const timeout = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('embedder health check timed out')), EMBEDDER_PROBE_TIMEOUT_MS);
        });
        const vectors = await Promise.race([
            embedder.embed(['memg-healthcheck']),
            timeout,
        ]);
        if (!Array.isArray(vectors) || vectors.length !== 1) {
            throw new Error(`embedder health check returned ${Array.isArray(vectors) ? vectors.length : 0} vectors`);
        }
        const vector = vectors[0];
        if (!Array.isArray(vector) || vector.length === 0) {
            throw new Error('embedder health check returned an empty vector');
        }
        const expectedDim = embedder.dimension();
        if (expectedDim > 0 && vector.length !== expectedDim) {
            throw new Error(`embedder health check dimension mismatch: got ${vector.length}, want ${expectedDim}`);
        }
        for (let i = 0; i < vector.length; i++) {
            if (!Number.isFinite(vector[i])) {
                throw new Error(`embedder health check returned invalid value at index ${i}`);
            }
        }
    }
}
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
