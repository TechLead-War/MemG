/**
 * Default configuration for the native MemG engine.
 * All values match the Go DefaultConfig().
 */
export const DEFAULT_CONFIG = {
    store: undefined,
    storeProvider: 'sqlite',
    storeUrl: '',
    dbPath: 'memg.db',
    embedProvider: 'sentence-transformers',
    embedModel: 'Xenova/all-MiniLM-L6-v2',
    embedDimension: 384,
    llmProvider: 'openai',
    llmModel: 'gpt-4o-mini',
    recallLimit: 100,
    recallThreshold: 0.10,
    maxRecallCandidates: 10000,
    sessionTimeout: 30 * 60 * 1000, // 30 minutes in ms
    workingMemoryTurns: 20,
    memoryTokenBudget: 4000,
    summaryTokenBudget: 1000,
    consciousMode: true,
    consciousLimit: 10,
    extract: true,
    openaiApiKey: '',
    geminiApiKey: '',
};
/**
 * Merge user config with defaults.
 */
export function resolveConfig(user) {
    return { ...DEFAULT_CONFIG, ...user };
}
