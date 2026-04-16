/**
 * Default configuration for the native MemG engine.
 * Sensible defaults for all options.
 */

import type { NativeConfig } from './types.js';

export const DEFAULT_CONFIG: Required<NativeConfig> = {
  store: undefined as any,
  storeProvider: 'sqlite',
  storeUrl: '',
  dbPath: 'memg.db',
  embedProvider: 'sentence-transformers',
  embedModel: 'Xenova/all-MiniLM-L6-v2',
  embedDimension: 384,
  llmProvider: 'openai',
  llmModel: 'gpt-4o-mini',
  recallLimit: 200,
  recallThreshold: 0.05,
  maxRecallCandidates: 0, // 0 = load all facts; let the search engine rank them.
  sessionTimeout: 30 * 60 * 1000, // 30 minutes in ms
  workingMemoryTurns: 10,
  memoryTokenBudget: 4000,
  summaryTokenBudget: 1000,
  consciousMode: true,
  consciousLimit: 10,
  extract: true,
  openaiApiKey: '',
  geminiApiKey: '',
  deepseekApiKey: '',
  groqApiKey: '',
  togetherApiKey: '',
  xaiApiKey: '',
  azureOpenaiApiKey: '',
  azureOpenaiEndpoint: '',
  azureOpenaiApiVersion: '2024-10-21',
  bedrockRegion: 'us-east-1',
  ollamaBaseUrl: 'http://localhost:11434',
  cohereApiKey: '',
  voyageApiKey: '',
  maxPersonalFacts: 15,
  diversifyTopics: true,
  freshnessBias: 0.3,
};

/**
 * Merge user config with defaults.
 */
export function resolveConfig(user?: NativeConfig): Required<NativeConfig> {
  return { ...DEFAULT_CONFIG, ...user };
}
