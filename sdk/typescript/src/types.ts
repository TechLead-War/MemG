/** A stored memory (fact) returned from the MemG server. */
export interface Memory {
  id: string;
  content: string;
  type: 'identity' | 'event' | 'pattern';
  temporalStatus: 'current' | 'historical';
  significance: 'low' | 'medium' | 'high';
  createdAt?: string;
  tag?: string;
  score?: number;
  reinforcedCount?: number;
}

/** Input payload for storing a new memory. */
export interface MemoryInput {
  content: string;
  type?: 'identity' | 'event' | 'pattern';
  significance?: 'low' | 'medium' | 'high';
  tag?: string;
}

/** Result of an add_memories call. */
export interface AddResult {
  inserted: number;
  reinforced: number;
}

/** Result of a search or list call. */
export interface SearchResult {
  memories: Memory[];
  count: number;
}

/** Options for wrapping an LLM client with MemG. */
export interface WrapOptions {
  /** External entity identifier (e.g. user ID). */
  entity?: string;
  /** Wrapping mode: proxy redirects traffic, client intercepts locally, native runs fully in-process. */
  mode?: 'proxy' | 'client' | 'native';
  /** MemG proxy URL (default: http://localhost:8787/v1). */
  proxyUrl?: string;
  /** MemG MCP server URL (default: http://localhost:8686). */
  mcpUrl?: string;
  /** Whether to extract and store knowledge from responses (default: true). */
  extract?: boolean;
  /** Native mode configuration. */
  nativeConfig?: NativeConfig;
}

/** Options for creating a MemG instance. */
export interface MemGOptions {
  /** MCP server URL (default: http://localhost:8686). */
  mcpUrl?: string;
  /** Proxy URL (default: http://localhost:8787/v1). */
  proxyUrl?: string;
}

/** A stored fact with full metadata (native engine internal type). */
export interface Fact {
  uuid: string;
  content: string;
  embedding?: number[];
  createdAt?: string;
  updatedAt?: string;
  factType: 'identity' | 'event' | 'pattern';
  temporalStatus: 'current' | 'historical';
  significance: number;
  contentKey: string;
  referenceTime?: string;
  expiresAt?: string;
  reinforcedAt?: string;
  reinforcedCount: number;
  tag: string;
  slot: string;
  confidence: number;
  embeddingModel: string;
  sourceRole: string;
  recallCount: number;
  lastRecalledAt?: string;
}

/** An entity record. */
export interface FactEntity {
  uuid: string;
  externalId: string;
  createdAt?: string;
}

/** A session record. */
export interface FactSession {
  uuid: string;
  entityId: string;
  processId: string;
  createdAt?: string;
  expiresAt?: string;
}

/** A conversation record. */
export interface FactConversation {
  uuid: string;
  sessionId: string;
  entityId: string;
  summary: string;
  summaryEmbedding?: number[];
  createdAt?: string;
  updatedAt?: string;
}

/** A message in a conversation. */
export interface FactMessage {
  uuid: string;
  conversationId: string;
  role: string;
  content: string;
  kind: string;
  createdAt?: string;
}

/** Filters for querying facts. */
export interface FactFilter {
  types?: string[];
  statuses?: string[];
  tags?: string[];
  minSignificance?: number;
  excludeExpired?: boolean;
  slots?: string[];
  minConfidence?: number;
  sourceRoles?: string[];
}

/** A recalled fact with its search score. */
export interface RecalledFact {
  id: string;
  content: string;
  score: number;
  temporalStatus: string;
  significance: number;
  createdAt?: string;
}

/** A high-significance fact loaded for conscious mode. */
export interface ConsciousFact {
  id: string;
  content: string;
  significance: number;
  tag: string;
}

/** A recalled conversation summary. */
export interface RecalledSummary {
  conversationId: string;
  summary: string;
  score: number;
  createdAt?: string;
}

/** Configuration for the native in-process engine. */
export interface NativeConfig {
  /** Custom store implementation. When provided, storeProvider and dbPath are ignored. */
  store?: import('./store').Store;
  /** Store provider: 'sqlite' (default), 'postgres', or 'mysql'. */
  storeProvider?: 'sqlite' | 'postgres' | 'mysql';
  /** Connection URL for postgres/mysql (e.g. 'postgresql://user:pass@host/db'). */
  storeUrl?: string;
  dbPath?: string;
  embedProvider?: 'sentence-transformers' | 'openai' | 'gemini';
  embedModel?: string;
  embedDimension?: number;
  llmProvider?: string;
  llmModel?: string;
  recallLimit?: number;
  recallThreshold?: number;
  maxRecallCandidates?: number;
  sessionTimeout?: number;
  workingMemoryTurns?: number;
  memoryTokenBudget?: number;
  summaryTokenBudget?: number;
  consciousMode?: boolean;
  consciousLimit?: number;
  extract?: boolean;
  openaiApiKey?: string;
  geminiApiKey?: string;
}
