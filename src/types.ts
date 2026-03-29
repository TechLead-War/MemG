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
  /** Emotional valence of this memory. */
  emotionalValence?: string;
  /** User's exact words (if available). */
  verbatim?: string;
  /** Thread status for ongoing situations. */
  threadStatus?: 'open' | 'resolved' | null;
  /** Whether user pinned this memory. */
  pinned?: boolean;
}

/** Input payload for storing a new memory. */
export interface MemoryInput {
  content: string;
  type?: 'identity' | 'event' | 'pattern';
  significance?: 'low' | 'medium' | 'high';
  tag?: string;
  /** User's exact words for self-reference mirroring. */
  verbatim?: string;
  /** Emotional valence category. */
  emotionalValence?: string;
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
  /** Emotional weight of this fact (0.0-1.0). Higher = more emotionally significant. */
  emotionalWeight?: number;
  /** Emotional valence category. */
  emotionalValence?: 'grief' | 'joy' | 'anxiety' | 'hope' | 'love' | 'anger' | 'fear' | 'pride' | 'neutral';
  /** User's exact words when this fact was extracted. */
  verbatim?: string;
  /** When the state/situation described by this fact began (ISO date). */
  startedAt?: string;
  /** Thread status for ongoing situations. */
  threadStatus?: 'open' | 'resolved' | null;
  /** Engagement score — how much the user engaged with this topic (0.0-1.0). */
  engagementScore?: number;
  /** Whether this fact is pinned by the user (never decays). */
  pinned?: boolean;
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
  entityMentions: string[];
  messageCount: number;
}

/** A conversation record. */
export interface FactConversation {
  uuid: string;
  sessionId: string;
  entityId: string;
  summary: string;
  summaryEmbedding?: number[];
  summaryEmbeddingModel?: string;
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

/** A turn summary record. */
export interface TurnSummary {
  uuid: string;
  conversationId: string;
  entityId: string;
  startTurn: number;
  endTurn: number;
  summary: string;
  summaryEmbedding?: number[];
  isOverview: boolean;
  createdAt: string;
}

/** An artifact record. */
export interface Artifact {
  uuid: string;
  conversationId: string;
  entityId: string;
  content: string;
  artifactType: string;
  language: string;
  description: string;
  descriptionEmbedding?: number[];
  supersededBy?: string;
  turnNumber: number;
  createdAt: string;
}

/** Filters for querying facts. */
export interface FactFilter {
  types?: string[];
  statuses?: string[];
  tags?: string[];
  minSignificance?: number;
  maxSignificance?: number;
  excludeExpired?: boolean;
  referenceTimeAfter?: string;
  referenceTimeBefore?: string;
  slots?: string[];
  minConfidence?: number;
  sourceRoles?: string[];
  unembeddedOnly?: boolean;
  /** Filter by thread status. */
  threadStatuses?: string[];
  /** Filter by pinned status. */
  pinned?: boolean;
  /** Filter by emotional valence. */
  emotionalValences?: string[];
}

/** A recalled fact with its search score. */
export interface RecalledFact {
  id: string;
  content: string;
  score: number;
  temporalStatus: string;
  significance: number;
  createdAt?: string;
  /** Confidence of extraction (0.0-1.0). */
  confidence?: number;
  /** Emotional valence. */
  emotionalValence?: string;
  /** Emotional weight (0.0-1.0). */
  emotionalWeight?: number;
  /** User's exact words. */
  verbatim?: string;
  /** Thread status. */
  threadStatus?: string;
  /** When the state began. */
  startedAt?: string;
  /** Fact type. */
  factType?: string;
  /** Semantic tag. */
  tag?: string;
}

/** A high-significance fact loaded for conscious mode. */
export interface ConsciousFact {
  id: string;
  content: string;
  significance: number;
  tag: string;
  /** Confidence of extraction. */
  confidence?: number;
  /** Emotional valence. */
  emotionalValence?: string;
  /** User's exact words. */
  verbatim?: string;
  /** Fact type. */
  factType?: string;
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
  /** Maximum personal facts per context build (personalization throttle). Default: 15. */
  maxPersonalFacts?: number;
  /** Diversify recalled topics to avoid hammering the same subject. Default: true. */
  diversifyTopics?: boolean;
  /** Freshness bias for recall (0.0-1.0). Higher = favor recent facts. Default: 0.3. */
  freshnessBias?: number;
}

/** Proactive context for re-engagement triggers. */
export interface ProactiveContext {
  type: 'open_thread' | 'emotional_checkin' | 'milestone' | 'nostalgia' | 'prediction_followup';
  content: string;
  sourceFactId: string;
  priority: number;
  /** How long ago the source fact was created/referenced. */
  daysSince?: number;
}

/** Hierarchical memory context organized by tier. */
export interface HierarchicalContext {
  /** Current session turns (compressed). */
  working: string;
  /** Relevant episodic memories (events, readings, predictions). */
  episodic: string;
  /** Distilled identity facts, preferences, patterns. */
  semantic: string;
  /** Proactive surfacing (open threads, callbacks, milestones). */
  proactive: string;
  /** Emotional state awareness. */
  emotional: string;
  /** Total tokens used across all tiers. */
  totalTokens: number;
  /** The full formatted context string for prompt injection. */
  formatted: string;
}

/** Input for segment-level extraction. */
export interface SegmentInput {
  messages: Array<{ role: string; content: string }>;
  /** Topic label for this segment (e.g., "career", "relationship"). */
  topic?: string;
  /** Classification type from the caller (e.g., "QUESTION", "PREDICTION"). */
  classification?: string;
}

/** Options for building hierarchical memory context. */
export interface HierarchicalContextOptions {
  /** Maximum tokens for the working memory tier. */
  workingBudget?: number;
  /** Maximum tokens for the episodic tier. */
  episodicBudget?: number;
  /** Maximum tokens for the semantic tier. */
  semanticBudget?: number;
  /** Include proactive surfacing. Default: true. */
  includeProactive?: boolean;
  /** Include emotional state. Default: true. */
  includeEmotional?: boolean;
  /** Confidence floor — exclude facts below this confidence. */
  confidenceFloor?: number;
  /** Maximum age of facts to include (in days). */
  maxAgeDays?: number;
}
