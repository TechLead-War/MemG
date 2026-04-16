/**
 * LoCoMo benchmark types.
 */

// ── Dataset types ──

export interface LoCoMoEntry {
  qa: LoCoMoQA[];
  conversation: LoCoMoConversation;
  event_summary: Record<string, Record<string, Array<{ date: string; event: string }>>>;
  observation: Record<string, Record<string, Array<[string, string]>>>;
  session_summary: Record<string, string>;
}

export interface LoCoMoQA {
  question: string;
  answer?: string | number;
  adversarial_answer?: string;
  evidence: string[];
  category: 1 | 2 | 3 | 4 | 5;
}

export interface LoCoMoConversation {
  speaker_a: string;
  speaker_b: string;
  [key: string]: any; // session_N, session_N_date_time, etc.
}

export interface LoCoMoTurn {
  speaker: string;
  dia_id: string;
  text: string;
  img_url?: string;
  blip_caption?: string;
  query?: string;
}

// ── Category names ──

export const CATEGORY_NAMES: Record<number, string> = {
  1: 'Multi-hop',
  2: 'Temporal',
  3: 'Open-domain',
  4: 'Single-hop',
  5: 'Adversarial',
};

// ── Benchmark result types ──

export interface QuestionResult {
  questionIdx: number;
  conversationIdx: number;
  question: string;
  category: number;
  categoryName: string;
  groundTruth: string;
  predicted: string;
  memoryContext: string;
  f1: number;
  exactMatch: boolean;
  llmJudge?: boolean;
  evidenceIds: string[];
  diagnostics?: QuestionDiagnostics;
}

export interface CategoryScore {
  category: number;
  name: string;
  total: number;
  f1Avg: number;
  exactMatchRate: number;
  llmJudgeRate?: number;
  diagnostics?: DiagnosticSummary;
}

export interface ConversationResult {
  conversationIdx: number;
  speakerA: string;
  speakerB: string;
  sessions: number;
  turns: number;
  totalQuestions: number;
  factsExtracted: number;
  questions: QuestionResult[];
  categoryScores: CategoryScore[];
  overallF1: number;
  overallExactMatch: number;
  overallLLMJudge?: number;
  diagnostics?: DiagnosticSummary;
  ingestionTimeMs: number;
  evaluationTimeMs: number;
}

export interface BenchmarkRun {
  id: string;
  startedAt: string;
  completedAt?: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  config: BenchmarkConfig;
  conversations: ConversationResult[];
  overall: OverallScore;
  progress: BenchmarkProgress;
  error?: string;
}

export interface OverallScore {
  totalQuestions: number;
  totalFacts: number;
  f1Avg: number;
  exactMatchRate: number;
  llmJudgeRate?: number;
  categoryScores: CategoryScore[];
  /** Adversarial rejection rate (higher = better). */
  adversarialRejectionRate?: number;
  diagnostics?: DiagnosticSummary;
}

export interface BenchmarkConfig {
  /** Which conversations to run (0-9). Empty = all. */
  conversations: number[];
  /** Which categories to evaluate (1-5). Empty = all. */
  categories: number[];
  /** LLM provider for extraction and answering. */
  llmProvider: string;
  /** LLM model for extraction. */
  extractionModel: string;
  /** LLM model for answering questions. */
  answerModel: string;
  /** LLM model for judge evaluation. */
  judgeModel: string;
  /** Embedding provider. */
  embedProvider: string;
  /** Whether to use LLM-as-judge evaluation. */
  useLLMJudge: boolean;
  /** API key (not stored in results). */
  apiKey?: string;
  /** Max conversations to run in parallel (default: 1 = sequential). */
  concurrency?: number;
  /** Questions per parallel batch during evaluation (default: 5). */
  questionBatchSize?: number;
}

export interface BenchmarkProgress {
  totalConversations: number;
  completedConversations: number;
  totalQuestions: number;
  completedQuestions: number;
  currentConversation?: number;
  currentPhase?: 'ingesting' | 'evaluating';
  /** Total sessions across all conversations to ingest. */
  totalSessions?: number;
  /** Sessions ingested so far. */
  completedSessions?: number;
}

export interface QuestionDiagnostics {
  refusalDetected: boolean;
  groundTruthKeywordCount: number;
  groundTruthKeywordsInContext: number;
  groundTruthKeywordsInPrediction: number;
  groundTruthContextCoverage: number;
  groundTruthPredictionCoverage: number;
  likelyFormatMismatch: boolean;
  likelyRetrievalMiss: boolean;
}

export interface DiagnosticSummary {
  questionCount: number;
  answerableQuestionCount: number;
  refusalDetectedCount: number;
  refusalDetectedRate?: number;
  likelyFormatMismatchCount: number;
  likelyFormatMismatchRate?: number;
  likelyRetrievalMissCount: number;
  likelyRetrievalMissRate?: number;
  zeroF1Count: number;
  zeroF1Rate?: number;
  averageGroundTruthKeywordCount: number;
  averageGroundTruthKeywordsInContext: number;
  averageGroundTruthKeywordsInPrediction: number;
  averageGroundTruthContextCoverage: number;
  averageGroundTruthPredictionCoverage: number;
}
