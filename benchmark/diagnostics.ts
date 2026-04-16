/**
 * Benchmark diagnostics for answerability, retrieval overlap, and formatting loss.
 *
 * These heuristics are intentionally local to the benchmark. They do not change
 * MemG runtime behavior; they only analyze the stored benchmark artifacts.
 */

import type { QuestionResult } from './types.ts';
import { isRefusal } from './evaluator.ts';

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'shall', 'can', 'need',
  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
  'into', 'about', 'like', 'through', 'after', 'over', 'between', 'out',
  'against', 'during', 'without', 'before', 'under', 'around', 'among',
  'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
  'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
  'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
  'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what',
  'which', 'who', 'whom', 'this', 'that', 'these', 'those',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
  'she', 'her', 'it', 'its', 'they', 'them', 'their',
]);

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\b(a|an|the|is|are|was|were|do|does|did|has|have|had)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text: string): string[] {
  const normalized = normalize(text);
  if (!normalized) return [];
  return normalized
    .split(' ')
    .filter(Boolean)
    .filter((token) => !STOP_WORDS.has(token));
}

function tokenOverlap(left: string[], right: string[]): number {
  if (left.length === 0 || right.length === 0) return 0;

  const rightFreq = new Map<string, number>();
  for (const token of right) {
    rightFreq.set(token, (rightFreq.get(token) ?? 0) + 1);
  }

  let overlap = 0;
  for (const token of left) {
    const count = rightFreq.get(token) ?? 0;
    if (count <= 0) continue;
    overlap++;
    rightFreq.set(token, count - 1);
  }

  return overlap;
}

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

export interface DiagnosticsSummary {
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

export interface DiagnosticQuestionInput {
  groundTruth: string;
  predicted: string;
  memoryContext: string;
  f1: number;
  isAdversarial?: boolean;
}

export function buildQuestionDiagnostics(input: DiagnosticQuestionInput): QuestionDiagnostics {
  const refusalDetected = isRefusal(input.predicted);
  const groundTruthTokens = tokenize(input.groundTruth);
  const contextTokens = tokenize(input.memoryContext);
  const predictionTokens = tokenize(input.predicted);

  const groundTruthKeywordCount = groundTruthTokens.length;
  const groundTruthKeywordsInContext = tokenOverlap(groundTruthTokens, contextTokens);
  const groundTruthKeywordsInPrediction = tokenOverlap(groundTruthTokens, predictionTokens);

  const groundTruthContextCoverage = groundTruthKeywordCount > 0
    ? groundTruthKeywordsInContext / groundTruthKeywordCount
    : 0;
  const groundTruthPredictionCoverage = groundTruthKeywordCount > 0
    ? groundTruthKeywordsInPrediction / groundTruthKeywordCount
    : 0;

  const likelyRetrievalMiss =
    !input.isAdversarial &&
    groundTruthKeywordCount > 0 &&
    groundTruthKeywordsInContext === 0;

  const likelyFormatMismatch =
    !input.isAdversarial &&
    !refusalDetected &&
    input.f1 === 0 &&
    groundTruthKeywordCount > 0 &&
    groundTruthKeywordsInContext > 0 &&
    groundTruthKeywordsInPrediction === 0;

  return {
    refusalDetected,
    groundTruthKeywordCount,
    groundTruthKeywordsInContext,
    groundTruthKeywordsInPrediction,
    groundTruthContextCoverage,
    groundTruthPredictionCoverage,
    likelyFormatMismatch,
    likelyRetrievalMiss,
  };
}

export function summarizeDiagnostics(questions: QuestionResult[], category?: number): DiagnosticsSummary | undefined {
  const filtered = category === undefined
    ? questions
    : questions.filter((q) => q.category === category);

  const answerable = filtered.filter((q) => q.category !== 5);
  if (answerable.length === 0) return undefined;

  const diagnostics = answerable.map((q) => q.diagnostics ?? buildQuestionDiagnostics({
    groundTruth: q.groundTruth,
    predicted: q.predicted,
    memoryContext: q.memoryContext,
    f1: q.f1,
    isAdversarial: q.category === 5,
  }));

  const refusalDetectedCount = diagnostics.filter((d) => d.refusalDetected).length;
  const likelyFormatMismatchCount = diagnostics.filter((d) => d.likelyFormatMismatch).length;
  const likelyRetrievalMissCount = diagnostics.filter((d) => d.likelyRetrievalMiss).length;
  const zeroF1Count = answerable.filter((q) => q.f1 === 0).length;

  return {
    questionCount: filtered.length,
    answerableQuestionCount: answerable.length,
    refusalDetectedCount,
    refusalDetectedRate: refusalDetectedCount / answerable.length,
    likelyFormatMismatchCount,
    likelyFormatMismatchRate: likelyFormatMismatchCount / answerable.length,
    likelyRetrievalMissCount,
    likelyRetrievalMissRate: likelyRetrievalMissCount / answerable.length,
    zeroF1Count,
    zeroF1Rate: zeroF1Count / answerable.length,
    averageGroundTruthKeywordCount: avg(diagnostics.map((d) => d.groundTruthKeywordCount)),
    averageGroundTruthKeywordsInContext: avg(diagnostics.map((d) => d.groundTruthKeywordsInContext)),
    averageGroundTruthKeywordsInPrediction: avg(diagnostics.map((d) => d.groundTruthKeywordsInPrediction)),
    averageGroundTruthContextCoverage: avg(diagnostics.map((d) => d.groundTruthContextCoverage)),
    averageGroundTruthPredictionCoverage: avg(diagnostics.map((d) => d.groundTruthPredictionCoverage)),
  };
}
