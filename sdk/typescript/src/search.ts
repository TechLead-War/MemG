/**
 * Hybrid vector + lexical search engine.
 * Exact port of the Go search package.
 */

import type { Fact, RecalledFact } from './types';

/** A candidate for ranking. */
export interface SearchCandidate {
  id: string;
  content: string;
  embedding?: number[];
  createdAt?: string;
  temporalStatus: string;
  significance: number;
  confidence: number;
}

/** A scored result that passed the relevance threshold. */
interface ScoredItem {
  idx: number;
  score: number;
}

const STOP_WORDS: Set<string> = new Set([
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

function tokenize(text: string): string[] {
  const lower = text.toLowerCase();
  const words = lower.split(/[^a-z0-9\u00C0-\u024F]+/).filter(Boolean);
  return words.filter((w) => !STOP_WORDS.has(w));
}

function containsTerm(doc: string[], term: string): boolean {
  for (const w of doc) {
    if (w === term) return true;
  }
  return false;
}

function termFrequencies(doc: string[], terms: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const w of doc) {
    for (const t of terms) {
      if (w === t) {
        tf.set(t, (tf.get(t) ?? 0) + 1);
      }
    }
  }
  return tf;
}

/** Cosine similarity between two vectors. */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return dot / denom;
}

/** Check if two vectors have compatible dimensions. */
export function dimensionMatch(a: number[] | undefined, b: number[] | undefined): boolean {
  return !!a && !!b && a.length > 0 && b.length > 0 && a.length === b.length;
}

/** Compute cosine similarity for each candidate against the query vector. */
function vectorScores(query: number[], candidates: SearchCandidate[]): number[] {
  return candidates.map((c) => {
    if (!c.embedding || c.embedding.length === 0) return 0;
    return cosineSimilarity(query, c.embedding);
  });
}

/** Compute BM25 scores for each candidate, normalized to [0, 1]. */
function bm25Scores(queryText: string, candidates: SearchCandidate[]): number[] {
  const terms = tokenize(queryText);
  if (terms.length === 0 || candidates.length === 0) {
    return new Array(candidates.length).fill(0);
  }

  const docs = candidates.map((c) => tokenize(c.content));
  let avgLen = 0;
  for (const d of docs) avgLen += d.length;
  avgLen /= docs.length;

  const n = docs.length;
  const idf = new Map<string, number>();
  for (const t of terms) {
    let df = 0;
    for (const doc of docs) {
      if (containsTerm(doc, t)) df++;
    }
    idf.set(t, Math.log((n - df + 0.5) / (df + 0.5) + 1.0));
  }

  const k1 = 1.2;
  const b = 0.75;

  const raw: number[] = new Array(docs.length).fill(0);
  let peak = 0;

  for (let i = 0; i < docs.length; i++) {
    const dl = docs[i].length;
    const tf = termFrequencies(docs[i], terms);
    for (const t of terms) {
      const f = tf.get(t) ?? 0;
      const num = f * (k1 + 1);
      const denom = f + k1 * (1 - b + b * (dl / avgLen));
      raw[i] += (idf.get(t) ?? 0) * num / denom;
    }
    if (raw[i] > peak) peak = raw[i];
  }

  if (peak > 0) {
    return raw.map((v) => v / peak);
  }
  return raw;
}

/** Blend weights based on query length. */
function blendWeights(queryText: string): [number, number] {
  const tokens = tokenize(queryText);
  if (tokens.length <= 2) {
    return [0.70, 0.30];
  }
  return [0.85, 0.15];
}

/**
 * Kneedle algorithm to find the cutoff point in a sorted score curve.
 * Returns the number of items to keep.
 */
function kneedleCutoff(items: ScoredItem[]): number {
  const n = items.length;
  if (n <= 2) return n;

  const yMin = items[n - 1].score;
  const yMax = items[0].score;
  const yRange = yMax - yMin;
  if (yRange === 0) return n;

  let maxDiff = 0;
  let kneeIdx = 0;

  for (let i = 0; i < n; i++) {
    const xNorm = i / (n - 1);
    const yNorm = (items[i].score - yMin) / yRange;
    const diagonal = 1.0 - xNorm;
    const diff = yNorm - diagonal;

    if (diff > maxDiff) {
      maxDiff = diff;
      kneeIdx = i;
    }
  }

  const minKneeStrength = 0.10;
  if (maxDiff < minKneeStrength) return n;

  return Math.max(1, kneeIdx + 1);
}

export interface RankResult {
  id: string;
  content: string;
  score: number;
  createdAt?: string;
  temporalStatus: string;
  significance: number;
}

/**
 * Hybrid search engine. Ranks candidates using weighted combination of
 * cosine similarity and BM25 lexical relevance.
 */
export class HybridEngine {
  rank(
    query: number[],
    queryText: string,
    candidates: SearchCandidate[],
    limit: number,
    threshold: number
  ): RankResult[] {
    if (candidates.length === 0) return [];

    const dense = vectorScores(query, candidates);
    const lexical = bm25Scores(queryText, candidates);
    const [wDense, wLex] = blendWeights(queryText);

    const merged: ScoredItem[] = candidates.map((c, i) => {
      let score = wDense * dense[i] + wLex * lexical[i];

      if (c.temporalStatus === 'historical') {
        score *= 0.85;
      }

      if (c.confidence > 0 && c.confidence < 1.0) {
        score *= 0.95 + 0.05 * c.confidence;
      }

      score += c.significance * 0.001;

      return { idx: i, score };
    });

    merged.sort((a, b) => b.score - a.score);

    const above: ScoredItem[] = [];
    for (const s of merged) {
      if (s.score < threshold) break;
      above.push(s);
      if (above.length >= limit) break;
    }

    const cutoff = kneedleCutoff(above);

    const results: RankResult[] = [];
    for (let i = 0; i < cutoff; i++) {
      const c = candidates[above[i].idx];
      results.push({
        id: c.id,
        content: c.content,
        score: above[i].score,
        createdAt: c.createdAt,
        temporalStatus: c.temporalStatus,
        significance: c.significance,
      });
    }
    return results;
  }
}
