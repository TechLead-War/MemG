/**
 * Hybrid vector + lexical search engine.
 * Exact port of the Go search package.
 */

import type { Fact, RecalledFact } from './types.js';

/** A candidate for ranking. */
export interface SearchCandidate {
  id: string;
  content: string;
  embedding?: number[];
  createdAt?: string;
  temporalStatus: string;
  significance: number;
  confidence: number;
  /** Emotional weight (0.0-1.0). Higher = more emotionally significant. */
  emotionalWeight?: number;
  /** Thread status for ongoing situations. */
  threadStatus?: string;
  /** Semantic tag for topic diversification. */
  tag?: string;
  /** Whether this candidate is pinned by the user. */
  pinned?: boolean;
  /** Engagement score (0.0-1.0). */
  engagementScore?: number;
  /** ISO date string for temporal relevance matching. */
  referenceTime?: string;
}

/** A scored result that passed the relevance threshold. */
interface ScoredItem {
  idx: number;
  score: number;
}

interface RankedCandidate {
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
  const words = lower.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
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
  const termSet = new Set(terms);
  for (const w of doc) {
    if (termSet.has(w)) {
      tf.set(w, (tf.get(w) ?? 0) + 1);
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

/** Month names and abbreviations mapped to zero-padded month numbers. */
const MONTH_MAP: Record<string, string> = {
  january: '01', jan: '01',
  february: '02', feb: '02',
  march: '03', mar: '03',
  april: '04', apr: '04',
  may: '05',
  june: '06', jun: '06',
  july: '07', jul: '07',
  august: '08', aug: '08',
  september: '09', sep: '09', sept: '09',
  october: '10', oct: '10',
  november: '11', nov: '11',
  december: '12', dec: '12',
};

/** Patterns indicating the query is asking about time/dates. */
const TEMPORAL_INTENT_PATTERNS = [
  /\bwhen\s+did\b/i,
  /\bwhen\s+was\b/i,
  /\bwhen\s+does\b/i,
  /\bwhen\s+will\b/i,
  /\bwhat\s+date\b/i,
  /\bwhat\s+time\b/i,
  /\bhow\s+long\b/i,
  /\bhow\s+many\s+(?:days|weeks|months|years)\b/i,
  /\bbefore\s+\w+\s+\d/i,
  /\bafter\s+\w+\s+\d/i,
  /\bon\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b/i,
  /\bin\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b/i,
  /\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b/i,
  /\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b/i,
  /\bhow\s+many\s+weeks\s+passed\b/i,
];

function hasTemporalIntent(query: string): boolean {
  return TEMPORAL_INTENT_PATTERNS.some((p) => p.test(query));
}

/**
 * Compute a temporal relevance boost by matching date components in the query
 * against a fact's referenceTime. Also gives a baseline boost to facts WITH
 * temporal data when the query has temporal intent (e.g., "When did X happen?").
 */
function computeTemporalBoost(query: string, referenceTime: string): number {
  const q = query.toLowerCase();
  const ref = referenceTime.toLowerCase();

  // Extract 4-digit years from query.
  const yearMatches = q.match(/\b(19|20)\d{2}\b/g) ?? [];
  // Extract month mentions from query.
  const queryMonths: string[] = [];
  for (const [name, num] of Object.entries(MONTH_MAP)) {
    if (q.includes(name)) {
      queryMonths.push(num);
    }
  }
  // Also detect numeric months in date-like patterns (e.g., "01/15", "2023-01").
  const numericMonthPatterns = q.match(/\b(\d{1,2})[\/\-](\d{1,2})\b/g) ?? [];
  for (const pat of numericMonthPatterns) {
    const parts = pat.split(/[\/\-]/);
    const m = parseInt(parts[0], 10);
    if (m >= 1 && m <= 12) {
      queryMonths.push(m.toString().padStart(2, '0'));
    }
  }

  // Extract day-of-month mentions from query.
  const queryDays: string[] = [];
  // "January 15" or "15 January" patterns.
  const dayAfterMonth = q.match(/(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})\b/g) ?? [];
  for (const m of dayAfterMonth) {
    const d = m.match(/(\d{1,2})$/);
    if (d) queryDays.push(d[1].padStart(2, '0'));
  }
  const dayBeforeMonth = q.match(/\b(\d{1,2})\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b/g) ?? [];
  for (const m of dayBeforeMonth) {
    const d = m.match(/^(\d{1,2})/);
    if (d) queryDays.push(d[1].padStart(2, '0'));
  }
  // ISO-style dates: 2023-01-15 or slash dates.
  const isoDateMatches = q.match(/\b(19|20)\d{2}[\/\-]\d{1,2}[\/\-](\d{1,2})\b/g) ?? [];
  for (const m of isoDateMatches) {
    const parts = m.split(/[\/\-]/);
    if (parts.length === 3) {
      queryDays.push(parts[2].padStart(2, '0'));
      const isoMonth = parseInt(parts[1], 10);
      if (isoMonth >= 1 && isoMonth <= 12) {
        queryMonths.push(isoMonth.toString().padStart(2, '0'));
      }
    }
  }

  // If the query has explicit date components, try exact matching.
  if (yearMatches.length > 0 || queryMonths.length > 0) {
    const yearMatch = yearMatches.some((y) => ref.includes(y));
    const monthMatch = queryMonths.some((m) => {
      return ref.includes(`-${m}-`) || ref.includes(`-${m}`) || ref.includes(`/${m}/`) || ref.includes(`/${m}`);
    });
    const dayMatch = queryDays.some((d) => {
      return ref.includes(`-${d}`) || ref.includes(`/${d}`);
    });

    if (yearMatch && monthMatch && dayMatch) return 0.5;
    if (yearMatch && monthMatch) return 0.35;
    if (yearMatch) return 0.2;
    if (monthMatch) return 0.15;
  }

  // If query has temporal intent ("When did...", "What date...", "How long..."),
  // boost ANY fact that has a referenceTime — these are the facts that contain
  // the temporal information the user is asking for.
  if (hasTemporalIntent(q) && ref) {
    return 0.20;
  }

  return 0.0;
}

/**
 * Extract likely entity names (capitalized words, multi-word names) from query.
 */
function extractQueryEntities(query: string): string[] {
  const entities: string[] = [];
  // Match capitalized words that aren't at start of sentence and common question words.
  const skipWords = new Set(['what', 'when', 'where', 'which', 'who', 'whom', 'how', 'why', 'did', 'does', 'do', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'not', 'with', 'from', 'by', 'about', 'this', 'that', 'these', 'those']);
  // Find capitalized words (likely proper nouns / entity names).
  const words = query.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g) ?? [];
  for (const w of words) {
    if (!skipWords.has(w.toLowerCase())) {
      entities.push(w.toLowerCase());
    }
  }
  return entities;
}

/**
 * Compute entity name match boost. If query mentions "Dave" and fact contains "Dave", boost.
 */
function entityBoost(queryEntities: string[], candidateContent: string): number {
  const matches = entityMatchCount(queryEntities, candidateContent);
  if (matches === 0) return 0;
  // Significant boost for entity matches — these are almost always relevant.
  return 0.15 * Math.min(matches, 2);
}

function entityMatchCount(queryEntities: string[], candidateContent: string): number {
  if (queryEntities.length === 0) return 0;
  const lower = candidateContent.toLowerCase();
  let matches = 0;
  for (const entity of queryEntities) {
    if (lower.includes(entity)) matches++;
  }
  return matches;
}

/** Blend weights — balanced to give BM25 meaningful influence. */
function blendWeights(queryText: string): [number, number] {
  const tokens = tokenize(queryText);
  if (tokens.length <= 2) {
    return [0.55, 0.45];
  }
  return [0.55, 0.45];
}

/**
 * Finds the "knee" in sorted scores using the Kneedle algorithm
 * (Satopaa et al., ICDCS 2011). Items must be sorted by score descending.
 * Normalizes both axes to [0,1], draws a diagonal from the first point
 * (highest score) to the last point (lowest score), and finds the point
 * where the actual curve deviates most below the diagonal.
 * Returns the number of items to keep.
 *
 * Only triggers the cutoff when there is a dramatic score drop (deviation > 0.3).
 * This prevents over-aggressive pruning that discards relevant-but-lower-scored facts.
 */
function kneedleCutoff(items: ScoredItem[]): number {
  if (items.length <= 5) return items.length;

  const n = items.length;
  const maxScore = items[0].score;
  const minScore = items[n - 1].score;
  const scoreRange = maxScore - minScore;

  if (scoreRange < 1e-9) return n;

  let bestDeviation = 0;
  let kneeIdx = n;

  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    const y = (items[i].score - minScore) / scoreRange;
    const diagonal = 1.0 - x;
    const deviation = diagonal - y;

    if (deviation > bestDeviation) {
      bestDeviation = deviation;
      kneeIdx = i;
    }
  }

  // Only apply the cutoff if there's a truly dramatic score drop.
  // A gentle slope means all items are roughly equally relevant.
  if (bestDeviation < 0.3) return n;

  // Ensure minimum results preserved.
  const minResults = Math.min(10, items.length);
  if (kneeIdx < minResults) return minResults;

  if (kneeIdx <= 0) return 1;
  if (kneeIdx >= n) return n;
  return kneeIdx;
}

function rankChannel(
  scores: number[],
  windowSize: number,
  predicate?: (idx: number, score: number) => boolean
): RankedCandidate[] {
  const ranked: RankedCandidate[] = [];
  for (let i = 0; i < scores.length; i++) {
    const score = scores[i];
    if (predicate && !predicate(i, score)) continue;
    ranked.push({ idx: i, score });
  }
  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, windowSize);
}

function reciprocalRankFuse(
  rankings: RankedCandidate[][],
  windowSize: number,
  k: number = 60
): RankedCandidate[] {
  const fused = new Map<number, number>();

  for (const ranking of rankings) {
    for (let i = 0; i < ranking.length; i++) {
      const cur = fused.get(ranking[i].idx) ?? 0;
      fused.set(ranking[i].idx, cur + 1 / (k + i + 1));
    }
  }

  return [...fused.entries()]
    .map(([idx, score]) => ({ idx, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, windowSize);
}

export interface RankResult {
  id: string;
  content: string;
  score: number;
  createdAt?: string;
  temporalStatus: string;
  significance: number;
  /** Emotional weight carried from the candidate. */
  emotionalWeight?: number;
  /** Semantic tag carried from the candidate. */
  tag?: string;
  /** Confidence carried from the candidate. */
  confidence: number;
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
    threshold: number,
    opts?: { emotionalBoost?: number; diversifyTopics?: boolean; pinnedBoost?: number }
  ): RankResult[] {
    if (candidates.length === 0) return [];

    const dense = vectorScores(query, candidates);
    const lexical = bm25Scores(queryText, candidates);
    const queryEntities = extractQueryEntities(queryText);
    const temporalIntent = hasTemporalIntent(queryText);
    const channelWindow = Math.min(
      candidates.length,
      Math.max(limit * 4, Math.min(100, candidates.length))
    );

    const denseRanking = rankChannel(
      dense,
      channelWindow,
      (_, score) => score > 0
    );
    const lexicalRanking = rankChannel(
      lexical,
      channelWindow,
      (_, score) => score > 0
    );
    const entityRanking = rankChannel(
      candidates.map((c, i) => {
        const matches = entityMatchCount(queryEntities, c.content);
        if (matches === 0) return 0;
        return matches + lexical[i] * 0.35 + dense[i] * 0.15;
      }),
      channelWindow,
      (_, score) => score > 0
    );
    const temporalRanking = rankChannel(
      candidates.map((c, i) => {
        if (!temporalIntent || !c.referenceTime) return 0;
        const temporalScore = computeTemporalBoost(queryText, c.referenceTime);
        if (temporalScore <= 0) return 0;
        const entityScore = entityMatchCount(queryEntities, c.content);
        return temporalScore + entityScore * 0.35 + lexical[i] * 0.15 + dense[i] * 0.1;
      }),
      channelWindow,
      (_, score) => score > 0
    );
    const activeRankings = [denseRanking, lexicalRanking];
    if (entityRanking.length > 0) activeRankings.push(entityRanking);
    if (temporalRanking.length > 0) activeRankings.push(temporalRanking);

    const fused = reciprocalRankFuse(activeRankings, channelWindow);
    const maxFusedScore = fused.length > 0 ? fused[0].score : 0;

    const merged: ScoredItem[] = fused.map(({ idx, score }) => {
      let finalScore = maxFusedScore > 0 ? score / maxFusedScore : 0;
      const c = candidates[idx];

      if (c.temporalStatus === 'historical') {
        finalScore *= 0.95;
      }

      let conf = c.confidence;
      if (conf <= 0) conf = 1.0;
      if (conf < 1.0) {
        finalScore *= 0.95 + 0.05 * conf;
      }

      // Emotional boost: emotionally significant facts get a small relevance boost (multiplicative).
      finalScore *= 1.0 + (c.emotionalWeight ?? 0) * (opts?.emotionalBoost ?? 0.05);

      // Pinned boost: user-pinned facts get a fixed multiplicative boost.
      if (c.pinned) {
        finalScore *= 1.0 + (opts?.pinnedBoost ?? 0.10);
      }

      // Open thread boost: unresolved situations get a small multiplicative boost.
      if (c.threadStatus === 'open') {
        finalScore *= 1.03;
      }

      // Engagement boost: highly-engaged topics get a minor multiplicative boost.
      finalScore *= 1.0 + (c.engagementScore ?? 0) * 0.02;

      // Preserve a small direct entity boost so entity-anchored facts still
      // break ties even when RRF scores are close.
      finalScore += entityBoost(queryEntities, c.content) * 0.1;

      return { idx, score: finalScore };
    });

    merged.sort((a, b) => b.score - a.score);

    const above: ScoredItem[] = [];
    for (const s of merged) {
      if (s.score < threshold) break;
      above.push(s);
      if (above.length >= limit) break;
    }

    // Add significance as tiebreaker for sort order only (after threshold filtering).
    for (const s of above) {
      s.score += candidates[s.idx].significance * 0.001;
    }
    above.sort((a, b) => b.score - a.score);

    // No kneedle cutoff — let the threshold and limit do the filtering.
    // Kneedle was over-aggressively pruning relevant facts.
    const cutoff = above.length;

    let results: RankResult[] = [];
    for (let i = 0; i < cutoff; i++) {
      const c = candidates[above[i].idx];
      results.push({
        id: c.id,
        content: c.content,
        score: above[i].score,
        createdAt: c.createdAt,
        temporalStatus: c.temporalStatus,
        significance: c.significance,
        emotionalWeight: c.emotionalWeight,
        tag: c.tag,
        confidence: c.confidence,
      });
    }

    // Topic diversification: prevent any single tag from dominating results.
    if (opts?.diversifyTopics && results.length > 0) {
      const tagCounts = new Map<string, number>();
      for (const r of results) {
        const t = r.tag ?? '';
        tagCounts.set(t, (tagCounts.get(t) ?? 0) + 1);
      }

      const maxAllowed = Math.max(1, Math.floor(results.length * 0.6));
      if (results.length > 2) {
        for (const [tag, count] of tagCounts) {
          if (count > maxAllowed) {
            let seen = 0;
            for (const r of results) {
              if ((r.tag ?? '') === tag) {
                seen++;
                if (seen > maxAllowed) {
                  r.score *= 0.85;
                }
              }
            }
          }
        }
        results.sort((a, b) => b.score - a.score);
      }
    }

    return results;
  }
}
