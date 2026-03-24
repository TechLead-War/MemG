/**
 * Context builder: merges conscious facts, recalled facts, and summaries
 * into a single context string under a token budget. Matches Go's BuildContext.
 */

import type { ConsciousFact, RecalledFact, RecalledSummary } from './types.js';

export interface ContextBudget {
  totalTokens: number;
  summaryTokens: number;
}

export interface ContextInput {
  consciousFacts: ConsciousFact[];
  recalledFacts: RecalledFact[];
  summaries: RecalledSummary[];
  budget: ContextBudget;
}

/**
 * Token estimation: ceil(wordCount * 1.3).
 * For CJK text with few spaces, falls back to rune-based estimation.
 */
export function estimateTokens(s: string): number {
  const words = s.split(/\s+/).filter(Boolean).length;
  if (words > 0) {
    return Math.ceil(words * 1.3);
  }
  // No spaces (likely CJK) — roughly 1 token per 1.5 characters.
  const runes = [...s].length;
  if (runes === 0) return 1;
  return Math.ceil((runes * 2) / 3);
}

/**
 * Normalize content for dedup. Strips common prefixes, lowercases, trims.
 */
function normalizeForDedup(s: string): string {
  let n = s.toLowerCase().trim();
  for (const prefix of ['the user ', 'user ', "user's ", '[historical] ']) {
    if (n.startsWith(prefix)) {
      n = n.slice(prefix.length);
    }
  }
  return n.trim();
}

/**
 * Format a date string as "Jan 2, 2006" for summary display.
 */
function formatSummaryDate(isoDate: string | undefined): string {
  if (!isoDate) return 'Unknown date';
  try {
    const d = new Date(isoDate);
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
  } catch {
    return 'Unknown date';
  }
}

/**
 * Build context string. Priority: conscious > recalled > summaries.
 * Dedup by normalized content across all components.
 */
export function buildContext(input: ContextInput): string {
  let budget = input.budget.totalTokens;
  if (budget <= 0) budget = 4000;
  let summaryBudget = input.budget.summaryTokens;
  if (summaryBudget <= 0) summaryBudget = 1000;

  const parts: string[] = [];
  const seen = new Set<string>();
  let tokensUsed = 0;

  // 1. Conscious facts (highest priority).
  if (input.consciousFacts.length > 0) {
    const lines: string[] = ['User profile:'];
    for (const f of input.consciousFacts) {
      const normalized = normalizeForDedup(f.content);
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      const line = `- ${f.content}`;
      const est = estimateTokens(line + '\n');
      if (tokensUsed + est > budget) break;
      lines.push(line);
      tokensUsed += est;
    }
    if (lines.length > 1) {
      parts.push(lines.join('\n'));
    }
  }

  // 2. Recalled facts (medium priority).
  if (input.recalledFacts.length > 0) {
    const header = '\nRelevant context from memory:';
    const headerTokens = estimateTokens(header + '\n');
    let sectionTokens = headerTokens;
    const lines: string[] = [header];
    let any = false;

    for (const f of input.recalledFacts) {
      const normalized = normalizeForDedup(f.content);
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      let line = '- ';
      if (f.temporalStatus === 'historical') {
        line += '[historical] ';
      }
      line += f.content;
      const est = estimateTokens(line + '\n');
      if (tokensUsed + sectionTokens + est > budget) break;
      lines.push(line);
      sectionTokens += est;
      any = true;
    }

    if (any) {
      parts.push(lines.join('\n'));
      tokensUsed += sectionTokens;
    }
  }

  // 3. Summaries (lowest priority, own sub-budget).
  if (input.summaries.length > 0) {
    let effectiveBudget = summaryBudget;
    const remaining = budget - tokensUsed;
    if (effectiveBudget > remaining) effectiveBudget = remaining;

    if (effectiveBudget > 0) {
      const header = '\nRelevant past conversations:';
      const headerTokens = estimateTokens(header + '\n');
      let sectionTokens = headerTokens;
      const lines: string[] = [header];
      let any = false;

      for (const s of input.summaries) {
        const normalized = normalizeForDedup(s.summary);
        if (seen.has(normalized)) continue;
        seen.add(normalized);
        const dateStr = formatSummaryDate(s.createdAt);
        const line = `- [${dateStr}] ${s.summary}`;
        const est = estimateTokens(line + '\n');
        if (sectionTokens + est > effectiveBudget) break;
        lines.push(line);
        sectionTokens += est;
        any = true;
      }

      if (any) {
        parts.push(lines.join('\n'));
      }
    }
  }

  return parts.join('\n').replace(/\n+$/, '');
}
