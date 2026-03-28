/**
 * Context builder: merges conscious facts, recalled facts, and summaries
 * into a single context string under a token budget. Matches Go's BuildContext.
 */

import type { Artifact, ConsciousFact, RecalledFact, RecalledSummary, TurnSummary } from './types.js';

export interface ContextBudget {
  totalTokens: number;
  summaryTokens: number;
}

export interface ContextInput {
  consciousFacts: ConsciousFact[];
  recalledFacts: RecalledFact[];
  summaries: RecalledSummary[];
  turnSummaries?: TurnSummary[];
  artifacts?: Artifact[];
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
  if (runes === 0) return 0;
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
 * Build context string. Priority: conscious > turn summaries > recalled > artifacts > summaries.
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

  // 2. Session context (turn summaries).
  const turnSummaries = input.turnSummaries ?? [];
  if (turnSummaries.length > 0) {
    const header = '\nSession context:';
    const headerTokens = estimateTokens(header + '\n');
    let sectionTokens = headerTokens;
    const lines: string[] = [header];
    let any = false;

    for (const ts of turnSummaries) {
      let line: string;
      if (ts.isOverview) {
        line = `- [Overview] ${ts.summary}`;
      } else {
        line = `- [Turns ${ts.startTurn}-${ts.endTurn}] ${ts.summary}`;
      }
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

  // 3. Recalled facts (medium priority) — dedup against conscious.
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

  // 4. Relevant code/data (artifacts).
  const artifacts = input.artifacts ?? [];
  if (artifacts.length > 0) {
    const header = '\nRelevant code/data:';
    const headerTokens = estimateTokens(header + '\n');
    let sectionTokens = headerTokens;
    const lines: string[] = [header];
    let any = false;

    for (const a of artifacts) {
      let entry = `- [${a.artifactType}`;
      if (a.language) {
        entry += `:${a.language}`;
      }
      entry += `] ${a.description}\n`;
      if (a.artifactType === 'code' && a.language) {
        entry += `  \`\`\`${a.language}\n`;
        entry += `  ${a.content}\n`;
        entry += '  ```\n';
      } else {
        entry += `  ${a.content}\n`;
      }

      const est = estimateTokens(entry);
      if (tokensUsed + sectionTokens + est > budget) break;
      lines.push(entry.replace(/\n$/, ''));
      sectionTokens += est;
      any = true;
    }

    if (any) {
      parts.push(lines.join('\n'));
      tokensUsed += sectionTokens;
    }
  }

  // 5. Summaries (lowest priority, own sub-budget).
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

/**
 * Calculate how many recent messages to load based on remaining token budget.
 * Matches Go AdaptiveWindowSize.
 */
export function adaptiveWindowSize(
  totalBudget: number,
  usedTokens: number,
  defaultTurns: number
): number {
  let remaining = totalBudget - usedTokens;
  const floorTokens = 800;
  if (remaining < floorTokens) remaining = floorTokens;
  const avgTokensPerMsg = 200;
  let maxMsgs = Math.floor(remaining / avgTokensPerMsg);
  if (maxMsgs > defaultTurns) maxMsgs = defaultTurns;
  const minMsgs = Math.floor(floorTokens / avgTokensPerMsg);
  if (maxMsgs < minMsgs) maxMsgs = minMsgs;
  return maxMsgs;
}
