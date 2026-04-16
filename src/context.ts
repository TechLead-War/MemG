/**
 * Context builder: merges conscious facts, recalled facts, and summaries
 * into a single context string under a token budget. Matches Go's BuildContext.
 */

import type {
  Artifact,
  ConsciousFact,
  HierarchicalContext,
  ProactiveContext,
  RecalledFact,
  RecalledSummary,
  TurnSummary,
} from './types.js';

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
  } catch (err) {
    console.warn('[memg] context: formatSummaryDate failed:', err);
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
      if (f.referenceTime) {
        line += `[${f.referenceTime}] `;
      }
      line += f.content;
      const est = estimateTokens(line + '\n');
      if (tokensUsed + sectionTokens + est > budget) {
        // Always include at least top 5 recalled facts even if over budget
        if (lines.length <= 5) {
          // Allow slight budget overflow for critical facts
        } else {
          break;
        }
      }
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

/* ------------------------------------------------------------------ */
/*  Hierarchical, confidence-graded, emotionally-aware context builder */
/* ------------------------------------------------------------------ */

export interface HierarchicalContextInput {
  consciousFacts: ConsciousFact[];
  recalledFacts: RecalledFact[];
  summaries: RecalledSummary[];
  turnSummaries?: TurnSummary[];
  artifacts?: Artifact[];
  openThreads?: RecalledFact[];
  emotionalFacts?: RecalledFact[];
  proactiveItems?: ProactiveContext[];
  budget: ContextBudget;
  options?: {
    maxPersonalFacts?: number;
    diversifyTopics?: boolean;
    confidenceFloor?: number;
  };
}

/** Map a confidence value to a human-readable label. */
function confidenceLabel(c: number): string {
  if (c >= 0.8) return 'verified';
  if (c >= 0.5) return 'likely';
  return 'inferred';
}

/**
 * Format an ISO date string as a relative duration ("3 days ago", "2 weeks ago", etc.).
 * Falls back to the formatted date if the input is missing or unparseable.
 */
function relativeDate(iso: string | undefined): string {
  if (!iso) return 'unknown time ago';
  try {
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return 'unknown time ago';
    const now = Date.now();
    const diffMs = now - then;
    if (diffMs < 0) return 'just now';
    const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    if (days === 0) return 'today';
    if (days === 1) return '1 day ago';
    if (days < 7) return `${days} days ago`;
    const weeks = Math.floor(days / 7);
    if (weeks === 1) return '1 week ago';
    if (weeks < 5) return `${weeks} weeks ago`;
    const months = Math.floor(days / 30);
    if (months === 1) return '1 month ago';
    return `${months} months ago`;
  } catch (err) {
    console.warn('[memg] context: relativeDate failed:', err);
    return 'unknown time ago';
  }
}

/**
 * Build a hierarchical, confidence-graded, emotionally-aware context string.
 *
 * Sections are rendered in priority order:
 *   1. IDENTITY (semantic tier)
 *   2. EMOTIONAL STATE
 *   3. OPEN THREADS
 *   4. RECALLED CONTEXT (episodic tier, split by confidence grade)
 *   5. PROACTIVE CONTEXT
 *   6. SESSION CONTEXT (working memory tier)
 *   7. PAST CONVERSATIONS (summaries)
 *
 * Returns a HierarchicalContext with per-tier text, totalTokens, and formatted string.
 */
export function buildHierarchicalContext(input: HierarchicalContextInput): HierarchicalContext {
  let budget = input.budget.totalTokens;
  if (budget <= 0) budget = 4000;
  let summaryBudget = input.budget.summaryTokens;
  if (summaryBudget <= 0) summaryBudget = 1000;

  const confidenceFloor = input.options?.confidenceFloor ?? 0.3;
  const maxPersonalFacts = input.options?.maxPersonalFacts;
  const diversifyTopics = input.options?.diversifyTopics ?? false;

  let tokensUsed = 0;
  let personalFactCount = 0;

  const canAddPersonal = (): boolean =>
    maxPersonalFacts === undefined || personalFactCount < maxPersonalFacts;

  const tierTexts = {
    semantic: '',
    emotional: '',
    episodic: '',
    proactive: '',
    working: '',
  };

  const allParts: string[] = [];

  // Helper: append a section and track budget.
  const commitSection = (text: string, tier: keyof typeof tierTexts): void => {
    if (!text) return;
    const tokens = estimateTokens(text);
    tierTexts[tier] = text;
    tokensUsed += tokens;
    allParts.push(text);
  };

  // ── 1. [IDENTITY] — Semantic tier ─────────────────────────────────
  {
    const eligible = input.consciousFacts.filter(
      (f) => (f.confidence ?? 1.0) >= confidenceFloor
    );
    if (eligible.length > 0) {
      // Group by tag.
      const byTag = new Map<string, ConsciousFact[]>();
      for (const f of eligible) {
        const tag = f.tag || 'general';
        let group = byTag.get(tag);
        if (!group) {
          group = [];
          byTag.set(tag, group);
        }
        group.push(f);
      }

      const lines: string[] = ['[IDENTITY] Who this user is:'];
      let sectionTokens = estimateTokens(lines[0] + '\n');

      for (const [, facts] of byTag) {
        for (const f of facts) {
          if (!canAddPersonal()) break;
          const label = confidenceLabel(f.confidence ?? 1.0);
          let line = `- ${f.content} (${label})`;
          if (f.verbatim) {
            const escaped = f.verbatim.replace(/"/g, '\\"');
            line += ` | User said: "${escaped}"`;
          }
          const est = estimateTokens(line + '\n');
          if (tokensUsed + sectionTokens + est > budget) break;
          lines.push(line);
          sectionTokens += est;
          personalFactCount++;
        }
      }

      if (lines.length > 1) {
        const text = lines.join('\n');
        commitSection(text, 'semantic');
      }
    }
  }

  // ── 2. [EMOTIONAL STATE] — Emotional awareness ────────────────────
  {
    const emotionalFacts = input.emotionalFacts ?? [];
    if (emotionalFacts.length > 0) {
      // Sort by recency (most recent first).
      const sorted = [...emotionalFacts].sort((a, b) => {
        const ta = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const tb = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return tb - ta;
      });

      const lines: string[] = ['[EMOTIONAL STATE] Recent emotional context:'];
      let sectionTokens = estimateTokens(lines[0] + '\n');

      for (const f of sorted) {
        if (!canAddPersonal()) break;
        const conf = f.confidence ?? 0.5;
        if (conf < confidenceFloor) continue;
        const when = relativeDate(f.createdAt);
        const confStr = conf >= 0.8 ? ', confidence: high' : conf >= 0.5 ? ', confidence: medium' : '';
        let line = `- ${f.content} (${when}${confStr})`;
        if (f.verbatim) {
          const escaped = f.verbatim.replace(/"/g, '\\"');
          line += `\n  User said: "${escaped}"`;
        }
        const est = estimateTokens(line + '\n');
        if (tokensUsed + sectionTokens + est > budget) break;
        lines.push(line);
        sectionTokens += est;
        personalFactCount++;
      }

      if (lines.length > 1) {
        const text = lines.join('\n');
        commitSection(text, 'emotional');
      }
    }
  }

  // ── 3. [OPEN THREADS] — Zeigarnik-driven re-engagement ────────────
  {
    const openThreads = input.openThreads ?? [];
    if (openThreads.length > 0) {
      const lines: string[] = ['[OPEN THREADS] Unresolved topics to follow up on:'];
      let sectionTokens = estimateTokens(lines[0] + '\n');

      for (const f of openThreads) {
        if (!canAddPersonal()) break;
        const conf = f.confidence ?? 0.5;
        if (conf < confidenceFloor) continue;
        const tag = f.tag || 'General';
        const startedStr = f.startedAt ? `, started ${relativeDate(f.startedAt)}` : '';
        const line = `- ${tag}: ${f.content} (still open${startedStr})`;
        const est = estimateTokens(line + '\n');
        if (tokensUsed + sectionTokens + est > budget) break;
        lines.push(line);
        sectionTokens += est;
        personalFactCount++;
      }

      if (lines.length > 1) {
        const text = lines.join('\n');
        // Open threads contribute to the proactive tier text.
        commitSection(text, 'proactive');
      }
    }
  }

  // ── 4. [RECALLED CONTEXT] — Episodic tier, split by confidence ────
  {
    let facts = input.recalledFacts.filter(
      (f) => (f.confidence ?? 0.5) >= confidenceFloor
    );

    // Diversification: cap any single tag at 40% of total.
    if (diversifyTopics && facts.length > 0) {
      const maxPerTag = Math.max(1, Math.ceil(facts.length * 0.4));
      const tagCounts = new Map<string, number>();
      facts = facts.filter((f) => {
        const tag = f.tag || '__untagged__';
        const count = tagCounts.get(tag) ?? 0;
        if (count >= maxPerTag) return false;
        tagCounts.set(tag, count + 1);
        return true;
      });
    }

    const verified: RecalledFact[] = [];
    const inferred: RecalledFact[] = [];
    for (const f of facts) {
      const c = f.confidence ?? 0.5;
      if (c >= 0.8) {
        verified.push(f);
      } else {
        inferred.push(f);
      }
    }

    const sectionLines: string[] = [];
    let sectionTokens = 0;

    // 4a. Verified facts.
    if (verified.length > 0) {
      const header = '[RECALLED CONTEXT \u2014 VERIFIED] Facts the user explicitly stated:';
      sectionLines.push(header);
      sectionTokens += estimateTokens(header + '\n');

      for (const f of verified) {
        let line = `- ${f.content} (confidence: ${(f.confidence ?? 1.0).toFixed(1)})`;
        if (f.verbatim) {
          const escaped = f.verbatim.replace(/"/g, '\\"');
          line += ` | User said: "${escaped}"`;
        }
        const est = estimateTokens(line + '\n');
        if (tokensUsed + sectionTokens + est > budget) break;
        sectionLines.push(line);
        sectionTokens += est;
      }
    }

    // 4b. Inferred / likely facts.
    if (inferred.length > 0) {
      if (sectionLines.length > 0) sectionLines.push('');
      const header = '[RECALLED CONTEXT \u2014 INFERRED] Likely facts based on conversation patterns:';
      sectionLines.push(header);
      sectionTokens += estimateTokens(header + '\n');

      for (const f of inferred) {
        const c = f.confidence ?? 0.5;
        const label = confidenceLabel(c);
        let prefix = '';
        if (label === 'inferred') {
          prefix = 'Possibly ';
        } else {
          prefix = 'May be ';
        }
        // Keep content starting lowercase if adding prefix.
        const contentLower = f.content.charAt(0).toLowerCase() + f.content.slice(1);
        let line = `- ${prefix}${contentLower} (confidence: ${c.toFixed(1)})`;
        if (f.verbatim) {
          const escaped = f.verbatim.replace(/"/g, '\\"');
          line += ` | User said: "${escaped}"`;
        }
        const est = estimateTokens(line + '\n');
        if (tokensUsed + sectionTokens + est > budget) break;
        sectionLines.push(line);
        sectionTokens += est;
      }
    }

    if (sectionLines.length > 0) {
      const text = sectionLines.join('\n');
      commitSection(text, 'episodic');
    }
  }

  // ── 5. [PROACTIVE] — Proactive surfacing ──────────────────────────
  {
    const items = input.proactiveItems ?? [];
    if (items.length > 0) {
      const lines: string[] = ['[PROACTIVE CONTEXT] Consider naturally weaving in:'];
      let sectionTokens = estimateTokens(lines[0] + '\n');

      // Sort by priority (highest first).
      const sorted = [...items].sort((a, b) => b.priority - a.priority);

      for (const item of sorted) {
        const line = `- [${item.type}] ${item.content}`;
        const est = estimateTokens(line + '\n');
        if (tokensUsed + sectionTokens + est > budget) break;
        lines.push(line);
        sectionTokens += est;
      }

      if (lines.length > 1) {
        const text = lines.join('\n');
        if (tierTexts.proactive) {
          // Open threads already wrote to proactive tier — merge.
          const oldText = tierTexts.proactive;
          const combined = oldText + '\n\n' + text;
          const oldTokens = estimateTokens(oldText);
          const newTokens = estimateTokens(combined);
          tokensUsed = tokensUsed - oldTokens + newTokens;
          tierTexts.proactive = combined;
          // Replace the open-threads entry in allParts.
          const idx = allParts.indexOf(oldText);
          if (idx !== -1) {
            allParts[idx] = combined;
          } else {
            allParts.push(combined);
          }
        } else {
          commitSection(text, 'proactive');
        }
      }
    }
  }

  // ── 6. [SESSION CONTEXT] — Working memory tier ────────────────────
  {
    const turnSummaries = input.turnSummaries ?? [];
    if (turnSummaries.length > 0) {
      const lines: string[] = ['[SESSION CONTEXT] Current conversation:'];
      let sectionTokens = estimateTokens(lines[0] + '\n');

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
      }

      if (lines.length > 1) {
        const text = lines.join('\n');
        commitSection(text, 'working');
      }
    }
  }

  // ── 7. [PAST CONVERSATIONS] — Summaries ───────────────────────────
  // Summaries use their own sub-budget but are also capped by remaining total budget.
  let summaryText = '';
  {
    if (input.summaries.length > 0) {
      let effectiveBudget = summaryBudget;
      const remaining = budget - tokensUsed;
      if (effectiveBudget > remaining) effectiveBudget = remaining;

      if (effectiveBudget > 0) {
        const lines: string[] = ['[PAST CONVERSATIONS] Previous sessions:'];
        let sectionTokens = estimateTokens(lines[0] + '\n');

        for (const s of input.summaries) {
          const dateStr = relativeDate(s.createdAt);
          const line = `- [${dateStr}] ${s.summary}`;
          const est = estimateTokens(line + '\n');
          if (sectionTokens + est > effectiveBudget) break;
          lines.push(line);
          sectionTokens += est;
        }

        if (lines.length > 1) {
          summaryText = lines.join('\n');
          tokensUsed += estimateTokens(summaryText);
          allParts.push(summaryText);
        }
      }
    }
  }

  const formatted = allParts.join('\n\n').replace(/\n+$/, '');

  return {
    semantic: tierTexts.semantic,
    emotional: tierTexts.emotional,
    episodic: tierTexts.episodic,
    proactive: tierTexts.proactive,
    working: tierTexts.working,
    totalTokens: tokensUsed,
    formatted,
  };
}
