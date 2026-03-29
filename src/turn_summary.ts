/**
 * Turn summary maintenance: generates immutable turn-range summaries for
 * messages that have fallen off the working memory window, and consolidates
 * old summaries when count exceeds 3.
 * Port of Go memory/turnsummary.go.
 */

import type { TurnSummary } from './types.js';
import type { Store } from './store.js';
import type { Embedder } from './embedder.js';

const turnSummaryPrompt =
  'Summarize these conversation turns. Focus on: decisions made, questions asked, code discussed, data referenced, open items. Keep under 200 tokens.\n\n';

const overviewConsolidatePrompt =
  'Consolidate these summaries into one overview. Keep under 300 tokens.\n\n';

export async function maintainTurnSummaries(
  store: Store,
  embedder: Embedder,
  llmChat: (prompt: string) => Promise<string>,
  conversationId: string,
  entityId: string,
  messages: Array<{ role: string; content: string }>,
  workingMemoryTurns: number
): Promise<void> {
  if (messages.length <= workingMemoryTurns) return;

  const existing = await store.listTurnSummaries(conversationId);

  let highestEnd = 0;
  for (const ts of existing) {
    if (!ts.isOverview && ts.endTurn > highestEnd) {
      highestEnd = ts.endTurn;
    }
  }

  const newEnd = messages.length - workingMemoryTurns;
  if (newEnd <= highestEnd) return;

  const toSummarize = messages.slice(highestEnd, newEnd);
  if (toSummarize.length === 0) return;

  let transcript = '';
  for (const m of toSummarize) {
    transcript += m.role + ': ' + m.content + '\n';
  }

  const summary = (await llmChat(turnSummaryPrompt + transcript)).trim();
  if (!summary) return;

  const vectors = await embedder.embed([summary]);
  if (vectors.length === 0) return;

  const ts: TurnSummary = {
    uuid: '',
    conversationId,
    entityId,
    startTurn: highestEnd + 1,
    endTurn: newEnd,
    summary,
    summaryEmbedding: vectors[0],
    isOverview: false,
    createdAt: new Date().toISOString(),
  };
  await store.insertTurnSummary(ts);

  await consolidateTurnSummaries(store, embedder, llmChat, conversationId, entityId);
}

async function consolidateTurnSummaries(
  store: Store,
  embedder: Embedder,
  llmChat: (prompt: string) => Promise<string>,
  conversationId: string,
  entityId: string
): Promise<void> {
  const all = await store.listTurnSummaries(conversationId);

  const nonOverview: TurnSummary[] = [];
  let overview: TurnSummary | null = null;
  for (const ts of all) {
    if (ts.isOverview) {
      overview = ts;
    } else {
      nonOverview.push(ts);
    }
  }

  if (nonOverview.length <= 3) return;

  nonOverview.sort((a, b) => a.startTurn - b.startTurn);
  const oldest = nonOverview.slice(0, 2);

  let combined = '';
  if (overview) {
    combined += overview.summary + '\n';
  }
  for (const s of oldest) {
    combined += s.summary + '\n';
  }

  const overviewText = (await llmChat(overviewConsolidatePrompt + combined)).trim();
  if (!overviewText) return;

  const vectors = await embedder.embed([overviewText]);
  if (vectors.length === 0) return;

  const newOverview: TurnSummary = {
    uuid: '',
    conversationId,
    entityId,
    startTurn: 1,
    endTurn: oldest[oldest.length - 1].endTurn,
    summary: overviewText,
    summaryEmbedding: vectors[0],
    isOverview: true,
    createdAt: new Date().toISOString(),
  };
  await store.insertTurnSummary(newOverview);

  const toDelete: string[] = [];
  if (overview) toDelete.push(overview.uuid);
  for (const s of oldest) {
    toDelete.push(s.uuid);
  }
  await store.deleteTurnSummaries(conversationId, toDelete);
}
