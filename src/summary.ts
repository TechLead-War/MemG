/**
 * Conversation summary generation and storage.
 * Matches the Go memory/summary.go logic.
 */

import type { Embedder } from './embedder.js';
import type { Store } from './store.js';

const SUMMARY_PROMPT = `Summarize this conversation. Focus on:
- What was discussed
- What decisions were made
- What is still pending or unresolved
- Any new information learned about the user

Be concise — 2-5 sentences. Only include what is meaningful and worth remembering.
If the conversation contains no meaningful content worth remembering (e.g. just greetings or trivial exchanges), respond with exactly: NONE`;

/**
 * Generate a summary for a conversation, embed it, and store both on the
 * conversation record. If the conversation is trivial (the LLM returns
 * "NONE"), no summary is stored.
 */
export async function generateAndStoreSummary(
  store: Store,
  embedder: Embedder,
  llmChat: (prompt: string) => Promise<string>,
  conversationUuid: string
): Promise<void> {
  const messages = await store.readMessages(conversationUuid);
  if (!messages || messages.length === 0) return;

  const transcript = messages
    .map((m) => `${m.role}: ${m.content}`)
    .join('\n');

  const fullPrompt = SUMMARY_PROMPT + '\n\n' + transcript;

  let summary: string;
  try {
    summary = await llmChat(fullPrompt);
  } catch (err) {
    console.warn('[memg] summary: LLM call failed:', err);
    return;
  }

  summary = summary.trim();
  if (!summary || summary.toUpperCase() === 'NONE') return;

  let embedding: number[] = [];
  try {
    const [vec] = await embedder.embed([summary]);
    embedding = vec;
  } catch (err) {
    console.warn('[memg] summary: embed failed, proceeding without embedding:', err);
  }

  const modelName = embedder.modelName();
  await store.updateConversationSummary(conversationUuid, summary, embedding, modelName);
}
