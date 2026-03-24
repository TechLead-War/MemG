/**
 * Extraction pipeline: extract structured facts from conversation messages.
 * Matches the Go proxy/extract.go logic exactly.
 */

import { createHash, randomUUID } from 'crypto';
import type { Embedder } from './embedder.js';
import { cosineSimilarity, dimensionMatch } from './search.js';
import { defaultContentKey, type Store } from './store.js';
import type { Fact, FactFilter } from './types.js';

/** Trivial conversation patterns that should not trigger extraction. */
const TRIVIAL_PATTERNS = new Set([
  'thanks', 'thank you', 'ok', 'okay', 'got it', 'sure',
  'hello', 'hi', 'hey', 'bye', 'goodbye', 'good morning',
  'good night', 'good evening', 'good afternoon',
  'cool', 'great', 'awesome', 'nice',
  'understood', 'roger', 'ack', 'k', 'kk', 'lol', 'haha',
]);

/** Valid tags for facts. */
const VALID_TAGS = new Set([
  'skill', 'preference', 'relationship', 'medical',
  'location', 'work', 'hobby', 'personal', 'financial', 'other',
]);

/** Semantic dedup threshold — 0.92 cosine similarity. */
const SEMANTIC_DEDUP_THRESHOLD = 0.92;

/** Reinforcement count threshold for promotion to high significance. */
const PROMOTION_THRESHOLD = 5;

/** JSON shape returned by the extraction LLM. */
interface ExtractedFact {
  content: string;
  type: string;
  significance: number;
  tag: string;
  slot: string;
  reference_time: string;
  confidence: number | null;
}

/** Message shape for extraction. */
export interface ExtractionMessage {
  role: string;
  content: string;
}

/**
 * Check if a set of messages is trivial (greetings, acknowledgments).
 */
export function isTrivialTurn(messages: ExtractionMessage[]): boolean {
  if (messages.length === 0) return true;

  for (const msg of messages) {
    if (msg.role !== 'user') continue;

    let content = msg.content.toLowerCase().trim();
    content = content.replace(/[^a-z0-9 ]/g, '').trim();

    if (content === '') continue;

    if (!TRIVIAL_PATTERNS.has(content)) {
      return false;
    }
  }

  return true;
}

/**
 * Build the extraction prompt. Matches Go exactly.
 */
function buildExtractionPrompt(): string {
  const now = new Date();
  const today = now.toISOString().slice(0, 10);
  const yesterday = new Date(now.getTime() - 86400000).toISOString().slice(0, 10);

  return `You are a knowledge extraction engine. Today's date is ${today}.

Extract facts from this conversation. Return a JSON array. Each fact:
{
  "content": "the fact as a clear statement about the user",
  "type": "identity|event|pattern",
  "significance": 1-10,
  "tag": "category label",
  "slot": "semantic slot name (e.g. location, job, diet, name, email, relationship, preference)",
  "reference_time": "ISO date if time-bound, empty string if not",
  "confidence": 0.0-1.0
}

Rules:
- "identity" = enduring truths (preferences, attributes, relationships)
- "event" = things that happened at a specific time
- "pattern" = behavioral tendencies observed across the conversation
- "tag" = a category label. Use one of: skill, preference, relationship, medical, location, work, hobby, personal, financial, or other
- "slot" = the semantic slot this fact fills. Use: location, job, diet, name, email, relationship, preference, medical, hobby, skill, or other
- "reference_time" = ISO 8601 date (YYYY-MM-DD) for time-bound facts, empty string otherwise
- "confidence" = how confident you are (1.0 = explicitly stated by user, 0.5 = inferred, 0.0 = guessing)
- Resolve relative dates: "today" → "${today}", "yesterday" → "${yesterday}"
- Significance: 10 = life-critical (allergies, medical), 7-9 = important (job, location), 4-6 = moderate, 1-3 = trivial (lunch, weather)
- Skip greetings, filler, "thank you", and trivial exchanges
- If nothing is worth extracting, return []

Return ONLY the JSON array, no other text.`;
}

/**
 * Parse LLM response into extracted facts.
 * Resilient to markdown code blocks and surrounding text.
 */
function parseExtractionResponse(content: string): ExtractedFact[] {
  content = content.trim();

  // Try direct parse.
  try {
    return JSON.parse(content) as ExtractedFact[];
  } catch { /* continue */ }

  // Try stripping code fences.
  const stripped = stripCodeFences(content);
  if (stripped !== content) {
    try {
      return JSON.parse(stripped) as ExtractedFact[];
    } catch { /* continue */ }
  }

  // Try finding JSON array within the text.
  const start = content.indexOf('[');
  const end = content.lastIndexOf(']');
  if (start >= 0 && end > start) {
    const candidate = content.slice(start, end + 1);
    try {
      return JSON.parse(candidate) as ExtractedFact[];
    } catch { /* continue */ }
  }

  throw new Error(`Could not parse JSON array from response: ${content.slice(0, 100)}`);
}

/**
 * Strip markdown code block wrappers.
 */
function stripCodeFences(s: string): string {
  s = s.trim();

  if (s.startsWith('```')) {
    const idx = s.indexOf('\n');
    if (idx >= 0) {
      s = s.slice(idx + 1);
    } else {
      s = s.replace(/^```json/, '').replace(/^```/, '');
    }
  }

  if (s.endsWith('```')) {
    s = s.slice(0, -3);
  }

  return s.trim();
}

/**
 * Validate and filter extracted facts.
 */
function validateExtraction(facts: ExtractedFact[]): ExtractedFact[] {
  const valid: ExtractedFact[] = [];

  for (const f of facts) {
    const content = (f.content ?? '').trim();
    if (content === '') continue;
    if (content.length > 500) continue;

    if (f.reference_time) {
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(f.reference_time)) {
        f.reference_time = '';
      } else {
        const parsed = new Date(f.reference_time);
        if (isNaN(parsed.getTime())) {
          f.reference_time = '';
        }
      }
    }

    if (f.confidence !== null && f.confidence !== undefined) {
      if (f.confidence < 0) f.confidence = 0;
      if (f.confidence > 1) f.confidence = 1;
    }

    let tag = (f.tag ?? '').toLowerCase().trim();
    if (tag && !VALID_TAGS.has(tag)) {
      tag = 'other';
    }

    valid.push({
      ...f,
      content,
      tag,
    });
  }

  return valid;
}

/**
 * Resolve fact type string to a valid type.
 */
function resolveFactType(t: string): 'identity' | 'event' | 'pattern' {
  switch ((t ?? '').toLowerCase().trim()) {
    case 'event':
      return 'event';
    case 'pattern':
      return 'pattern';
    case 'identity':
    default:
      return 'identity';
  }
}

/**
 * Clamp significance to [1, 10], defaulting to 5.
 */
function clampSignificance(v: number): number {
  if (!v || v < 1 || v > 10) return 5;
  return v;
}

/**
 * Get confidence value, defaulting to 0.8 if null.
 */
function confidenceValue(v: number | null | undefined): number {
  if (v === null || v === undefined) return 0.8;
  return v;
}

/**
 * Compute TTL expiry based on significance. Matches Go TTLForSignificance.
 */
function ttlForSignificance(sig: number): string | null {
  if (sig >= 10) return null; // never expires
  const now = Date.now();
  if (sig >= 5) {
    return new Date(now + 30 * 24 * 60 * 60 * 1000).toISOString(); // ~1 month
  }
  return new Date(now + 7 * 24 * 60 * 60 * 1000).toISOString(); // ~1 week
}

/**
 * Call an LLM via fetch for extraction. Supports OpenAI-compatible APIs.
 */
export async function callLLM(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userContent: string,
  provider: string
): Promise<string> {
  let url: string;
  let headers: Record<string, string>;
  let body: string;

  if (provider === 'gemini') {
    url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    headers = { 'Content-Type': 'application/json' };
    body = JSON.stringify({
      systemInstruction: { parts: [{ text: systemPrompt }] },
      contents: [{ role: 'user', parts: [{ text: userContent }] }],
      generationConfig: { temperature: 0.1 },
    });
  } else if (provider === 'anthropic') {
    url = 'https://api.anthropic.com/v1/messages';
    headers = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    };
    body = JSON.stringify({
      model,
      max_tokens: 2048,
      system: systemPrompt,
      messages: [{ role: 'user', content: userContent }],
    });
  } else {
    url = 'https://api.openai.com/v1/chat/completions';
    headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    };
    body = JSON.stringify({
      model,
      max_tokens: 2048,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent },
      ],
    });
  }

  let lastErr: Error | null = null;
  const maxRetries = 3;
  let backoff = 1000;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(url, { method: 'POST', headers, body });

      if (!resp.ok) {
        const respBody = await resp.text();
        const errMsg = `LLM API error: HTTP ${resp.status} ${respBody}`;
        lastErr = new Error(errMsg);

        const retryable =
          resp.status === 429 ||
          resp.status >= 500;

        if (!retryable || attempt === maxRetries) throw lastErr;

        await new Promise((r) => setTimeout(r, backoff));
        backoff *= 2;
        continue;
      }

      const data = await resp.json() as any;

      if (provider === 'gemini') {
        return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
      }

      if (provider === 'anthropic') {
        const blocks = data.content as any[];
        const textBlock = blocks?.find((b: any) => b.type === 'text');
        return textBlock?.text ?? '';
      }

      return data.choices?.[0]?.message?.content ?? '';
    } catch (err: any) {
      lastErr = err;
      if (attempt === maxRetries) throw lastErr;

      await new Promise((r) => setTimeout(r, backoff));
      backoff *= 2;
    }
  }

  throw lastErr ?? new Error('LLM call failed');
}

/**
 * Find the most semantically similar existing fact above the dedup threshold.
 */
function findSemanticMatch(
  embedding: number[],
  existingFacts: Fact[]
): Fact | null {
  let best: Fact | null = null;
  let bestScore = 0;

  for (const f of existingFacts) {
    if (!f.embedding || f.embedding.length === 0) continue;
    const score = cosineSimilarity(embedding, f.embedding);
    if (score > bestScore) {
      bestScore = score;
      best = f;
    }
  }

  if (bestScore >= SEMANTIC_DEDUP_THRESHOLD) return best;
  return null;
}

/**
 * Find existing current identity facts that occupy the same slot.
 */
function findSlotConflicts(slot: string, existing: Fact[]): Fact[] {
  if (!slot) return [];
  return existing.filter(
    (f) =>
      f.slot === slot &&
      f.temporalStatus === 'current' &&
      f.factType === 'identity'
  );
}

/**
 * Run the full extraction pipeline:
 * 1. Trivial turn detection
 * 2. LLM extraction call
 * 3. Parse and validate
 * 4. Embed extracted facts
 * 5. Dedup (content key + semantic)
 * 6. Slot conflict resolution
 * 7. TTL assignment
 * 8. Persist
 */
export async function runExtraction(
  store: Store,
  embedder: Embedder | null,
  messages: ExtractionMessage[],
  entityUuid: string,
  apiKey: string,
  llmModel: string,
  llmProvider: string
): Promise<{ inserted: number; reinforced: number }> {
  if (messages.length === 0) return { inserted: 0, reinforced: 0 };
  if (isTrivialTurn(messages)) return { inserted: 0, reinforced: 0 };

  const userMessages = messages.filter((m) => m.role === 'user');
  if (userMessages.length === 0) return { inserted: 0, reinforced: 0 };

  // Build transcript from user messages only.
  const transcript = userMessages.map((m) => `${m.role}: ${m.content}`).join('\n');

  const systemPrompt = buildExtractionPrompt();

  let responseContent: string;
  try {
    responseContent = await callLLM(apiKey, llmModel, systemPrompt, transcript, llmProvider);
  } catch (err) {
    console.warn('[memg] extraction LLM call failed:', err);
    return { inserted: 0, reinforced: 0 };
  }

  let extracted: ExtractedFact[];
  try {
    extracted = parseExtractionResponse(responseContent);
  } catch (err) {
    console.warn('[memg] extraction parse failed:', err);
    return { inserted: 0, reinforced: 0 };
  }

  extracted = validateExtraction(extracted);
  if (extracted.length === 0) return { inserted: 0, reinforced: 0 };

  // Build facts.
  const embeddingModelName = embedder?.modelName() ?? '';
  const facts: Fact[] = [];
  const contents: string[] = [];

  for (const ef of extracted) {
    const factType = resolveFactType(ef.type);
    const significance = clampSignificance(ef.significance);

    const fact: Fact = {
      uuid: randomUUID(),
      content: ef.content,
      factType,
      temporalStatus: 'current',
      significance,
      contentKey: defaultContentKey(ef.content),
      tag: (ef.tag ?? '').toLowerCase().trim(),
      slot: (ef.slot ?? '').toLowerCase().trim(),
      confidence: confidenceValue(ef.confidence),
      embeddingModel: embeddingModelName,
      sourceRole: 'user',
      reinforcedCount: 0,
      recallCount: 0,
      expiresAt: ttlForSignificance(significance) ?? undefined,
      referenceTime: undefined,
    };

    if (ef.reference_time) {
      const parsed = new Date(ef.reference_time);
      if (!isNaN(parsed.getTime())) {
        fact.referenceTime = ef.reference_time;
      }
    }

    facts.push(fact);
    contents.push(ef.content);
  }

  // Embed all fact contents in a single batch.
  if (embedder) {
    try {
      const embeddings = await embedder.embed(contents);
      for (let i = 0; i < embeddings.length && i < facts.length; i++) {
        facts[i].embedding = embeddings[i];
      }
    } catch (err) {
      console.warn('[memg] extraction embed failed, continuing without embeddings:', err);
    }
  }

  // Load existing facts for dedup and conflict detection.
  const existingFacts = await store.listFactsForRecall(
    entityUuid,
    { statuses: ['current'], excludeExpired: true },
    0
  );

  let inserted = 0;
  let reinforced = 0;

  for (const fact of facts) {
    // Step 1: Exact dedup by content key.
    const existing = await store.findFactByKey(entityUuid, fact.contentKey);
    if (existing) {
      await reinforceAndPromote(store, existing, fact.expiresAt ?? null);
      reinforced++;
      continue;
    }

    // Step 2: Slot conflict resolution.
    if (
      fact.slot &&
      fact.factType === 'identity' &&
      fact.temporalStatus === 'current'
    ) {
      const conflicts = findSlotConflicts(fact.slot, existingFacts);
      for (const conflict of conflicts) {
        await store.updateTemporalStatus(conflict.uuid, 'historical');
        conflict.temporalStatus = 'historical';
      }
    }

    // Step 3: Semantic dedup.
    if (fact.embedding && fact.embedding.length > 0 && existingFacts.length > 0) {
      const match = findSemanticMatch(fact.embedding, existingFacts);
      if (match) {
        await reinforceAndPromote(store, match, fact.expiresAt ?? null);
        reinforced++;
        continue;
      }
    }

    // Step 4: Insert the new fact.
    await store.insertFact(entityUuid, fact);
    inserted++;
  }

  return { inserted, reinforced };
}

/**
 * Reinforce an existing fact and promote to high significance if warranted.
 */
async function reinforceAndPromote(
  store: Store,
  existing: Fact,
  newExpiresAt: string | null
): Promise<void> {
  const newCount = existing.reinforcedCount + 1;
  const shouldPromote =
    newCount >= PROMOTION_THRESHOLD && existing.significance < 10;

  let expiresAt = newExpiresAt;
  if (shouldPromote) {
    expiresAt = null; // high-significance facts never expire
  }

  await store.reinforceFact(existing.uuid, expiresAt);

  if (shouldPromote) {
    await store.updateSignificance(existing.uuid, 10);
  }
}
