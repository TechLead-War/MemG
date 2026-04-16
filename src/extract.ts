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

/** Valid emotional valence categories. */
const VALID_EMOTIONAL_VALENCES = new Set([
  'grief', 'joy', 'anxiety', 'hope', 'love', 'anger', 'fear', 'pride', 'neutral',
]);

/** Semantic dedup threshold — higher = keep more distinct facts. */
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
  emotional_weight: number | null;
  emotional_valence: string | null;
  verbatim: string | null;
  started_at: string | null;
  thread_status: string | null;
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
function buildExtractionPrompt(sessionDate?: string): string {
  let now: Date;
  if (sessionDate) {
    const parsed = new Date(sessionDate);
    now = isNaN(parsed.getTime()) ? new Date() : parsed;
  } else {
    now = new Date();
  }
  const today = now.toISOString().slice(0, 10);
  const yesterday = new Date(now.getTime() - 86400000).toISOString().slice(0, 10);

  return `You are an exhaustive knowledge extraction engine. Today's date is ${today}.

Extract EVERY fact from this conversation. Be thorough — extract even minor details. Return a JSON array. Each fact:
{
  "content": "the fact as a clear statement (MUST include the person's name)",
  "type": "identity|event|pattern",
  "significance": 1-10,
  "tag": "skill|preference|relationship|medical|location|work|hobby|personal|financial|other",
  "slot": "semantic slot name",
  "reference_time": "ISO date (YYYY-MM-DD) if time-bound, empty string if not",
  "confidence": 0.0-1.0,
  "emotional_weight": 0.0-1.0 or null,
  "emotional_valence": "grief|joy|anxiety|hope|love|anger|fear|pride|neutral or null",
  "verbatim": "speaker's exact words or null",
  "started_at": "ISO date or null",
  "thread_status": "open or null"
}

CRITICAL extraction rules:
- Extract facts about ALL speakers by name. Always include the person's name in the content.
- Extract EVERY specific detail mentioned: names of people, pets, places, restaurants, bands, books, movies, foods, activities, hobbies, classes, jobs, events, items, gifts, vehicles, etc.
- For events: always include WHO did WHAT, WHERE, WHEN, and WITH WHOM if mentioned.
- For preferences: "X likes Y", "X enjoys Y", "X's favorite Y is Z"
- For activities: "X did Y", "X started Y", "X went to Y", "X attended Y"
- For temporal facts: resolve relative dates. "today" → "${today}", "yesterday" → "${yesterday}", "last week" → compute date, "next month" → compute date
- For advice/opinions: "X told Y to do Z", "X thinks Y about Z"
- Significance: 10=life-critical, 7-9=important, 4-6=moderate, 1-3=minor details (specific items, food, small activities — STILL EXTRACT THESE)
- confidence: 1.0=explicitly stated, 0.5=inferred, 0.0=guessing
- DO NOT skip details just because they seem minor. A mention of a specific band, food, painting subject, car model, or game name is worth extracting.
- DO NOT summarize multiple facts into one. Each distinct piece of information should be a separate fact.
- If nothing factual is discussed, return []

Return ONLY the JSON array.`;
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
  } catch (err) { console.warn('[memg] extract: direct JSON parse failed:', err); }

  // Try stripping code fences.
  const stripped = stripCodeFences(content);
  if (stripped !== content) {
    try {
      return JSON.parse(stripped) as ExtractedFact[];
    } catch (err) { console.warn('[memg] extract: code-fence JSON parse failed:', err); }
  }

  // Try finding JSON array within the text.
  const start = content.indexOf('[');
  const end = content.lastIndexOf(']');
  if (start >= 0 && end > start) {
    const candidate = content.slice(start, end + 1);
    try {
      return JSON.parse(candidate) as ExtractedFact[];
    } catch (err) { console.warn('[memg] extract: array-slice JSON parse failed:', err); }
  }

  // Salvage: try to recover complete JSON objects from a truncated array.
  const arrStart = content.indexOf('[');
  if (arrStart >= 0) {
    const inner = content.slice(arrStart + 1);
    const objectMatches = inner.match(/\{[^{}]*\}/g);
    if (objectMatches) {
      const recovered: ExtractedFact[] = [];
      for (const raw of objectMatches) {
        try {
          recovered.push(JSON.parse(raw) as ExtractedFact);
        } catch (err) { console.warn('[memg] extract: skipping malformed object:', err); }
      }
      if (recovered.length > 0) {
        return recovered;
      }
    }
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
    if (content.length > 1000) continue;

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

    // Validate emotional_weight: clamp to [0, 1], default null.
    let emotionalWeight: number | null = f.emotional_weight ?? null;
    if (emotionalWeight !== null) {
      if (typeof emotionalWeight !== 'number' || isNaN(emotionalWeight)) {
        emotionalWeight = null;
      } else {
        if (emotionalWeight < 0) emotionalWeight = 0;
        if (emotionalWeight > 1) emotionalWeight = 1;
      }
    }

    // Validate emotional_valence: must be in allowed set, default null.
    let emotionalValence: string | null = f.emotional_valence ?? null;
    if (emotionalValence !== null) {
      emotionalValence = emotionalValence.toLowerCase().trim();
      if (!VALID_EMOTIONAL_VALENCES.has(emotionalValence)) {
        emotionalValence = null;
      }
    }

    // Validate verbatim: trim, max 300 chars, default null.
    let verbatim: string | null = f.verbatim ?? null;
    if (verbatim !== null) {
      verbatim = verbatim.trim();
      if (verbatim === '') {
        verbatim = null;
      } else if (verbatim.length > 300) {
        verbatim = verbatim.slice(0, 300);
      }
    }

    // Validate started_at: ISO date format, not in the future, default null.
    let startedAt: string | null = f.started_at ?? null;
    if (startedAt !== null) {
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(startedAt)) {
        startedAt = null;
      } else {
        const parsed = new Date(startedAt);
        if (isNaN(parsed.getTime()) || parsed.getFullYear() < 1900 || parsed > new Date()) {
          startedAt = null;
        }
      }
    }

    // Validate thread_status: 'open' or null.
    let threadStatus: string | null = f.thread_status ?? null;
    if (threadStatus !== null) {
      threadStatus = threadStatus.toLowerCase().trim();
      if (threadStatus !== 'open') {
        threadStatus = null;
      }
    }

    valid.push({
      ...f,
      content,
      tag,
      emotional_weight: emotionalWeight,
      emotional_valence: emotionalValence,
      verbatim,
      started_at: startedAt,
      thread_status: threadStatus,
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
 * Get confidence value, defaulting to 0.5 (neutral) if null/undefined.
 */
function confidenceValue(v: number | null | undefined): number {
  if (v === null || v === undefined) return 0.5;
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
 * Resolve the chat completions URL for OpenAI-compatible providers.
 */
function resolveOpenAICompatibleUrl(provider: string): string {
  switch (provider) {
    case 'deepseek': return 'https://api.deepseek.com/v1/chat/completions';
    case 'groq': return 'https://api.groq.com/openai/v1/chat/completions';
    case 'togetherai': return 'https://api.together.xyz/v1/chat/completions';
    case 'xai': return 'https://api.x.ai/v1/chat/completions';
    case 'ollama': return `${process.env.OLLAMA_BASE_URL || 'http://localhost:11434'}/v1/chat/completions`;
    default: return 'https://api.openai.com/v1/chat/completions';
  }
}

/**
 * Call an LLM via fetch for extraction. Supports all 10 providers.
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
      generationConfig: { temperature: 0.1, maxOutputTokens: 8192 },
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
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: 'user', content: userContent }],
    });
  } else if (provider === 'azureopenai') {
    const endpoint = process.env.AZURE_OPENAI_ENDPOINT || '';
    const apiVersion = process.env.AZURE_OPENAI_API_VERSION || '2024-10-21';
    url = `${endpoint}/openai/deployments/${model}/chat/completions?api-version=${apiVersion}`;
    headers = {
      'Content-Type': 'application/json',
      'api-key': apiKey,
    };
    body = JSON.stringify({
      max_tokens: 4096,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent },
      ],
    });
  } else if (provider === 'bedrock') {
    return await callBedrockLLM(model, systemPrompt, userContent);
  } else {
    // OpenAI and all OpenAI-compatible providers (deepseek, groq, togetherai, xai, ollama).
    url = resolveOpenAICompatibleUrl(provider);
    headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    };
    body = JSON.stringify({
      model,
      max_tokens: 4096,
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
 * Call AWS Bedrock Converse API for extraction.
 * Requires @aws-sdk/client-bedrock-runtime as optional peer dependency.
 */
async function callBedrockLLM(
  model: string,
  systemPrompt: string,
  userContent: string
): Promise<string> {
  let bedrock: any;
  try {
    const mod = '@aws-sdk/client-bedrock-runtime';
    bedrock = await import(mod);
  } catch (err) {
    console.warn('[memg] extract: bedrock SDK import failed:', err);
    throw new Error(
      'Bedrock LLM provider requires @aws-sdk/client-bedrock-runtime. Install it: npm i @aws-sdk/client-bedrock-runtime'
    );
  }

  const region = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || 'us-east-1';
  const client = new bedrock.BedrockRuntimeClient({ region });
  const command = new bedrock.ConverseCommand({
    modelId: model,
    system: [{ text: systemPrompt }],
    messages: [{ role: 'user', content: [{ text: userContent }] }],
    inferenceConfig: { maxTokens: 4096 },
  });

  const response = await client.send(command);
  return response.output?.message?.content?.[0]?.text ?? '';
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
  llmProvider: string,
  sessionDate?: string
): Promise<{ inserted: number; reinforced: number }> {
  if (messages.length === 0) return { inserted: 0, reinforced: 0 };
  if (isTrivialTurn(messages)) return { inserted: 0, reinforced: 0 };

  // Build transcript from ALL messages (both speakers share facts).
  const transcript = messages.map((m) => `${m.role}: ${m.content}`).join('\n');

  const systemPrompt = buildExtractionPrompt(sessionDate);

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
      sourceRole: 'mixed',
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

    if (ef.emotional_weight !== null && ef.emotional_weight !== undefined) {
      fact.emotionalWeight = ef.emotional_weight;
    }
    if (ef.emotional_valence !== null && ef.emotional_valence !== undefined) {
      fact.emotionalValence = ef.emotional_valence as Fact['emotionalValence'];
    }
    if (ef.verbatim !== null && ef.verbatim !== undefined) {
      fact.verbatim = ef.verbatim;
    }
    if (ef.started_at !== null && ef.started_at !== undefined) {
      fact.startedAt = ef.started_at;
    }
    fact.threadStatus = (ef.thread_status as Fact['threadStatus']) ?? null;

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
        await store.updateTemporalStatus(conflict.uuid, 'historical', fact.uuid);
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

/**
 * Run extraction over pre-segmented conversation chunks.
 *
 * Each segment carries its own messages plus optional topic/classification
 * metadata. Trivial segments are skipped. For non-trivial segments the
 * topic and classification are prepended to the transcript so the LLM has
 * extra context, then runExtraction handles the rest.
 */
export async function runSegmentedExtraction(
  store: Store,
  embedder: Embedder | null,
  segments: Array<{ messages: Array<{ role: string; content: string }>; topic?: string; classification?: string }>,
  entityUuid: string,
  apiKey: string,
  llmModel: string,
  llmProvider: string,
  sessionDate?: string
): Promise<{ inserted: number; reinforced: number }> {
  let totalInserted = 0;
  let totalReinforced = 0;

  for (const segment of segments) {
    if (!segment.messages || segment.messages.length === 0) continue;
    if (isTrivialTurn(segment.messages)) continue;

    // Prepend topic and classification as context lines.
    const contextLines: string[] = [];
    if (segment.topic) {
      contextLines.push(`Topic: ${segment.topic}`);
    }
    if (segment.classification) {
      contextLines.push(`Classification: ${segment.classification}`);
    }

    let messages = segment.messages;
    if (contextLines.length > 0) {
      const prefix = contextLines.join('\n');
      // Prepend context to the first message's content.
      messages = messages.map((m, i) => {
        if (i === 0) {
          return { ...m, content: `${prefix}\n${m.content}` };
        }
        return m;
      });
    }

    const result = await runExtraction(
      store,
      embedder,
      messages,
      entityUuid,
      apiKey,
      llmModel,
      llmProvider,
      sessionDate
    );

    totalInserted += result.inserted;
    totalReinforced += result.reinforced;
  }

  return { inserted: totalInserted, reinforced: totalReinforced };
}
