/**
 * LoCoMo benchmark runner for MemG.
 *
 * Pipeline per conversation:
 * 1. Create a fresh MemG instance (isolated SQLite DB)
 * 2. Ingest conversation session-by-session using extractFromMessages
 * 3. For each QA pair: recall context via MemG → ask LLM → score
 * 4. Aggregate scores
 */

import { readFileSync, existsSync, writeFileSync, unlinkSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';
import type {
  LoCoMoEntry,
  LoCoMoTurn,
  QuestionResult,
  CategoryScore,
  ConversationResult,
  BenchmarkRun,
  BenchmarkConfig,
  BenchmarkProgress,
  OverallScore,
} from './types.ts';
import { CATEGORY_NAMES } from './types.ts';
import { tokenF1, exactMatch, isRefusal, llmJudge, callLLM } from './evaluator.ts';
import { buildQuestionDiagnostics, summarizeDiagnostics } from './diagnostics.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Parse LoCoMo date format "4:04 pm on 20 January, 2023" to ISO date string "2023-01-20".
 */
function parseLoCoMoDate(dateStr: string): string {
  if (!dateStr) return '';
  // Try direct parse first
  const direct = new Date(dateStr);
  if (!isNaN(direct.getTime())) return direct.toISOString().slice(0, 10);

  // Parse "H:MM am/pm on DD Month, YYYY" format
  const match = dateStr.match(/(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})/i);
  if (match) {
    const [, , , , day, month, year] = match;
    const monthNames: Record<string, string> = {
      january: '01', february: '02', march: '03', april: '04',
      may: '05', june: '06', july: '07', august: '08',
      september: '09', october: '10', november: '11', december: '12',
    };
    const m = monthNames[month.toLowerCase()];
    if (m) {
      return `${year}-${m}-${day.padStart(2, '0')}`;
    }
  }

  return '';
}

// ── Dataset loading ──

export function loadDataset(path: string): LoCoMoEntry[] {
  const raw = readFileSync(path, 'utf8');
  return JSON.parse(raw);
}

function getConversationSessions(conv: any): { sessionKey: string; date: string; turns: LoCoMoTurn[] }[] {
  const sessions: { sessionKey: string; date: string; turns: LoCoMoTurn[] }[] = [];
  const sessionKeys = Object.keys(conv)
    .filter((k) => /^session_\d+$/.test(k) && Array.isArray(conv[k]))
    .sort((a, b) => {
      const na = parseInt(a.split('_')[1]);
      const nb = parseInt(b.split('_')[1]);
      return na - nb;
    });

  for (const key of sessionKeys) {
    const dateKey = `${key}_date_time`;
    sessions.push({
      sessionKey: key,
      date: conv[dateKey] ?? '',
      turns: conv[key] as LoCoMoTurn[],
    });
  }

  return sessions;
}

// ── Build answer prompt ──

function buildAnswerPrompt(
  question: string,
  memoryContext: string,
  speakerA: string,
  speakerB: string
): string {
  return `You answer questions about conversations between ${speakerA} and ${speakerB} using ONLY the memories below.

## Memories
${memoryContext || '(No memories found)'}

## Instructions
- "the user" or "User" in memories = ${speakerA}. The other speaker = ${speakerB}.
- Answer using ONLY information from the memories. Do not add outside knowledge.
- Use the EXACT words and phrases from the memories. Do not paraphrase or rephrase.
- For list questions ("what things", "which places", "who"): list ALL matching items from memories as a comma-separated list. Nothing else.
- For "when" questions: give the specific date or time period mentioned in the memories. Just the date, nothing else.
- For yes/no questions: answer "yes" or "no" followed by brief evidence.
- Keep answers as SHORT as possible. Ideally 1-10 words. No sentences, no explanations.
- If a memory says "[historical]" it still happened — include it.
- If the answer is genuinely not in the memories at all, say "This information is not available in the conversation."

Question: ${question}
Answer:`;
}

// ── Main runner ──

export type ProgressCallback = (progress: BenchmarkProgress, runId: string) => void;

let _cancelledRuns = new Set<string>();

export function cancelRun(runId: string) {
  _cancelledRuns.add(runId);
}

export async function runBenchmark(
  config: BenchmarkConfig,
  onProgress?: ProgressCallback
): Promise<BenchmarkRun> {
  const runId = randomUUID().slice(0, 8);
  const dataPath = join(__dirname, 'data', 'locomo10.json');

  if (!existsSync(dataPath)) {
    throw new Error(`Dataset not found at ${dataPath}. Run download first.`);
  }

  const dataset = loadDataset(dataPath);

  // Determine which conversations to run.
  const convIndices =
    config.conversations.length > 0
      ? config.conversations.filter((i) => i >= 0 && i < dataset.length)
      : dataset.map((_, i) => i);

  // Count total questions and total sessions.
  let totalQuestions = 0;
  let totalSessions = 0;
  for (const ci of convIndices) {
    const qas = dataset[ci].qa.filter(
      (q) => config.categories.length === 0 || config.categories.includes(q.category)
    );
    totalQuestions += qas.length;
    const sessKeys = Object.keys(dataset[ci].conversation).filter((k) => /^session_\d+$/.test(k) && Array.isArray(dataset[ci].conversation[k]));
    totalSessions += sessKeys.length;
  }

  const progress: BenchmarkProgress = {
    totalConversations: convIndices.length,
    completedConversations: 0,
    totalQuestions,
    completedQuestions: 0,
    totalSessions,
    completedSessions: 0,
  };

  const run: BenchmarkRun = {
    id: runId,
    startedAt: new Date().toISOString(),
    status: 'running',
    config: { ...config, apiKey: undefined },
    conversations: [],
    overall: emptyOverall(),
    progress,
  };

  onProgress?.(progress, runId);

  const concurrency = Math.max(1, config.concurrency ?? 1);

  try {
    if (concurrency <= 1) {
      // Sequential mode (original behavior).
      for (const ci of convIndices) {
        if (_cancelledRuns.has(runId)) {
          run.status = 'cancelled';
          break;
        }

        progress.currentConversation = ci;
        progress.currentPhase = 'ingesting';
        onProgress?.(progress, runId);

        const result = await runConversation(dataset[ci], ci, config, run, (completedQs) => {
          progress.completedQuestions += completedQs;
          onProgress?.(progress, runId);
        }, () => {
          progress.completedSessions = (progress.completedSessions ?? 0) + 1;
          onProgress?.(progress, runId);
        });

        run.conversations.push(result);
        progress.completedConversations++;
        onProgress?.(progress, runId);
      }
    } else {
      // Parallel mode with concurrency limit.
      const results = new Map<number, ConversationResult>();
      let nextIdx = 0;

      async function worker() {
        while (true) {
          if (_cancelledRuns.has(runId)) return;
          const workIdx = nextIdx++;
          if (workIdx >= convIndices.length) return;
          const ci = convIndices[workIdx];

          const result = await runConversation(dataset[ci], ci, config, run, (completedQs) => {
            progress.completedQuestions += completedQs;
            onProgress?.(progress, runId);
          }, () => {
            progress.completedSessions = (progress.completedSessions ?? 0) + 1;
            onProgress?.(progress, runId);
          });

          results.set(ci, result);
          progress.completedConversations++;
          onProgress?.(progress, runId);
        }
      }

      const workers = Array.from({ length: Math.min(concurrency, convIndices.length) }, () => worker());
      await Promise.all(workers);

      // Preserve original conversation order in results.
      for (const ci of convIndices) {
        const r = results.get(ci);
        if (r) run.conversations.push(r);
      }
    }

    if (run.status !== 'cancelled') {
      run.status = 'completed';
    }
  } catch (err: any) {
    run.status = 'failed';
    run.error = err.message;
  }

  run.completedAt = new Date().toISOString();
  run.overall = computeOverall(run.conversations);

  // Save results.
  const resultsDir = join(__dirname, 'results');
  writeFileSync(
    join(resultsDir, `run_${runId}.json`),
    JSON.stringify(run, null, 2)
  );

  _cancelledRuns.delete(runId);
  return run;
}

// ── Per-conversation runner ──

async function runConversation(
  entry: LoCoMoEntry,
  convIdx: number,
  config: BenchmarkConfig,
  run: BenchmarkRun,
  onQuestionBatch: (count: number) => void,
  onSessionIngested: () => void
): Promise<ConversationResult> {
  const conv = entry.conversation;
  const speakerA = conv.speaker_a;
  const speakerB = conv.speaker_b;
  const sessions = getConversationSessions(conv);

  // Create a temporary DB for this conversation.
  const dbPath = join(__dirname, 'data', `_bench_${run.id}_conv${convIdx}.db`);
  const entityId = `locomo_conv_${convIdx}`;

  // Set env vars for providers that read from env.
  if (config.llmProvider === 'anthropic' && config.apiKey) {
    process.env.ANTHROPIC_API_KEY = config.apiKey;
  }

  // Dynamically import MemG from source.
  const { MemG } = await import('../src/index.ts');

  // Determine embed API key — embeddings might use a different provider than LLM.
  const embedApiKey = config.embedProvider === 'openai'
    ? config.apiKey
    : config.embedProvider === 'gemini'
      ? config.apiKey
      : undefined;

  // Resolve embed model/dimension per provider.
  const embedModelMap: Record<string, { model: string; dim: number }> = {
    gemini: { model: 'gemini-embedding-001', dim: 3072 },
    openai: { model: 'text-embedding-3-small', dim: 1536 },
  };
  const embedCfg = embedModelMap[config.embedProvider] ?? embedModelMap.gemini;

  const memg = new MemG({
    dbPath,
    llmProvider: config.llmProvider,
    llmModel: config.extractionModel,
    embedProvider: config.embedProvider as any,
    embedModel: embedCfg.model,
    embedDimension: embedCfg.dim,
    openaiApiKey: (config.llmProvider === 'openai' || config.embedProvider === 'openai') ? config.apiKey : undefined,
    geminiApiKey: (config.llmProvider === 'gemini' || config.embedProvider === 'gemini') ? config.apiKey : undefined,
    extract: true,
    recallLimit: 200,
    recallThreshold: 0.03,
    maxRecallCandidates: 0,  // Load ALL facts — let the search engine rank them.
    consciousMode: true,
    consciousLimit: 10,
    memoryTokenBudget: 8000,
    diversifyTopics: false,  // Don't penalize topic clusters — benchmark questions are topic-focused.
    freshnessBias: 0.1,
  } as any);

  await memg.init();

  // ── Phase 1: Ingest conversation ──
  const ingestStart = Date.now();
  let totalFacts = 0;

  for (let si = 0; si < sessions.length; si++) {
    const session = sessions[si];
    if (_cancelledRuns.has(run.id)) break;

    // Build messages array from this session's turns.
    const messages = session.turns.map((turn) => ({
      role: turn.speaker === speakerA ? 'user' : 'assistant',
      content: `[${turn.speaker}]: ${turn.text}`,
    }));

    if (messages.length === 0) continue;

    console.log(`[bench] conv ${convIdx} | ingesting session ${si + 1}/${sessions.length} (${messages.length} turns)`);

    // Break sessions into small chunks for thorough extraction.
    // Smaller chunks = LLM misses fewer specific details.
    const isoDate = parseLoCoMoDate(session.date);
    const contextHeader = `[Context: Conversation between ${speakerA} and ${speakerB} on ${session.date}. ${speakerA} speaks as "user" role, ${speakerB} speaks as "assistant" role. Extract facts about BOTH speakers by name. Extract EVERY specific detail: names, places, dates, activities, items, preferences, advice given, emotions expressed.]`;
    const CHUNK_SIZE = 8;
    for (let ci = 0; ci < messages.length; ci += CHUNK_SIZE) {
      const chunk = messages.slice(ci, ci + CHUNK_SIZE);
      const chunkMessages = [
        { role: 'user', content: contextHeader },
        ...chunk,
      ];
      try {
        await memg.extractFromMessages(entityId, chunkMessages, isoDate);
      } catch (err) {
        console.warn(`[bench] conv ${convIdx} | session ${si + 1} chunk ${Math.floor(ci / CHUNK_SIZE) + 1} extraction FAILED: ${(err as Error).message}`);
      }
    }
    onSessionIngested();
  }

  // Count facts using list() which returns all facts without semantic filtering.
  try {
    const listResult = await memg.list(entityId, { limit: 9999 });
    totalFacts = listResult.count;
  } catch {
    totalFacts = -1;
  }

  const ingestionTimeMs = Date.now() - ingestStart;
  console.log(`[bench] conv ${convIdx} | ingestion DONE — ${totalFacts} facts in ${(ingestionTimeMs / 1000).toFixed(1)}s`);

  // ── Phase 2: Evaluate QA ──
  run.progress.currentPhase = 'evaluating';

  const evalStart = Date.now();
  const questions: QuestionResult[] = [];
  const qas = entry.qa.filter(
    (q) => config.categories.length === 0 || config.categories.includes(q.category)
  );

  // Process questions in batches (configurable for throughput vs rate limits).
  const Q_BATCH = config.questionBatchSize ?? 5;
  for (let i = 0; i < qas.length; i += Q_BATCH) {
    if (_cancelledRuns.has(run.id)) break;

    const batch = qas.slice(i, i + Q_BATCH);
    console.log(`[bench] conv ${convIdx} | evaluating qs ${i + 1}-${Math.min(i + Q_BATCH, qas.length)}/${qas.length}`);
    const results = await Promise.all(
      batch.map(async (qa, batchIdx) => {
        const qIdx = i + batchIdx;
        const isAdversarial = qa.category === 5;
        const groundTruth = isAdversarial ? '' : String(qa.answer ?? '');

        // Recall context from MemG.
        let memoryContext = '';
        try {
          memoryContext = await memg.buildMemoryContext(entityId, qa.question);
        } catch (err) {
          console.warn(`[bench] recall failed for conv ${convIdx} q ${qIdx}:`, err);
        }

        // Ask the LLM to answer.
        let predicted = '';
        try {
          const prompt = buildAnswerPrompt(qa.question, memoryContext, speakerA, speakerB);
          predicted = await callLLM(prompt, config.apiKey!, config.answerModel, config.llmProvider);
        } catch (err) {
          console.warn(`[bench] answer LLM failed for conv ${convIdx} q ${qIdx}:`, err);
          predicted = '[ERROR]';
        }

        // Score.
        let f1 = 0;
        let em = false;
        let judge: boolean | undefined;

        if (isAdversarial) {
          // For adversarial: score based on refusal.
          const refused = isRefusal(predicted);
          f1 = refused ? 1.0 : 0.0;
          em = refused;

          if (config.useLLMJudge && config.apiKey) {
            try {
              judge = await llmJudge(qa.question, '', predicted, true, config.apiKey, config.judgeModel, config.llmProvider);
            } catch {
              judge = undefined;
            }
          }
        } else {
          f1 = tokenF1(predicted, groundTruth);
          em = exactMatch(predicted, groundTruth);

          if (config.useLLMJudge && config.apiKey) {
            try {
              judge = await llmJudge(qa.question, groundTruth, predicted, false, config.apiKey, config.judgeModel, config.llmProvider);
            } catch {
              judge = undefined;
            }
          }
        }

        const diagnostics = buildQuestionDiagnostics({
          groundTruth,
          predicted,
          memoryContext,
          f1,
          isAdversarial,
        });

        return {
          questionIdx: qIdx,
          conversationIdx: convIdx,
          question: qa.question,
          category: qa.category,
          categoryName: CATEGORY_NAMES[qa.category] ?? `Category ${qa.category}`,
          groundTruth: isAdversarial ? `[Adversarial — should refuse] ${qa.adversarial_answer ?? ''}` : groundTruth,
          predicted,
          memoryContext: memoryContext.slice(0, 2000), // Truncate for storage.
          f1,
          exactMatch: em,
          llmJudge: judge,
          evidenceIds: qa.evidence,
          diagnostics,
        } satisfies QuestionResult;
      })
    );

    questions.push(...results);
    onQuestionBatch(results.length);
  }

  const evaluationTimeMs = Date.now() - evalStart;

  // Cleanup temp DB.
  memg.close();
  try {
    unlinkSync(dbPath);
  } catch {}

  const categoryScores = computeCategoryScores(questions);

  const nonAdv = questions.filter((q) => q.category !== 5);
  const overallF1 = nonAdv.length > 0 ? nonAdv.reduce((s, q) => s + q.f1, 0) / nonAdv.length : 0;
  const overallEM = nonAdv.length > 0 ? nonAdv.filter((q) => q.exactMatch).length / nonAdv.length : 0;
  const judged = nonAdv.filter((q) => q.llmJudge !== undefined);
  const overallLLMJudge = judged.length > 0 ? judged.filter((q) => q.llmJudge).length / judged.length : undefined;

  return {
    conversationIdx: convIdx,
    speakerA,
    speakerB,
    sessions: sessions.length,
    turns: sessions.reduce((s, sess) => s + sess.turns.length, 0),
    totalQuestions: questions.length,
    factsExtracted: totalFacts,
    questions,
    categoryScores,
    overallF1,
    overallExactMatch: overallEM,
    overallLLMJudge: overallLLMJudge,
    diagnostics: summarizeDiagnostics(questions),
    ingestionTimeMs,
    evaluationTimeMs,
  };
}

// ── Scoring helpers ──

function computeCategoryScores(questions: QuestionResult[]): CategoryScore[] {
  const categories = [4, 1, 2, 3, 5]; // LoCoMo ordering.
  const scores: CategoryScore[] = [];

  for (const cat of categories) {
    const qs = questions.filter((q) => q.category === cat);
    if (qs.length === 0) continue;

    const f1Avg = qs.reduce((s, q) => s + q.f1, 0) / qs.length;
    const emRate = qs.filter((q) => q.exactMatch).length / qs.length;
    const judged = qs.filter((q) => q.llmJudge !== undefined);
    const llmRate = judged.length > 0 ? judged.filter((q) => q.llmJudge).length / judged.length : undefined;

    scores.push({
      category: cat,
      name: CATEGORY_NAMES[cat] ?? `Category ${cat}`,
      total: qs.length,
      f1Avg,
      exactMatchRate: emRate,
      llmJudgeRate: llmRate,
      diagnostics: summarizeDiagnostics(questions, cat),
    });
  }

  return scores;
}

function computeOverall(conversations: ConversationResult[]): OverallScore {
  const allQs = conversations.flatMap((c) => c.questions);
  const nonAdv = allQs.filter((q) => q.category !== 5);
  const advQs = allQs.filter((q) => q.category === 5);

  const f1Avg = nonAdv.length > 0 ? nonAdv.reduce((s, q) => s + q.f1, 0) / nonAdv.length : 0;
  const emRate = nonAdv.length > 0 ? nonAdv.filter((q) => q.exactMatch).length / nonAdv.length : 0;
  const judged = nonAdv.filter((q) => q.llmJudge !== undefined);
  const llmRate = judged.length > 0 ? judged.filter((q) => q.llmJudge).length / judged.length : undefined;

  const advJudged = advQs.filter((q) => q.llmJudge !== undefined);
  const advRejection = advJudged.length > 0
    ? advJudged.filter((q) => q.llmJudge).length / advJudged.length
    : advQs.length > 0
      ? advQs.filter((q) => q.f1 === 1.0).length / advQs.length
      : undefined;

  return {
    totalQuestions: allQs.length,
    totalFacts: conversations.reduce((s, c) => s + Math.max(0, c.factsExtracted), 0),
    f1Avg,
    exactMatchRate: emRate,
    llmJudgeRate: llmRate,
    categoryScores: computeCategoryScores(allQs),
    adversarialRejectionRate: advRejection,
    diagnostics: summarizeDiagnostics(allQs),
  };
}

function emptyOverall(): OverallScore {
  return {
    totalQuestions: 0,
    totalFacts: 0,
    f1Avg: 0,
    exactMatchRate: 0,
    categoryScores: [],
  };
}

// ── List past runs ──

export function listRuns(): BenchmarkRun[] {
  const resultsDir = join(__dirname, 'results');
  if (!existsSync(resultsDir)) return [];

  const files: string[] = readdirSync(resultsDir).filter((f: string) => f.startsWith('run_') && f.endsWith('.json'));

  return files
    .map((f: string) => {
      try {
        return JSON.parse(readFileSync(join(resultsDir, f), 'utf8')) as BenchmarkRun;
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((a: any, b: any) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime()) as BenchmarkRun[];
}

export function getRun(runId: string): BenchmarkRun | null {
  const resultsDir = join(__dirname, 'results');
  const filePath = join(resultsDir, `run_${runId}.json`);
  if (!existsSync(filePath)) return null;
  return JSON.parse(readFileSync(filePath, 'utf8'));
}
