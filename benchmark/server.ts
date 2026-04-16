/**
 * Benchmark HTTP server.
 * Serves the UI and exposes API endpoints for running benchmarks.
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname, dirname } from 'path';
import { fileURLToPath } from 'url';
import { runBenchmark, cancelRun, listRuns, getRun } from './runner.ts';
import type { BenchmarkConfig, BenchmarkRun, BenchmarkProgress } from './types.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = parseInt(process.env.BENCH_PORT ?? '3847', 10);

// ── Server-side API key resolution ──

function resolveApiKey(clientKey?: string): string | undefined {
  if (clientKey && clientKey !== '__USE_SERVER_KEY__') return clientKey;
  return process.env.GEMINI_API_KEY
    || process.env.OPENAI_API_KEY
    || process.env.ANTHROPIC_API_KEY
    || undefined;
}

function detectDefaultProvider(): string {
  if (process.env.GEMINI_API_KEY) return 'gemini';
  if (process.env.OPENAI_API_KEY) return 'openai';
  if (process.env.ANTHROPIC_API_KEY) return 'anthropic';
  return 'openai';
}

// ── Active run tracking ──

let activeRun: BenchmarkRun | null = null;
let activeProgress: BenchmarkProgress | null = null;

// ── MIME types ──

const MIME: Record<string, string> = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

// ── Request handling ──

async function handleRequest(req: IncomingMessage, res: ServerResponse) {
  const url = new URL(req.url ?? '/', `http://localhost:${PORT}`);
  const path = url.pathname;

  // CORS headers for dev.
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // ── API routes ──

  if (path === '/api/runs' && req.method === 'GET') {
    const runs = listRuns();
    // Return lightweight list (no question details).
    const lightweight = runs.map((r) => ({
      id: r.id,
      startedAt: r.startedAt,
      completedAt: r.completedAt,
      status: r.status,
      config: r.config,
      overall: r.overall,
      progress: r.progress,
      conversationCount: r.conversations.length,
    }));
    json(res, lightweight);
    return;
  }

  if (path.startsWith('/api/runs/') && req.method === 'GET') {
    const runId = path.split('/')[3];
    if (!runId) {
      json(res, { error: 'Missing run ID' }, 400);
      return;
    }

    // Check if it's the active run.
    if (activeRun && activeRun.id === runId) {
      json(res, { ...activeRun, progress: activeProgress });
      return;
    }

    const run = getRun(runId);
    if (!run) {
      json(res, { error: 'Run not found' }, 404);
      return;
    }
    json(res, run);
    return;
  }

  if (path === '/api/defaults' && req.method === 'GET') {
    const provider = detectDefaultProvider();
    json(res, {
      hasApiKey: !!resolveApiKey(),
      provider,
      embedProvider: provider === 'gemini' ? 'gemini' : provider === 'openai' ? 'openai' : 'openai',
    });
    return;
  }

  if (path === '/api/progress' && req.method === 'GET') {
    if (activeRun) {
      json(res, {
        running: true,
        runId: activeRun.id,
        progress: activeProgress,
        status: activeRun.status,
      });
    } else {
      json(res, { running: false });
    }
    return;
  }

  if (path === '/api/start' && req.method === 'POST') {
    if (activeRun && activeRun.status === 'running') {
      json(res, { error: 'A benchmark is already running', runId: activeRun.id }, 409);
      return;
    }

    const body = await readBody(req);
    let config: BenchmarkConfig;

    try {
      config = JSON.parse(body);
    } catch {
      json(res, { error: 'Invalid JSON body' }, 400);
      return;
    }

    // Resolve API key: client-provided or env var.
    config.apiKey = resolveApiKey(config.apiKey);
    if (!config.apiKey) {
      json(res, { error: 'No API key — set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY env var, or provide one in the UI' }, 400);
      return;
    }

    // Defaults.
    const defaultProvider = detectDefaultProvider();
    config.conversations = config.conversations ?? [];
    config.categories = config.categories ?? [];
    config.llmProvider = config.llmProvider ?? defaultProvider;
    config.extractionModel = config.extractionModel ?? 'gemini-2.0-flash';
    config.answerModel = config.answerModel ?? 'gemini-2.0-flash';
    config.judgeModel = config.judgeModel ?? 'gemini-2.0-flash';
    config.embedProvider = config.embedProvider ?? (defaultProvider === 'gemini' ? 'gemini' : 'openai');
    config.useLLMJudge = config.useLLMJudge ?? true;

    // Start benchmark in background.
    activeProgress = null;
    let resolvedRunId = '';

    activeRun = {
      id: '(starting)',
      startedAt: new Date().toISOString(),
      status: 'running',
      config: { ...config, apiKey: undefined },
      conversations: [],
      overall: { totalQuestions: 0, totalFacts: 0, f1Avg: 0, exactMatchRate: 0, categoryScores: [] },
      progress: { totalConversations: 0, completedConversations: 0, totalQuestions: 0, completedQuestions: 0 },
    };

    const runPromise = runBenchmark(config, (progress, runId) => {
      resolvedRunId = runId;
      activeProgress = progress;
      if (activeRun) {
        activeRun.id = runId;
        activeRun.progress = progress;
      }
      const phase = progress.currentPhase ?? 'init';
      const qPct = progress.totalQuestions > 0
        ? (progress.completedQuestions / progress.totalQuestions * 100).toFixed(1)
        : '0.0';
      console.log(`[bench ${runId}] ${phase} | conv ${progress.completedConversations}/${progress.totalConversations} | qs ${progress.completedQuestions}/${progress.totalQuestions} (${qPct}%)`);
    });

    runPromise
      .then((result) => {
        activeRun = result;
        console.log(`[bench ${result.id}] DONE — status: ${result.status} | F1: ${result.overall.f1Avg.toFixed(4)} | judge: ${result.overall.llmJudgeRate?.toFixed(4) ?? 'N/A'} | questions: ${result.overall.totalQuestions}`);
      })
      .catch((err) => {
        console.error(`[bench] FAILED:`, err.message);
        if (activeRun) {
          activeRun.status = 'failed';
          activeRun.error = err.message;
        }
      })
      .finally(() => {
        setTimeout(() => {
          activeRun = null;
          activeProgress = null;
        }, 30_000);
      });

    // Wait briefly for the first progress callback to fire (sets the real run ID).
    await new Promise((r) => setTimeout(r, 500));
    json(res, { started: true, runId: resolvedRunId || activeRun?.id });
    return;
  }

  if (path === '/api/cancel' && req.method === 'POST') {
    if (activeRun && activeRun.status === 'running') {
      cancelRun(activeRun.id);
      json(res, { cancelled: true, runId: activeRun.id });
    } else {
      json(res, { error: 'No active run to cancel' }, 400);
    }
    return;
  }

  // ── Static UI files ──

  let filePath: string;
  if (path === '/' || path === '/index.html') {
    filePath = join(__dirname, 'ui', 'index.html');
  } else {
    filePath = join(__dirname, 'ui', path);
  }

  if (existsSync(filePath)) {
    const ext = extname(filePath);
    const mime = MIME[ext] ?? 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(readFileSync(filePath));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not found');
}

// ── Helpers ──

function json(res: ServerResponse, data: any, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString()));
    req.on('error', reject);
  });
}

// ── Start server ──

const server = createServer((req, res) => {
  handleRequest(req, res).catch((err) => {
    console.error('[bench-server] Error:', err);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: err.message }));
  });
});

server.listen(PORT, () => {
  console.log(`\n  LoCoMo Benchmark UI → http://localhost:${PORT}\n`);
});
