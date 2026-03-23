/**
 * Generates a thin ESM wrapper over the CJS build.
 *
 * This avoids recompiling the source (which uses require() for lazy loading)
 * while giving ESM consumers proper import/export support.
 */

const fs = require('fs');
const path = require('path');

const esmDir = path.join(__dirname, '..', 'dist', 'esm');
fs.mkdirSync(esmDir, { recursive: true });

// ESM wrapper — uses createRequire to load the CJS build and re-exports.
const wrapper = `import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const memg = require('../index.js');

// Classes
export const MemG = memg.MemG;
export const MemGClient = memg.MemGClient;
export const MemGStore = memg.MemGStore;
export const HybridEngine = memg.HybridEngine;
export const TransformersEmbedder = memg.TransformersEmbedder;
export const OpenAIEmbedder = memg.OpenAIEmbedder;
export const GeminiEmbedder = memg.GeminiEmbedder;
export const PostgresStore = memg.PostgresStore;
export const MySQLStore = memg.MySQLStore;

// Functions
export const wrapOpenAIProxy = memg.wrapOpenAIProxy;
export const wrapAnthropicProxy = memg.wrapAnthropicProxy;
export const wrapOpenAIClient = memg.wrapOpenAIClient;
export const wrapAnthropicClient = memg.wrapAnthropicClient;
export const wrapGeminiClient = memg.wrapGeminiClient;
export const detectProvider = memg.detectProvider;
export const defaultContentKey = memg.defaultContentKey;
export const cosineSimilarity = memg.cosineSimilarity;
export const dimensionMatch = memg.dimensionMatch;
export const buildContext = memg.buildContext;
export const estimateTokens = memg.estimateTokens;
export const runExtraction = memg.runExtraction;
export const isTrivialTurn = memg.isTrivialTurn;
export const recallFacts = memg.recallFacts;
export const recallSummaries = memg.recallSummaries;

export default memg;
`;

fs.writeFileSync(path.join(esmDir, 'index.mjs'), wrapper);
console.log('ESM wrapper generated at dist/esm/index.mjs');
