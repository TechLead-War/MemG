/**
 * LoCoMo evaluation metrics: token-level F1, exact match, and LLM-as-judge.
 */

// ── Text normalization ──

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\b(a|an|the|is|are|was|were|do|does|did|has|have|had)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text: string): string[] {
  return normalize(text).split(' ').filter(Boolean);
}

// ── Token-level F1 ──

export function tokenF1(prediction: string, groundTruth: string): number {
  const predTokens = tokenize(prediction);
  const truthTokens = tokenize(groundTruth);

  if (predTokens.length === 0 && truthTokens.length === 0) return 1.0;
  if (predTokens.length === 0 || truthTokens.length === 0) return 0.0;

  // Bag-of-words overlap: count min(pred_freq, truth_freq) per token.
  const truthFreq = new Map<string, number>();
  for (const t of truthTokens) {
    truthFreq.set(t, (truthFreq.get(t) ?? 0) + 1);
  }
  const predFreq = new Map<string, number>();
  for (const t of predTokens) {
    predFreq.set(t, (predFreq.get(t) ?? 0) + 1);
  }

  let overlap = 0;
  for (const [token, pCount] of predFreq) {
    const tCount = truthFreq.get(token) ?? 0;
    overlap += Math.min(pCount, tCount);
  }

  if (overlap === 0) return 0.0;

  const precision = overlap / predTokens.length;
  const recall = overlap / truthTokens.length;
  return (2 * precision * recall) / (precision + recall);
}

// ── Exact match ──

export function exactMatch(prediction: string, groundTruth: string): boolean {
  return normalize(prediction) === normalize(groundTruth);
}

// ── Adversarial detection ──

const REFUSAL_PATTERNS = [
  /not mentioned/i,
  /not discussed/i,
  /no information/i,
  /never (mentioned|discussed|talked|said)/i,
  /doesn'?t (mention|discuss|say|talk)/i,
  /don'?t (have|know|find|see)/i,
  /cannot (find|determine|answer)/i,
  /no evidence/i,
  /not (in|from) (the|this) conversation/i,
  /wasn'?t (mentioned|discussed)/i,
  /i (don'?t|do not) (have|know)/i,
  /there (is|are) no/i,
  /not covered/i,
  /unanswerable/i,
  /unable to (find|determine|answer)/i,
  /not enough information/i,
];

export function isRefusal(response: string): boolean {
  const lower = response.toLowerCase().trim();
  if (lower.length < 5) return true; // extremely short = likely no real answer
  return REFUSAL_PATTERNS.some((p) => p.test(lower));
}

// ── LLM-as-judge ──

export function buildJudgePrompt(question: string, groundTruth: string, predicted: string): string {
  return `You are an impartial judge evaluating whether a predicted answer is correct for a given question. Compare the predicted answer against the ground truth answer.

Question: ${question}

Ground Truth Answer: ${groundTruth}

Predicted Answer: ${predicted}

Evaluate whether the predicted answer conveys the same information as the ground truth. The predicted answer does not need to match word-for-word — it just needs to contain the key facts. Minor wording differences, additional context, or slightly different phrasing are acceptable.

Respond with ONLY one word: "CORRECT" or "INCORRECT"`;
}

export function buildAdversarialJudgePrompt(question: string, predicted: string): string {
  return `You are an impartial judge evaluating whether a system correctly refused to answer an unanswerable question. The question asks about information that was NEVER discussed in the conversation.

Question: ${question}

System Response: ${predicted}

A correct response should indicate that the information was not discussed, not mentioned, or that it cannot answer. An incorrect response would be one that fabricates or guesses an answer.

Respond with ONLY one word: "CORRECT" (if the system correctly refused/indicated it doesn't know) or "INCORRECT" (if the system fabricated an answer)`;
}

export async function llmJudge(
  question: string,
  groundTruth: string,
  predicted: string,
  isAdversarial: boolean,
  apiKey: string,
  model: string,
  provider: string
): Promise<boolean> {
  const prompt = isAdversarial
    ? buildAdversarialJudgePrompt(question, predicted)
    : buildJudgePrompt(question, groundTruth, predicted);

  const response = await callLLM(prompt, apiKey, model, provider);
  return response.trim().toUpperCase().startsWith('CORRECT');
}

// ── LLM calling utility ──

export async function callLLM(
  prompt: string,
  apiKey: string,
  model: string,
  provider: string
): Promise<string> {
  if (provider === 'openai' || provider === 'openai-compatible') {
    return callOpenAI(prompt, apiKey, model);
  }
  if (provider === 'anthropic') {
    return callAnthropic(prompt, apiKey, model);
  }
  if (provider === 'gemini') {
    return callGemini(prompt, apiKey, model);
  }
  throw new Error(`Unsupported LLM provider for evaluation: ${provider}`);
}

async function callOpenAI(prompt: string, apiKey: string, model: string): Promise<string> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      max_tokens: 1024,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error ${res.status}: ${err}`);
  }

  const json = await res.json();
  return json.choices?.[0]?.message?.content ?? '';
}

async function callAnthropic(prompt: string, apiKey: string, model: string): Promise<string> {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      max_tokens: 1024,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Anthropic API error ${res.status}: ${err}`);
  }

  const json = await res.json();
  return json.content?.[0]?.text ?? '';
}

async function callGemini(prompt: string, apiKey: string, model: string): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0, maxOutputTokens: 1024 },
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini API error ${res.status}: ${err}`);
  }

  const json = await res.json();
  return json.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
}
