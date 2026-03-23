/**
 * Embedding providers for the native MemG engine.
 */

/** Embedder interface matching the Go embed.Embedder contract. */
export interface Embedder {
  embed(texts: string[]): Promise<number[][]>;
  dimension(): number;
  modelName(): string;
}

/**
 * Local embedding using @huggingface/transformers (ONNX models).
 * No API keys required. Optional peer dependency.
 */
export class TransformersEmbedder implements Embedder {
  private pipe: any;
  private dim: number;
  private name: string;

  private constructor(pipe: any, dim: number, name: string) {
    this.pipe = pipe;
    this.dim = dim;
    this.name = name;
  }

  static async create(
    modelName: string = 'Xenova/all-MiniLM-L6-v2'
  ): Promise<TransformersEmbedder> {
    let transformers: any;
    try {
      transformers = require('@huggingface/transformers');
    } catch {
      throw new Error(
        'TransformersEmbedder requires @huggingface/transformers. Install it: npm i @huggingface/transformers'
      );
    }

    const pipe = await transformers.pipeline('feature-extraction', modelName, {
      quantized: true,
    });

    // Run a test embedding to determine dimension.
    const testResult = await pipe('test', { pooling: 'mean', normalize: true });
    const testData = Array.from(testResult.data as Float32Array) as number[];
    const dim = testData.length;

    return new TransformersEmbedder(pipe, dim, modelName);
  }

  async embed(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (const text of texts) {
      const output = await this.pipe(text, { pooling: 'mean', normalize: true });
      results.push(Array.from(output.data as Float32Array));
    }
    return results;
  }

  dimension(): number {
    return this.dim;
  }

  modelName(): string {
    return this.name;
  }
}

/**
 * OpenAI embeddings API provider.
 * Uses native fetch (Node 18+).
 */
export class OpenAIEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dim: number;

  constructor(apiKey: string, model: string = 'text-embedding-3-small', dim: number = 1536) {
    if (!apiKey) {
      throw new Error('OpenAIEmbedder requires an API key');
    }
    this.apiKey = apiKey;
    this.model = model;
    this.dim = dim;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
        dimensions: this.dim,
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`OpenAI embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    // Sort by index to preserve input order.
    const sorted = data.data.sort((a, b) => a.index - b.index);
    return sorted.map((d) => d.embedding);
  }

  dimension(): number {
    return this.dim;
  }

  modelName(): string {
    return this.model;
  }
}

/**
 * Google Gemini embeddings API provider.
 * Uses the batchEmbedContents endpoint for efficient batching (up to 100 texts per call).
 */
export class GeminiEmbedder implements Embedder {
  private static MAX_BATCH_SIZE = 100;
  private apiKey: string;
  private model: string;
  private dim: number;
  private baseURL: string;

  constructor(
    apiKey: string,
    model: string = 'text-embedding-004',
    dim: number = 768,
    baseURL: string = 'https://generativelanguage.googleapis.com'
  ) {
    if (!apiKey) {
      throw new Error('GeminiEmbedder requires an API key');
    }
    this.apiKey = apiKey;
    this.model = model;
    this.dim = dim;
    this.baseURL = baseURL;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const results: number[][] = [];

    for (let start = 0; start < texts.length; start += GeminiEmbedder.MAX_BATCH_SIZE) {
      const batch = texts.slice(start, start + GeminiEmbedder.MAX_BATCH_SIZE);
      const vecs = await this.embedBatch(batch);
      results.push(...vecs);
    }

    return results;
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    const modelRef = `models/${this.model}`;
    const url = `${this.baseURL}/v1beta/${modelRef}:batchEmbedContents?key=${this.apiKey}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        requests: texts.map((text) => ({
          model: modelRef,
          content: { parts: [{ text }] },
        })),
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Gemini embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as {
      embeddings: Array<{ values: number[] }>;
    };

    if (data.embeddings.length !== texts.length) {
      throw new Error(
        `Gemini: expected ${texts.length} embeddings, got ${data.embeddings.length}`
      );
    }

    return data.embeddings.map((e) => e.values);
  }

  dimension(): number {
    return this.dim;
  }

  modelName(): string {
    return this.model;
  }
}
