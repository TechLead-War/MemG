/**
 * Embedding providers for the native MemG engine.
 */

/** Embedder interface — pluggable embedding provider contract. */
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
      const mod = '@huggingface/transformers';
      transformers = await import(mod);
      transformers = transformers.default ?? transformers;
    } catch (err) {
      console.warn('[memg] embedder: transformers import failed:', err);
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
    if (texts.length === 0) return [];

    // Process in batches of 32 to balance throughput vs. memory.
    const BATCH_SIZE = 32;
    const results: number[][] = [];

    for (let start = 0; start < texts.length; start += BATCH_SIZE) {
      const batch = texts.slice(start, start + BATCH_SIZE);
      const output = await this.pipe(batch, { pooling: 'mean', normalize: true });
      const data = output.data as Float32Array;

      if (batch.length === 1) {
        // Single input: data is [dim] flat array.
        results.push(Array.from(data));
      } else {
        // Multiple inputs: data is [batch_size * dim] flat array.
        const dim = this.dim;
        for (let i = 0; i < batch.length; i++) {
          results.push(Array.from(data.slice(i * dim, (i + 1) * dim)));
        }
      }
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

/**
 * Ollama local embeddings API provider.
 * Runs against a local Ollama instance (default: localhost:11434).
 */
export class OllamaEmbedder implements Embedder {
  private baseUrl: string;
  private model: string;
  private dim: number;

  constructor(
    model: string,
    dim: number,
    baseUrl: string = 'http://localhost:11434'
  ) {
    this.model = model;
    this.dim = dim;
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    // The /api/embed endpoint accepts `input` as a string or array of strings.
    // Send all texts in a single request for batch efficiency.
    const response = await fetch(`${this.baseUrl}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: texts }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Ollama embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as { embeddings: number[][] };
    if (!data.embeddings || data.embeddings.length !== texts.length) {
      throw new Error(
        `Ollama: expected ${texts.length} embeddings, got ${data.embeddings?.length ?? 0}`
      );
    }

    return data.embeddings;
  }

  dimension(): number {
    return this.dim;
  }

  modelName(): string {
    return this.model;
  }
}

/**
 * Azure OpenAI embeddings API provider.
 * Uses the Azure OpenAI resource endpoint with api-key auth.
 */
export class AzureOpenAIEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dim: number;
  private endpoint: string;
  private apiVersion: string;

  constructor(
    apiKey: string,
    endpoint: string,
    model: string,
    dim: number,
    apiVersion: string = '2024-10-21'
  ) {
    if (!apiKey) throw new Error('AzureOpenAIEmbedder requires an API key');
    if (!endpoint) throw new Error('AzureOpenAIEmbedder requires an endpoint URL');
    this.apiKey = apiKey;
    this.endpoint = endpoint.replace(/\/+$/, '');
    this.model = model;
    this.dim = dim;
    this.apiVersion = apiVersion;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const url = `${this.endpoint}/openai/deployments/${this.model}/embeddings?api-version=${this.apiVersion}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-key': this.apiKey,
      },
      body: JSON.stringify({ input: texts }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Azure OpenAI embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

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
 * AWS Bedrock embeddings API provider.
 * Requires @aws-sdk/client-bedrock-runtime as optional peer dependency.
 */
export class BedrockEmbedder implements Embedder {
  private client: any;
  private model: string;
  private dim: number;

  private constructor(client: any, model: string, dim: number) {
    this.client = client;
    this.model = model;
    this.dim = dim;
  }

  static async create(
    region: string = 'us-east-1',
    model: string = 'amazon.titan-embed-text-v2:0',
    dim: number = 1024
  ): Promise<BedrockEmbedder> {
    let bedrock: any;
    try {
      const mod = '@aws-sdk/client-bedrock-runtime';
      bedrock = await import(mod);
    } catch (err) {
      console.warn('[memg] embedder: bedrock SDK import failed:', err);
      throw new Error(
        'BedrockEmbedder requires @aws-sdk/client-bedrock-runtime. Install it: npm i @aws-sdk/client-bedrock-runtime'
      );
    }

    const client = new bedrock.BedrockRuntimeClient({ region });
    return new BedrockEmbedder(client, model, dim);
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const InvokeModelCommand = await this.getInvokeClass();
    const MAX_CONCURRENCY = 5;

    const embedOne = async (text: string): Promise<number[]> => {
      const command = new InvokeModelCommand({
        modelId: this.model,
        contentType: 'application/json',
        accept: 'application/json',
        body: JSON.stringify({ inputText: text }),
      });
      const response = await this.client.send(command);
      const body = JSON.parse(new TextDecoder().decode(response.body));
      return body.embedding;
    };

    // Process in parallel with bounded concurrency.
    const results: number[][] = new Array(texts.length);
    for (let start = 0; start < texts.length; start += MAX_CONCURRENCY) {
      const batch = texts.slice(start, start + MAX_CONCURRENCY);
      const batchResults = await Promise.all(batch.map((text) => embedOne(text)));
      for (let i = 0; i < batchResults.length; i++) {
        results[start + i] = batchResults[i];
      }
    }

    return results;
  }

  private async getInvokeClass(): Promise<any> {
    const mod = '@aws-sdk/client-bedrock-runtime';
    const bedrock = await import(mod);
    return bedrock.InvokeModelCommand;
  }

  dimension(): number {
    return this.dim;
  }

  modelName(): string {
    return this.model;
  }
}

/**
 * Together AI embeddings API provider.
 * Uses the OpenAI-compatible embeddings endpoint.
 */
export class TogetherEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dim: number;

  constructor(
    apiKey: string,
    model: string = 'togethercomputer/m2-bert-80M-8k-retrieval',
    dim: number = 768
  ) {
    if (!apiKey) throw new Error('TogetherEmbedder requires an API key');
    this.apiKey = apiKey;
    this.model = model;
    this.dim = dim;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const response = await fetch('https://api.together.xyz/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input: texts }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Together AI embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

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
 * Cohere embeddings API provider.
 * Uses the Cohere v2 embed endpoint.
 */
export class CohereEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dim: number;

  constructor(
    apiKey: string,
    model: string = 'embed-english-v3.0',
    dim: number = 1024
  ) {
    if (!apiKey) throw new Error('CohereEmbedder requires an API key');
    this.apiKey = apiKey;
    this.model = model;
    this.dim = dim;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const response = await fetch('https://api.cohere.ai/v2/embed', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        texts,
        input_type: 'search_document',
        embedding_types: ['float'],
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Cohere embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as {
      embeddings: { float: number[][] };
    };

    return data.embeddings.float;
  }

  dimension(): number {
    return this.dim;
  }

  modelName(): string {
    return this.model;
  }
}

/**
 * VoyageAI embeddings API provider.
 * Uses the Voyage embeddings endpoint.
 */
export class VoyageEmbedder implements Embedder {
  private apiKey: string;
  private model: string;
  private dim: number;

  constructor(
    apiKey: string,
    model: string = 'voyage-3',
    dim: number = 1024
  ) {
    if (!apiKey) throw new Error('VoyageEmbedder requires an API key');
    this.apiKey = apiKey;
    this.model = model;
    this.dim = dim;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const response = await fetch('https://api.voyageai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input: texts }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`VoyageAI embeddings API error: HTTP ${response.status} ${body}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

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
