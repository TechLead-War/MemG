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
export declare class TransformersEmbedder implements Embedder {
    private pipe;
    private dim;
    private name;
    private constructor();
    static create(modelName?: string): Promise<TransformersEmbedder>;
    embed(texts: string[]): Promise<number[][]>;
    dimension(): number;
    modelName(): string;
}
/**
 * OpenAI embeddings API provider.
 * Uses native fetch (Node 18+).
 */
export declare class OpenAIEmbedder implements Embedder {
    private apiKey;
    private model;
    private dim;
    constructor(apiKey: string, model?: string, dim?: number);
    embed(texts: string[]): Promise<number[][]>;
    dimension(): number;
    modelName(): string;
}
/**
 * Google Gemini embeddings API provider.
 * Uses the batchEmbedContents endpoint for efficient batching (up to 100 texts per call).
 */
export declare class GeminiEmbedder implements Embedder {
    private static MAX_BATCH_SIZE;
    private apiKey;
    private model;
    private dim;
    private baseURL;
    constructor(apiKey: string, model?: string, dim?: number, baseURL?: string);
    embed(texts: string[]): Promise<number[][]>;
    private embedBatch;
    dimension(): number;
    modelName(): string;
}
