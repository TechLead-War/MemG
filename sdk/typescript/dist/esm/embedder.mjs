/**
 * Embedding providers for the native MemG engine.
 */
/**
 * Local embedding using @huggingface/transformers (ONNX models).
 * No API keys required. Optional peer dependency.
 */
export class TransformersEmbedder {
    constructor(pipe, dim, name) {
        this.pipe = pipe;
        this.dim = dim;
        this.name = name;
    }
    static async create(modelName = 'Xenova/all-MiniLM-L6-v2') {
        let transformers;
        try {
            const mod = '@huggingface/transformers';
            transformers = await import(mod);
            transformers = transformers.default ?? transformers;
        }
        catch {
            throw new Error('TransformersEmbedder requires @huggingface/transformers. Install it: npm i @huggingface/transformers');
        }
        const pipe = await transformers.pipeline('feature-extraction', modelName, {
            quantized: true,
        });
        // Run a test embedding to determine dimension.
        const testResult = await pipe('test', { pooling: 'mean', normalize: true });
        const testData = Array.from(testResult.data);
        const dim = testData.length;
        return new TransformersEmbedder(pipe, dim, modelName);
    }
    async embed(texts) {
        const results = [];
        for (const text of texts) {
            const output = await this.pipe(text, { pooling: 'mean', normalize: true });
            results.push(Array.from(output.data));
        }
        return results;
    }
    dimension() {
        return this.dim;
    }
    modelName() {
        return this.name;
    }
}
/**
 * OpenAI embeddings API provider.
 * Uses native fetch (Node 18+).
 */
export class OpenAIEmbedder {
    constructor(apiKey, model = 'text-embedding-3-small', dim = 1536) {
        if (!apiKey) {
            throw new Error('OpenAIEmbedder requires an API key');
        }
        this.apiKey = apiKey;
        this.model = model;
        this.dim = dim;
    }
    async embed(texts) {
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
        const data = (await response.json());
        // Sort by index to preserve input order.
        const sorted = data.data.sort((a, b) => a.index - b.index);
        return sorted.map((d) => d.embedding);
    }
    dimension() {
        return this.dim;
    }
    modelName() {
        return this.model;
    }
}
/**
 * Google Gemini embeddings API provider.
 * Uses the batchEmbedContents endpoint for efficient batching (up to 100 texts per call).
 */
export class GeminiEmbedder {
    constructor(apiKey, model = 'text-embedding-004', dim = 768, baseURL = 'https://generativelanguage.googleapis.com') {
        if (!apiKey) {
            throw new Error('GeminiEmbedder requires an API key');
        }
        this.apiKey = apiKey;
        this.model = model;
        this.dim = dim;
        this.baseURL = baseURL;
    }
    async embed(texts) {
        if (texts.length === 0)
            return [];
        const results = [];
        for (let start = 0; start < texts.length; start += GeminiEmbedder.MAX_BATCH_SIZE) {
            const batch = texts.slice(start, start + GeminiEmbedder.MAX_BATCH_SIZE);
            const vecs = await this.embedBatch(batch);
            results.push(...vecs);
        }
        return results;
    }
    async embedBatch(texts) {
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
        const data = (await response.json());
        if (data.embeddings.length !== texts.length) {
            throw new Error(`Gemini: expected ${texts.length} embeddings, got ${data.embeddings.length}`);
        }
        return data.embeddings.map((e) => e.values);
    }
    dimension() {
        return this.dim;
    }
    modelName() {
        return this.model;
    }
}
GeminiEmbedder.MAX_BATCH_SIZE = 100;
