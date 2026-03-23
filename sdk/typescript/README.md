# MemG TypeScript SDK

Memory layer for LLM applications. Zero runtime dependencies.

## Install

```bash
npm install memg
```

## Quick Start

### Proxy Mode (Recommended)

Route LLM traffic through the MemG proxy for transparent memory recall and extraction.

```typescript
import { MemG } from 'memg';
import OpenAI from 'openai';

const openai = MemG.wrap(new OpenAI(), { entity: 'user-123' });

const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'What do I usually order?' }],
});
```

### Client Mode

Intercepts calls locally without running the proxy. Requires only the MCP server.

```typescript
import { MemG } from 'memg';
import Anthropic from '@anthropic-ai/sdk';

const anthropic = MemG.wrap(new Anthropic(), {
  entity: 'user-123',
  mode: 'client',
});

const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Remind me about my preferences' }],
});
```

### Direct Memory Operations

```typescript
import { MemG } from 'memg';

const m = new MemG();

// Add memories (flexible input)
await m.add('user-123', 'likes coffee');
await m.add('user-123', ['works at Acme', 'prefers dark mode']);
await m.add('user-123', [
  { content: 'allergic to peanuts', type: 'identity', significance: 'high' },
]);

// Search
const results = await m.search('user-123', 'food preferences');
console.log(results.memories);

// List all
const all = await m.list('user-123', { type: 'identity' });

// Delete
await m.delete('user-123', 'memory-uuid');
await m.deleteAll('user-123');
```

### Low-Level MCP Client

```typescript
import { MemGClient } from 'memg';

const client = new MemGClient('http://localhost:8686');

await client.add('user-1', [{ content: 'likes TypeScript' }]);
const results = await client.search('user-1', 'programming languages');
```

### Gemini

```typescript
import { MemG } from 'memg';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genai = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const model = MemG.wrap(genai.getGenerativeModel({ model: 'gemini-2.5-flash' }), {
  entity: 'user-123',
  nativeConfig: { geminiApiKey: process.env.GEMINI_API_KEY, embedProvider: 'gemini' },
});

const result = await model.generateContent('What do you remember about me?');
```

### Custom Store (MySQL, Postgres, etc.)

The SDK defaults to SQLite via `better-sqlite3`, but you can pass any object implementing the `Store` interface:

```typescript
import { MemG, type Store } from 'memg';

class MyPostgresStore implements Store {
  // Implement all interface methods...
}

const m = new MemG({ store: new MyPostgresStore(connectionString) });
```

See `Store` in `store.ts` for the full interface contract (30 methods).

## Supported Providers

| Provider | `wrap()` | Extraction | Embeddings |
|----------|----------|------------|------------|
| OpenAI | `new OpenAI()` | Yes | Yes |
| Anthropic | `new Anthropic()` | Yes | No |
| Gemini | `genai.getGenerativeModel()` | Yes | Yes |

## Configuration

| Option     | Default                    | Description                          |
| ---------- | -------------------------- | ------------------------------------ |
| `mcpUrl`   | `http://localhost:8686`    | MCP server URL                       |
| `proxyUrl` | `http://localhost:8787/v1` | MemG proxy URL                       |
| `entity`   | —                          | Entity identifier for memory scoping |
| `mode`     | `native`                   | `native`, `proxy`, or `client`       |
| `extract`  | `true`                     | Extract knowledge from responses     |
| `nativeConfig.store` | —              | Custom `Store` implementation        |
| `nativeConfig.embedProvider` | `sentence-transformers` | `sentence-transformers`, `openai`, or `gemini` |
| `nativeConfig.geminiApiKey` | —       | Gemini API key for embeddings/LLM    |

## Requirements

- Node.js >= 18 (native `fetch`)
- For native mode: `better-sqlite3` (auto-installed)
- For proxy/client mode: MemG server running
- OpenAI SDK >= 4.0.0 (optional peer dependency)
- Anthropic SDK >= 0.20.0 (optional peer dependency)
- @google/generative-ai (optional peer dependency)
