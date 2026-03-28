# MemG

A pluggable memory layer for language model applications, written in Go.

MemG intercepts LLM calls to automatically inject relevant recalled facts from stored knowledge, tracks conversations across sessions, and asynchronously extracts and manages knowledge with an enriched fact lifecycle — decay, reinforcement, evolution, and deduplication.

## Features

- **Zero-code memory proxy** — add persistent memory to any LLM app in any language by changing one env var
- **10 LLM providers** — OpenAI, Anthropic, Gemini, Ollama, Azure OpenAI, AWS Bedrock, DeepSeek, Groq, Together AI, xAI
- **11 embedding providers** — OpenAI, Gemini, Ollama, HuggingFace, Azure OpenAI, AWS Bedrock, Together AI, Cohere, VoyageAI, in-process ONNX Runtime, plus a local sentence-transformers gRPC service
- **Plug-and-play storage** — PostgreSQL, SQLite, MySQL out of the box
- **Hybrid recall** — cosine similarity + BM25 lexical scoring with Kneedle dynamic cutoff
- **Memory lifecycle** — fact types (identity/event/pattern), temporal status (current/historical), significance-based decay, reinforcement, and automatic deduplication
- **Sliding sessions** — sessions extend on activity, auto-summarize on rollover, cap history to recent turns
- **Conversation summaries** — auto-generated on session expiry, recalled by relevance, pruned after 90 days
- **Built-in extraction** — the proxy ships with a default LLM-based extraction stage with slot/confidence/provenance output
- **Async augmentation pipeline** — fixed-worker pool with bounded queue, slot-based conflict resolution, trivial-turn gating
- **Fact provenance** — confidence, source role, embedding model, and slot tracked on every fact
- **Query transformation** — optional hook rewrites follow-up queries before embedding for better retrieval
- **Re-embedding** — migrate facts to a new embedding model without data loss
- **Consolidation** — background worker clusters old events into pattern facts
- **Local embeddings** — in-process ONNX Runtime (no Python, no external services) or legacy Python gRPC service, both with zero API keys
- **API key management** — per-provider config with env var fallback

## Install SDKs

```bash
# Python
pip install memg-sdk

# TypeScript / Node.js
npm install memg-core-js
```

## Publish SDKs

```bash
# PyPI
cd sdk/python && python -m build && twine upload dist/*

# npm
cd sdk/typescript && npm run build && npm publish
```

## Quick Start (Proxy — Any Language, Zero Code Changes)

The fastest way to use MemG. Start the proxy, set one env var, and your existing app gets persistent memory.

**1. Start the proxy:**

```bash
# Install
go install memg/cmd/memg@latest

# Start (uses SQLite at ~/.memg/memory.db, OpenAI for extraction + embeddings)
export OPENAI_API_KEY=sk-...
memg proxy
```

**2. Point your app to the proxy (change nothing else):**

```bash
# Python
OPENAI_BASE_URL=http://localhost:8787/v1 python my_app.py

# Node.js
OPENAI_BASE_URL=http://localhost:8787/v1 node my_app.js

# Any language, any SDK
OPENAI_BASE_URL=http://localhost:8787/v1 ./my_app
```

That's it. Every LLM call through your app now has persistent memory. No code changes. The proxy:
- Intercepts chat completion requests only (other API calls pass through untouched)
- Recalls relevant facts and conversation summaries
- Injects them into the system prompt
- Forwards to the real API
- Extracts knowledge from the response in the background
- Manages decay, deduplication, and reinforcement automatically

**Proxy options:**

```bash
memg proxy --help

  --port 8787              # proxy port (default: 8787)
  --target https://api.openai.com  # upstream API (default: OpenAI)
  --entity user-123        # single-entity mode
  --db ~/.memg/memory.db   # database path (default: ~/.memg/memory.db)
  --embed-provider openai  # embedding provider (default: openai)
  --embed-model ...        # embedding model (default: provider default)
  --llm-provider openai    # extraction LLM (default: openai)
  --llm-model gpt-4o-mini  # cheaper model for extraction
  --debug                  # verbose logging
```

**Works with any OpenAI-compatible API:**

```bash
# DeepSeek
memg proxy --target https://api.deepseek.com
DEEPSEEK_API_KEY=... OPENAI_BASE_URL=http://localhost:8787/v1 python my_app.py

# Groq
memg proxy --target https://api.groq.com/openai
GROQ_API_KEY=... OPENAI_BASE_URL=http://localhost:8787/v1 python my_app.py

# Anthropic
memg proxy --target https://api.anthropic.com
ANTHROPIC_API_KEY=... ANTHROPIC_BASE_URL=http://localhost:8787 python my_app.py

# Ollama (local, no API key needed)
memg proxy --target http://localhost:11434 --embed-provider ollama
OPENAI_BASE_URL=http://localhost:8787/v1 python my_app.py
```

**Entity identification:**

The proxy identifies users via the `X-MemG-Entity` header or the `--entity` flag:

```python
# Per-request entity (multi-user apps)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_headers={"X-MemG-Entity": "user-123"}
)

# Or single-entity mode (personal use)
# memg proxy --entity user-123
```

## Quick Start (Go Library)

```go
package main

import (
	"context"
	"database/sql"
	"log"

	_ "github.com/jackc/pgx/v5/stdlib"

	"memg"
	"memg/embed"
	"memg/llm"
	"memg/store/sqlstore"

	_ "memg/embed/openai"   // register OpenAI embedder
	_ "memg/llm/anthropic"  // register Anthropic provider
)

func main() {
	db, err := sql.Open("pgx", "postgres://localhost:5432/myapp")
	if err != nil {
		log.Fatal(err)
	}

	repo := sqlstore.NewPostgres(db)

	// One instance serves all users.
	g, err := memg.New(repo,
		memg.WithLLMProvider("anthropic", llm.ProviderConfig{
			Model: "claude-sonnet-4-20250514",
		}),
		memg.WithEmbedProvider("openai", embed.ProviderConfig{
			Model: "text-embedding-3-small",
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer g.Close()

	if err := g.Migrate(context.Background()); err != nil {
		log.Fatal(err)
	}

	// Pass the user ID per call — no need for separate instances.
	resp, err := g.Chat(context.Background(), []*llm.Message{
		llm.UserMessage("What do you remember about me?"),
	}, memg.ForEntity("user-alice"))
	if err != nil {
		log.Fatal(err)
	}
	log.Println(resp.Content)

	// Same instance, different user — completely isolated memory.
	resp, err = g.Chat(context.Background(), []*llm.Message{
		llm.UserMessage("Hello!"),
	}, memg.ForEntity("user-bob"))
	if err != nil {
		log.Fatal(err)
	}
	log.Println(resp.Content)
}
```

## Quick Start (Python SDK)

```bash
pip install memg-sdk
```

```python
from memg import MemG
import openai

# One line — wraps your existing client with memory
client = MemG.wrap(openai.OpenAI(), entity="user-123")

# Everything else stays the same
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What do you remember about me?"}],
)
```

Also supports Anthropic (`MemG.wrap(anthropic.Anthropic(), ...)`), Gemini (`MemG.wrap(model, entity="user-123")`), custom stores (`MemG(store=my_store)`), client mode (`mode="client"`), and direct memory operations (`MemG().add(...)`, `MemG().search(...)`).

See [`sdk/python/`](sdk/python/) for full documentation.

## Quick Start (TypeScript SDK)

```bash
npm install memg-core-js
```

```typescript
import { MemG } from 'memg';
import OpenAI from 'openai';

// One line — wraps your existing client with memory
const client = MemG.wrap(new OpenAI(), { entity: 'user-123' });

// Everything else stays the same
const resp = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'What do you remember about me?' }],
});
```

Also supports Anthropic (`MemG.wrap(new Anthropic(), ...)`), Gemini (`MemG.wrap(geminiModel, { entity: 'user-123' })`), custom stores (`new MemG({ store: myStore })`), client mode (`mode: 'client'`), and direct memory operations (`new MemG().add(...)`, `new MemG().search(...)`).

See [`sdk/typescript/`](sdk/typescript/) for full documentation.

## LLM Providers

Select a provider by name with `WithLLMProvider`. Import the provider package to register it.

| Provider | Registry Name | Default Model | API Key Env Var | Import |
|---|---|---|---|---|
| OpenAI | `openai` | gpt-4o | `OPENAI_API_KEY` | `_ "memg/llm/openai"` |
| Anthropic | `anthropic` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` | `_ "memg/llm/anthropic"` |
| Google Gemini | `gemini` | gemini-2.5-flash | `GEMINI_API_KEY` | `_ "memg/llm/gemini"` |
| Ollama (local) | `ollama` | (user specified) | none | `_ "memg/llm/ollama"` |
| Azure OpenAI | `azureopenai` | (user specified) | `AZURE_OPENAI_API_KEY` | `_ "memg/llm/azureopenai"` |
| AWS Bedrock | `bedrock` | anthropic.claude-sonnet-4-20250514-v1:0 | `AWS_ACCESS_KEY_ID` | `_ "memg/llm/bedrock"` |
| DeepSeek | `deepseek` | deepseek-chat | `DEEPSEEK_API_KEY` | `_ "memg/llm/deepseek"` |
| Groq | `groq` | llama-3.3-70b-versatile | `GROQ_API_KEY` | `_ "memg/llm/groq"` |
| Together AI | `togetherai` | meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo | `TOGETHER_API_KEY` | `_ "memg/llm/togetherai"` |
| xAI | `xai` | grok-3 | `XAI_API_KEY` | `_ "memg/llm/xai"` |

API keys are resolved in order: explicit `ProviderConfig.APIKey` field, then the environment variable, then error.

## Embedding Providers

| Provider | Registry Name | Default Model | Dimension | API Key Env Var | Import |
|---|---|---|---|---|---|
| OpenAI | `openai` | text-embedding-3-small | 1536 | `OPENAI_API_KEY` | `_ "memg/embed/openai"` |
| Google Gemini | `gemini` | text-embedding-004 | 768 | `GEMINI_API_KEY` | `_ "memg/embed/gemini"` |
| Ollama (local) | `ollama` | (user specified) | (user specified) | none | `_ "memg/embed/ollama"` |
| HuggingFace | `huggingface` | sentence-transformers/all-MiniLM-L6-v2 | 384 | `HF_API_KEY` | `_ "memg/embed/huggingface"` |
| Azure OpenAI | `azureopenai` | (user specified) | (user specified) | `AZURE_OPENAI_API_KEY` | `_ "memg/embed/azureopenai"` |
| AWS Bedrock | `bedrock` | amazon.titan-embed-text-v2:0 | 1024 | `AWS_ACCESS_KEY_ID` | `_ "memg/embed/bedrock"` |
| Together AI | `togetherai` | togethercomputer/m2-bert-80M-8k-retrieval | 768 | `TOGETHER_API_KEY` | `_ "memg/embed/togetherai"` |
| Cohere | `cohere` | embed-english-v3.0 | 1024 | `COHERE_API_KEY` | `_ "memg/embed/cohere"` |
| VoyageAI | `voyageai` | voyage-3 | 1024 | `VOYAGE_API_KEY` | `_ "memg/embed/voyageai"` |
| ONNX Runtime | `onnx` | all-MiniLM-L6-v2 | 384 | none | `_ "memg/embed/onnx"` |
| Local (gRPC) | `local` | all-MiniLM-L6-v2 | auto-detected | none | `_ "memg/embed/local"` |

## Embedding Modes

**Most users need 1 service — just the Go proxy.** If you use a cloud embedding provider (OpenAI, Cohere, etc.), the proxy handles everything. No Python needed:

```bash
memg proxy   # uses OpenAI for embeddings (same API key as your LLM calls)
```

**Local embeddings (recommended: ONNX)** — in-process, no Python, no external services, no API keys:

```bash
memg proxy --embed-provider onnx
```

This runs the embedding model directly inside the Go process using ONNX Runtime. See [ONNX Local Embeddings](#onnx-local-embeddings-recommended) below.

**Local embeddings (legacy: gRPC)** — the Go proxy + a separate Python embedding service:

```bash
# Terminal 1: Python embedding service
cd local/embedder && python server.py

# Terminal 2: Go proxy
memg proxy --embed-provider local
```

The gRPC approach requires Python and a separate process. It remains available for users who need PyTorch-specific features or custom models not exported to ONNX.

## ONNX Local Embeddings (Recommended)

Run embedding models directly inside the Go process using ONNX Runtime. No Python, no external services, no API keys, no data leaves your machine.

### Prerequisites

**1. Install ONNX Runtime:**

```bash
# macOS (Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-osx-arm64-1.24.4.tgz
tar xzf onnxruntime-osx-arm64-1.24.4.tgz
cp onnxruntime-osx-arm64-1.24.4/lib/libonnxruntime.dylib /opt/homebrew/lib/

# Linux (x86_64)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz
tar xzf onnxruntime-linux-x64-1.24.4.tgz
sudo cp onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so /usr/local/lib/

# Or set ONNX_RUNTIME_LIB to point to the library directly
export ONNX_RUNTIME_LIB=/path/to/libonnxruntime.dylib
```

**2. Download model files:**

```bash
mkdir -p ~/.memg/models/all-MiniLM-L6-v2
# Download model.onnx and vocab.txt for all-MiniLM-L6-v2
# from HuggingFace: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### Use

```bash
# Proxy mode
memg proxy --embed-provider onnx

# With custom model directory
memg proxy --embed-provider onnx --embed-base-url /path/to/model/dir
```

### Use from Go

```go
import _ "memg/embed/onnx"

g, _ := memg.New(repo,
    memg.WithEmbedProvider("onnx", embed.ProviderConfig{}), // defaults to ~/.memg/models/all-MiniLM-L6-v2
)

// Or with a custom model directory:
g, _ := memg.New(repo,
    memg.WithEmbedProvider("onnx", embed.ProviderConfig{
        BaseURL: "/path/to/my/model/dir",  // must contain model.onnx + vocab.txt
    }),
)
```

### How It Works

The ONNX provider loads the model and vocabulary into the Go process at startup. Each `Embed()` call runs inference in-process via ONNX Runtime — no network calls, no Python, no gRPC. Embeddings are mean-pooled and L2-normalized, matching the output of the Python sentence-transformers service.

## Local Embeddings via gRPC (Legacy)

MemG also includes a Python gRPC service that runs sentence-transformers models. This is the legacy approach — use the ONNX provider above unless you need PyTorch-specific features.

### Setup

```bash
cd local/embedder
pip install -r requirements.txt
```

### Run

```bash
# Default: all-MiniLM-L6-v2 (384d, fast)
python server.py

# Higher quality model
python server.py --model all-mpnet-base-v2

# Any HuggingFace sentence-transformers model
python server.py --model BAAI/bge-small-en-v1.5
python server.py --model intfloat/e5-large-v2

# Custom port and GPU
python server.py --port 50052 --device cuda
```

### Docker

```bash
# Build from repo root
docker build -f local/embedder/Dockerfile -t memg-embedder .

# Run with default model
docker run -p 50051:50051 memg-embedder

# Run with custom model
docker run -p 50051:50051 memg-embedder --model all-mpnet-base-v2

# Persist model cache across restarts
docker run -p 50051:50051 -v hf_cache:/root/.cache/huggingface memg-embedder
```

### Use from Go

```go
import _ "memg/embed/local"

g, _ := memg.New(repo,
    memg.WithEmbedProvider("local", embed.ProviderConfig{}), // defaults to localhost:50051
)
```

The Go client auto-detects the model dimension from the service — no manual configuration needed.

## Supported Databases

| Database | Constructor | Driver |
|---|---|---|
| PostgreSQL | `sqlstore.NewPostgres(db)` | `github.com/jackc/pgx/v5/stdlib` |
| SQLite | `sqlstore.NewSQLite(db)` | `modernc.org/sqlite` |
| MySQL | `sqlstore.NewMySQL(db)` | `github.com/go-sql-driver/mysql` |

## Memory Lifecycle

MemG doesn't just store facts — it manages their lifecycle:

- **Fact types** — identity (enduring truths), event (point-in-time occurrences), pattern (observed tendencies)
- **Temporal status** — facts are current or historical. "I live in Austin" becomes historical when "I live in Seattle" arrives
- **Significance-based decay** — high-significance facts (allergies, life events) live indefinitely. Low-significance facts (what you ate for lunch) expire in days
- **Reinforcement** — repeated mentions reset a fact's TTL. Deduplication is automatic via content key hashing
- **Slot-based conflict resolution** — extraction assigns a `slot` (e.g. `"location"`) to single-valued facts. The pipeline auto-reclassifies old values to historical. For domain-specific conflicts, stages implement `ConflictDetector`
- **Provenance tracking** — each fact records its `confidence` (0.0–1.0), `source_role` (user vs assistant), `embedding_model`, and `slot`. Low-confidence assistant guesses are filtered out
- **Recall usage tracking** — `recall_count` and `last_recalled_at` track which facts are actually used. Stale unreinforced facts are demoted in conscious mode
- **Dynamic recall** — the Kneedle algorithm finds the natural cutoff in score distributions instead of returning a fixed top-N. Confidence acts as a ranking tiebreaker
- **Consolidation** — old event facts are periodically clustered into pattern facts by the background consolidator
- **Summary pruning** — conversation summaries older than 90 days are automatically cleared

See [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) for the full technical deep-dive.

## Configuration

Configuration is resolved in this order: **CLI flags > environment variables > config file > defaults**. You can use any combination.

### Config File

Create `memg.json` in your working directory, or `~/.memg/config.json` for global settings:

```json
{
  "port": 8787,
  "target": "https://api.openai.com",
  "entity": "",
  "db": "~/.memg/memory.db",

  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini"
  },

  "embed": {
    "provider": "openai",
    "model": "text-embedding-3-small"
  },

  "recall": {
    "limit": 100,
    "threshold": 0.10,
    "summary_limit": 5,
    "summary_threshold": 0.30
  },

  "session": {
    "timeout": "30m"
  },

  "working_memory": {
    "turns": 20
  },

  "memory": {
    "token_budget": 4000,
    "summary_budget": 1000
  },

  "conscious": true,
  "conscious_limit": 10,
  "conscious_cache_ttl": "30s",
  "prune_interval": "5m",
  "debug": false
}
```

Copy `memg.example.json` from the repo as a starting point. The proxy auto-detects config files in this order:
1. `memg.json` (working directory)
2. `memg.config.json` (working directory)
3. `~/.memg/config.json` (home directory)

Or specify explicitly: `memg proxy --config /path/to/config.json`

With a config file, the proxy command becomes just:

```bash
memg proxy
```

### Environment Variables

| Variable | Description |
|---|---|
| `MEMG_LLM_PROVIDER` | LLM provider name (e.g. `openai`, `anthropic`) |
| `MEMG_EMBED_PROVIDER` | Embedding provider name (e.g. `openai`, `local`) |
| `MEMG_RECALL_FACTS_LIMIT` | Max facts per recall query (default: 100) |
| `MEMG_RECALL_EMBEDDINGS_LIMIT` | Max stored embeddings loaded per recall pass (default: 100000) |
| `MEMG_RECALL_THRESHOLD` | Minimum relevance score (default: 0.10) |
| `MEMG_SESSION_TIMEOUT` | Session timeout duration (e.g. `30m`) |
| `MEMG_WORKING_MEMORY_TURNS` | Max recent conversation turns loaded (library default: 10, proxy default: 20) |
| `MEMG_MEMORY_TOKEN_BUDGET` | Total token budget for injected memory (default: 4000) |
| `MEMG_SUMMARY_TOKEN_BUDGET` | Token budget for summary section (default: 1000) |
| `MEMG_MAX_RECALL_CANDIDATES` | Safety cap on facts loaded per recall (library default: 50, proxy default: 500) |
| `MEMG_CONSCIOUS_CACHE_TTL` | Conscious facts cache lifetime (default: `30s`) |
| `MEMG_DEBUG` | Enable debug logging (`1` or `true`) |

Provider-specific API keys use their standard env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

### Go Library Options

**Instance-level options** (shared across all calls):

```go
memg.New(repo,
    memg.WithLLMProvider("openai", llm.ProviderConfig{Model: "gpt-4o"}),
    memg.WithEmbedProvider("local", embed.ProviderConfig{}),
    memg.WithRecallLimit(100),
    memg.WithRecallThreshold(0.10),
    memg.WithSessionTimeout(30 * time.Minute),
    memg.WithPruneInterval(5 * time.Minute),
    memg.WithConsciousMode(true),
    memg.WithConsciousLimit(10),
    memg.WithWorkingMemoryTurns(10),
    memg.WithMemoryTokenBudget(4000),
    memg.WithSummaryTokenBudget(1000),
    memg.WithMaxRecallCandidates(50),
    memg.WithEmbedDimension(384),
    memg.WithDebug(),
)
```

**Per-call options** (vary per request):

```go
// Entity scoping — one instance serves all users
g.Chat(ctx, messages, memg.ForEntity("user-alice"))
g.Chat(ctx, messages, memg.ForEntity("user-bob"))

// Filter recall to specific fact categories
g.Chat(ctx, messages, memg.ForEntity("user-alice"),
    memg.WithFactFilter(store.FactFilter{
        Tags:            []string{"medical"},
        MinSignificance: 7,
    }),
)

// LLM options can be combined with MemG options
g.Chat(ctx, messages, memg.ForEntity("user-alice"), llm.WithModel("gpt-4o-mini"))
```

`WithEntity("user-42")` still works at the instance level for single-user apps. `ForEntity()` overrides it per call for multi-user apps.

Both `Chat()` and `Stream()` accept the same per-call options (`ForEntity`, `WithFactFilter`, plus LLM options).

## Architecture

```
Any app (Python, Node.js, Go, curl, ...)
       │
       │  OPENAI_BASE_URL=http://localhost:8787/v1
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│   MemG Proxy (:8787)                                        │
│                                                              │
│   /v1/chat/completions      → intercept (OpenAI format)     │
│   /v1/messages              → intercept (Anthropic format)  │
│   :generateContent          → intercept (Gemini format)     │
│   :streamGenerateContent    → intercept (Gemini streaming)  │
│   /v1/models, /v1/files,... → pass through untouched        │
│                                                              │
│   ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌─────────────┐ │
│   │ Recall   │  │ Summary  │  │ Augment │  │ Built-in    │ │
│   │ (facts + │  │ Recall   │  │ Pipeline│  │ Extraction  │ │
│   │ summaries│  │          │  │ + Pruner│  │ Stage       │ │
│   │ Kneedle) │  │          │  │         │  │             │ │
│   └────┬─────┘  └────┬─────┘  └────┬────┘  └──────┬──────┘ │
│        │             │             │               │        │
│   ┌────┴─────────────┴─────────────┴───────────────┴──────┐ │
│   │              store.Repository                         │ │
│   │  (facts with type, status, significance, TTL,         │ │
│   │   content key, reinforcement, reference time,         │ │
│   │   conversation summaries with embeddings)             │ │
│   └───────────────────────────────────────────────────────┘ │
└───────────┬─────────────────┬─────────────────┬─────────────┘
       ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
       │Postgres │       │ SQLite  │       │  MySQL  │
       └─────────┘       └─────────┘       └─────────┘

Provider Layer:
  LLM:   OpenAI │ Anthropic │ Gemini │ Ollama │ Azure │ Bedrock │ DeepSeek │ Groq │ Together │ xAI
  Embed: OpenAI │ Gemini │ Ollama │ HuggingFace │ Azure │ Bedrock │ Together │ Cohere │ Voyage │ ONNX(local) │ gRPC(legacy)
```

## Direct Provider Construction

Instead of the registry, you can construct providers directly:

```go
import (
    "memg/embed/openai"
    "memg/llm/anthropic"
)

emb, err := openai.New(embed.ProviderConfig{
    APIKey: "sk-...",
    Model:  "text-embedding-3-large",
})

prov, err := anthropic.New(llm.ProviderConfig{
    APIKey: "sk-ant-...",
    Model:  "claude-sonnet-4-20250514",
})

g, _ := memg.New(repo)
g.SetProvider(prov)
g.SetEmbedder(emb)

// Use ForEntity per call for multi-user support
g.Chat(ctx, messages, memg.ForEntity("user-42"))
```

## Custom Augmentation Stages

MemG extracts knowledge through user-defined stages. The pipeline handles deduplication, reinforcement, and default-filling automatically.

```go
type MyStage struct{}

func (s *MyStage) Name() string { return "my-stage" }

func (s *MyStage) Execute(ctx context.Context, job *augment.Job) (*augment.Extraction, error) {
    // Analyze job.Messages, extract facts
    return &augment.Extraction{
        Facts: []*store.Fact{
            {
                Content:      "User is allergic to peanuts",
                Type:         store.FactTypeIdentity,
                Significance: store.SignificanceHigh,
            },
        },
    }, nil
}

g.AddStage(&MyStage{})
```

The pipeline automatically:
- Fills defaults for unset fields (type=identity, status=current, significance=medium)
- Computes content key and deduplicates against existing facts
- Reinforces existing facts instead of creating duplicates
- Resolves slot conflicts for mutable identity facts (e.g., `slot: "location"` — new value replaces old)
- Calls `ConflictDetector.DetectConflicts()` if the stage implements it
- Reclassifies conflicting identity facts from current to historical
- Drops low-confidence assistant-sourced facts (confidence < 0.7) by default

## Query Transformer

Register an optional query transformer to rewrite follow-up queries into standalone retrieval queries:

```go
type MyTransformer struct {
    llm llm.Provider
}

func (t *MyTransformer) TransformQuery(ctx context.Context, query string, recentHistory []string) (*memory.QueryTransform, error) {
    // Use an LLM or rules to rewrite "what about that?" into
    // "user dietary preferences and restrictions"
    return &memory.QueryTransform{RewrittenQuery: rewritten}, nil
}

g.SetQueryTransformer(&MyTransformer{llm: provider})
```

The transformer runs before the query is embedded, so the retrieval vector matches the rewritten intent rather than the ambiguous follow-up.

## Re-Embedding Facts

When you change your embedding model, existing facts become invisible (dimension mismatch). Re-embed all facts for an entity:

```go
updated, err := g.ReEmbedFacts(ctx, "user-alice")
// updated = number of facts re-embedded
```

This processes facts in batches of 50 and updates each fact's vector and `embedding_model` field in place.

## Consolidator

The background consolidator clusters old event facts into pattern facts:

```go
consolidator := memory.NewConsolidator(repo, provider, embedder, 24*time.Hour)
consolidator.Start()
defer consolidator.Stop()

// Or trigger for a specific entity on demand:
err := consolidator.ConsolidateEntity(ctx, entityUUID)
```

It finds event facts older than 30 days, groups them by tag, asks the LLM to summarize each cluster into a behavioral pattern, and marks originals as historical. Disabled by default (zero interval).
