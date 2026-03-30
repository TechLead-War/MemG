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

## Advanced Memory Features

MemG includes six advanced subsystems that go beyond basic memory storage to deliver sharper recall, reduced token usage, and psychology-backed user retention. See [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) for the full technical deep-dive.

### Hierarchical Memory (Subsystem 8)

Three-tier memory inspired by the Atkinson-Shiffrin model and research from Memoria (2025), MemGPT (2023), and the Lost in the Middle phenomenon (Liu et al., 2023):

| Tier | What It Holds | Token Budget | Persistence |
|---|---|---|---|
| **Semantic** | Identity facts, user profile, pinned facts | ~600 tokens | Never decays |
| **Episodic** | Events, predictions, emotionally weighted memories | ~1400 tokens | Ebbinghaus decay curve |
| **Working** | Current session turns (compressed) | ~2000 tokens | Session-scoped |

Projected impact: **60-75% token reduction** vs. flat context injection, with better answer quality due to principled budget allocation and positional optimization (identity at start, recent context at end).

### Relational Memory Graph (Subsystem 9)

Promotes the existing `graph/` package to a first-class recall mechanism. Built on research from HippoRAG (NeurIPS 2024), AriGraph (IJCAI 2025), and A-MEM (NeurIPS 2025):

- **Graph-augmented recall** — seed facts expand to connected subgraphs via 1-hop traversal
- **Entity resolution** — "mom", "mother", "Priya" merge into one node using embedding similarity
- **Knowledge cards** — structured entity summaries injected instead of disconnected fact snippets
- **Anti-hallucination** — graph provides closed-world assumption for personal facts

**SDK Interface:**

```go
// Go — query the knowledge graph for an entity
graph, err := m.GetEntityGraph(ctx, entityID, "Priya")
// Returns: []Triple — all triples where "Priya" appears as subject or object

// Go — graph-augmented recall is automatic when GraphRecall is enabled
ctx := memory.RecallAndBuildContext(ctx, repo, embedder, engine, entityUUID, query, cfg)
// The pipeline checks for TripleStore and expands if available
```

```typescript
// TypeScript — query the knowledge graph
const graph = await memg.getEntityGraph(entityId, "Priya");
// Returns: Triple[] — all triples involving "Priya"

// TypeScript — graph-augmented context building
const context = await memg.buildMemoryContext(entityId, query, {
  graphRecall: true,
  graphExpansionHops: 1,
  graphProximityBonus: 0.05,
  entityResolutionThreshold: 0.85
});
// Graph expansion happens internally when the store supports it
```

```python
# Python — query the knowledge graph
graph = memg.get_entity_graph(entity_id, "Priya")
# Returns: list[Triple] — all triples involving "Priya"

# Python — graph-augmented context building
context = memg.build_memory_context(
    entity_id=entity_id,
    query=query,
    graph_recall=True,
    graph_expansion_hops=1,
    graph_proximity_bonus=0.05,
    entity_resolution_threshold=0.85
)
```

### Emotional Memory Scoring (Subsystem 10)

Emotional annotation on facts based on flashbulb memory research (Brown & Kulik, 1977) and the peak-end rule (Kahneman et al., 1993):

- Three new fact fields: `emotional_valence`, `emotional_arousal`, `emotional_category`
- High-arousal memories decay slower (flashbulb effect)
- Emotional relevance matching boosts recall for emotionally similar queries
- Peak moment detection identifies the most impactful conversations
- Empathetic annotations mark sensitive facts for careful handling

**SDK Interface:**

```go
// Go — retrieve peak moments for an entity
peaks, err := m.GetPeakMoments(ctx, entityID, memory.PeakMomentFilter{
    MinScore:  7.0,
    Limit:     10,
    SinceDate: time.Now().AddDate(-1, 0, 0), // last year
})
// Returns: []PeakMoment — conversations flagged as peaks, with summaries and scores

// Go — emotionally-aware context building (automatic when EmotionalScoring enabled)
ctx := memory.RecallAndBuildContext(ctx, repo, embedder, engine, entityUUID, query, cfg)
// Emotional metadata on recalled facts is used for:
//   - Ebbinghaus decay modulation (emotional_weight)
//   - Emotional relevance matching (category/valence bonus)
//   - Empathetic annotation ([Emotionally sensitive] tags)
```

```typescript
// TypeScript — retrieve peak moments
const peaks = await memg.getPeakMoments(entityId, {
  minScore: 7.0,
  limit: 10,
  sinceDate: new Date('2025-03-29')
});
// Returns: PeakMoment[] — { conversationId, summary, peakScore, date, emotionalCategories }

// TypeScript — emotionally-aware context building
const context = await memg.buildMemoryContext(entityId, query, {
  emotionalScoring: true,
  emotionalRecallBoost: 0.03,
  empatheticAnnotation: true
});
// Emotional metadata is applied automatically during recall and context assembly
```

```python
# Python — retrieve peak moments
peaks = memg.get_peak_moments(
    entity_id=entity_id,
    min_score=7.0,
    limit=10,
    since_date=datetime(2025, 3, 29)
)
# Returns: list[PeakMoment] — conversations with peak_score >= threshold

# Python — emotionally-aware context building
context = memg.build_memory_context(
    entity_id=entity_id,
    query=query,
    emotional_scoring=True,
    emotional_recall_boost=0.03,
    empathetic_annotation=True
)
# Emotional metadata applied during recall scoring and context injection
```

### Proactive Memory Surfacing (Subsystem 11)

Context surfacing without a user query, based on variable ratio reinforcement (Skinner), the Zeigarnik effect, and nostalgia research (Santini et al., 2023):

| Trigger | What It Does | Frequency |
|---|---|---|
| Prediction follow-up | Revisits predictions whose dates have passed | High (70% base rate) |
| Emotional callback | Checks in on emotionally significant past conversations | Moderate (30%) |
| Milestone | Acknowledges conversation count milestones | Always (rare events) |
| Nostalgia | References early conversations and growth | Low (15%) |
| Pattern insight | Surfaces statistical patterns across conversations | Low (20%) |

Triggers fire on a **variable schedule** — unpredictable timing creates the dopamine response that drives habit formation (Nir Eyal's Hook Model).

**SDK Interface:**

```go
// Go — evaluate proactive triggers at session start
func (m *MemG) GetProactiveContext(ctx context.Context, entityID string, opts ...ProactiveOption) (*ProactiveResult, error)

// ProactiveResult contains the output of a proactive evaluation.
type ProactiveResult struct {
    Triggered bool              // Whether a trigger fired
    Type      string            // Trigger type that fired (e.g. "milestone", "prediction_followup")
    Context   string            // Formatted context string for injection
    Metadata  map[string]any    // Trigger-specific metadata (e.g. session_count, prediction content)
}

// Go — register a prediction for future follow-up (alternative to extraction-stage tagging)
func (m *MemG) TrackPrediction(ctx context.Context, entityID string, p Prediction) error

type Prediction struct {
    Content              string     // What was predicted: "Career improvement by end of March"
    TargetDate           time.Time  // When the prediction should be evaluated
    SourceConversationID string     // Which conversation produced this prediction
}
```

```typescript
// TypeScript — evaluate proactive triggers
const proactive = await memg.getProactiveContext(entityId, {
  trigger: 'session_start',
  currentDate: new Date(),
  sessionCount: 47
});
// Returns: { triggered: boolean, type: string, context: string, metadata: {...} } | null

// TypeScript — register a prediction
await memg.trackPrediction(entityId, {
  content: "Career improvement by end of March",
  targetDate: new Date('2026-03-31'),
  sourceConversationId: 'conv-uuid'
});

// TypeScript — include proactive context in unified recall
const context = await memg.buildMemoryContext(entityId, query, {
  proactiveRecall: true,
  sessionCount: 47,
  currentDate: new Date()
});
```

```python
# Python — evaluate proactive triggers
proactive = await memg.get_proactive_context(entity_id,
    trigger="session_start",
    current_date=datetime.now(),
    session_count=47
)
# Returns: ProactiveResult(triggered=True, type="milestone", context="...", metadata={...}) or None

# Python — register a prediction
await memg.track_prediction(entity_id,
    content="Career improvement by end of March",
    target_date=datetime(2026, 3, 31),
    source_conversation_id="conv-uuid"
)

# Python — include proactive context in unified recall
context = await memg.build_memory_context(entity_id, query,
    proactive_recall=True,
    session_count=47,
    current_date=datetime.now()
)
```

### Confidence-Gated Generation (Subsystem 12)

Anti-hallucination through confidence tiers, informed by Chain-of-Verification (ACL 2024), SelfCheckGPT (EMNLP 2023), and the Barnum/Forer effect:

| Tier | Confidence | LLM Instruction |
|---|---|---|
| **Verified** | >= 0.8 | State as known fact |
| **Inferred** | 0.5 -- 0.79 | Frame as "it seems like..." |
| **Uncertain** | < 0.5 | Never state as fact; ask to confirm |

Key insight for astrology/tarot: the Barnum effect means vague readings are accepted — but getting specific remembered facts wrong destroys trust. Confidence gating ensures the AI is **vague by creative choice, not by confusion**.

**SDK Interface:**

```typescript
// TypeScript SDK
const context = await memg.buildMemoryContext(entityId, query, {
  confidenceLabeling: true,
  confidenceFloor: 0.5,
  sourceProvenance: true,
  confidenceVerifiedThreshold: 0.8,
  confidenceInferredThreshold: 0.5
});
// Each fact in the result includes:
// { content, confidence, tier, sourceRole, lastReinforced, reinforcedCount }

// Promote fact confidence via user confirmation
await memg.confirmFact(entityId, factId);
// Sets confidence to 1.0, source_role to "user"

// Replace a wrong fact with the correct version
await memg.correctFact(entityId, factId, "User does not have a sister");
// Old fact reclassified to historical
// New fact created with confidence 1.0, source_role "user"
```

```python
# Python SDK
context = memg.build_memory_context(
    entity_id=entity_id,
    query=query,
    confidence_labeling=True,
    confidence_floor=0.5,
    source_provenance=True
)
# context.facts -> list of facts with .confidence, .tier, .source_role

memg.confirm_fact(entity_id, fact_id)
memg.correct_fact(entity_id, fact_id, "User does not have a sister")
```

```go
// Go library
ctx := memory.RecallAndBuildContext(ctx, repo, embedder, engine, entityUUID, query, memory.RecallConfig{
    ConfidenceLabeling: true,
    ConfidenceFloor:    0.5,
    SourceProvenance:   true,
})

// Confirm a fact
memory.ConfirmFact(ctx, repo, entityUUID, factUUID)

// Correct a fact
memory.CorrectFact(ctx, repo, embedder, entityUUID, factUUID, "User does not have a sister")
```

### User-Visible Memory (Subsystem 13)

User-facing memory operations based on the IKEA effect (Norton et al., 2012), endowment effect (Thaler, 1980), and commitment/consistency (Cialdini, 1984):

```typescript
// Users can see, correct, confirm, pin, and contribute to their memory
await memg.list(entityId, { userVisible: true });     // "What do you know about me?"
await memg.correct(entityId, factId, { newContent }); // Fix wrong facts
await memg.confirm(entityId, factId);                  // Verify correct facts
await memg.pin(entityId, factId);                      // Prevent decay
await memg.addUserNote(entityId, { content, tag });   // Tell the AI something
await memg.exportMemory(entityId);                     // Data portability
```

The retention loop: extraction -> visibility (endowment) -> correction (IKEA effect) -> investment (commitment) -> accurate recall (reciprocity) -> deeper bond -> return.

**SDK Interface (Complete):**

```typescript
// TypeScript SDK

// List memory profile
const profile = await memg.list(entityId, { userVisible: true, groupBy: 'category' });

// Correct a fact
await memg.correct(entityId, factId, { newContent: "Works at Infosys" });

// Confirm a fact (boost confidence to 1.0, reinforce)
await memg.confirm(entityId, factId);

// Pin a fact (never decays, always in semantic tier)
await memg.pin(entityId, factId);

// Unpin a fact (restore original significance and TTL)
await memg.unpin(entityId, factId);

// Add user note (bypass extraction pipeline)
await memg.addUserNote(entityId, { content: "...", tag: "work" });

// Deny a fact (delete and prevent re-extraction)
await memg.deny(entityId, factId);

// Delete a fact (permanent removal, no deny list entry)
await memg.delete(entityId, factId);

// Export all memories
const exported = await memg.exportMemory(entityId);
// Returns: { facts: [...], summaries: [...], graph: [...], preferences: {...}, metadata: {...} }

// Configure extraction preferences
await memg.setExtractionPreferences(entityId, {
  excludeTags: ['financial'],
  patternVisibility: false
});
```

```python
# Python SDK

profile = memg.list_memory(entity_id, user_visible=True, group_by="category")

memg.correct_fact(entity_id, fact_id, new_content="Works at Infosys")

memg.confirm_fact(entity_id, fact_id)

memg.pin_fact(entity_id, fact_id)

memg.unpin_fact(entity_id, fact_id)

memg.add_user_note(entity_id, content="...", tag="work")

memg.deny_fact(entity_id, fact_id)

memg.delete_fact(entity_id, fact_id)

exported = memg.export_memory(entity_id)

memg.set_extraction_preferences(entity_id, exclude_tags=["financial"], pattern_visibility=False)
```

```go
// Go library

profile, err := m.ListMemoryProfile(ctx, entityUUID, memg.ProfileOptions{
    UserVisible: true,
    GroupBy:     "category",
})

err = m.CorrectFact(ctx, entityUUID, factUUID, "Works at Infosys")

err = m.ConfirmFact(ctx, entityUUID, factUUID)

err = m.PinFact(ctx, entityUUID, factUUID)

err = m.UnpinFact(ctx, entityUUID, factUUID)

err = m.AddUserNote(ctx, entityUUID, store.UserNote{Content: "...", Tag: "work"})

err = m.DenyFact(ctx, entityUUID, factUUID)

err = m.DeleteFact(ctx, entityUUID, factUUID)

exported, err := m.ExportMemory(ctx, entityUUID)

err = m.SetExtractionPreferences(ctx, entityUUID, store.ExtractionPreferences{
    ExcludeTags:       []string{"financial"},
    PatternVisibility: false,
})
```

---

### Advanced Memory Features (v2) — TypeScript SDK

The TypeScript SDK (`memg-core-js`) implements the advanced memory subsystems as a self-contained, in-process engine. This section documents the concrete TypeScript APIs, data model extensions, and behavioral details for each feature. All implementations described below are verified against the current source code.

| Feature | Description | API |
|---|---|---|
| Hierarchical Context | Three-tier memory (working/episodic/semantic) with structured sections | `buildHierarchicalMemoryContext()` |
| Emotional Memory | Tracks emotional weight and valence of facts | Automatic in extraction |
| Confidence Grading | Facts labeled verified/likely/inferred, LLM uses hedging | `confidenceFloor` option |
| Open Thread Tracking | Detects and tracks unresolved life situations | `getOpenThreads()`, `resolveThread()` |
| Verbatim Store | Stores user's exact words for self-reference mirroring | Automatic in extraction |
| Temporal Memory | Tracks when states began, not just when discussed | `startedAt` field |
| Proactive Surfacing | Re-engagement triggers (Zeigarnik, nostalgia, milestones) | `getProactiveContext()` |
| Segment Extraction | Topic-level extraction instead of turn-level | `extractFromSegments()` |
| Pin & Confirm | User-visible memory management | `pin()`, `confirm()` |
| Personalization Throttle | Prevents over-personalization | `maxPersonalFacts` config |

---

#### Hierarchical Memory Architecture

**Problem solved:** The flat context builder packs all memories into a single token budget with no structural separation. Identity facts compete with episodic events and session context for the same budget space. The LLM receives an unstructured list and must infer which memories are permanent identity, which are situational recall, and which are current conversation state.

**Design:** Three tiers partition the memory budget, each with distinct retrieval strategies and injection order. The `buildHierarchicalContext()` function replaces the flat `buildContext()` for applications that want structured memory injection.

**Tiers:**

| Tier | Content | Retrieval | Budget Config |
|---|---|---|---|
| **Semantic** (always-on) | Identity facts, high-significance patterns, pinned facts | Filtered DB query — no embedding search | `semanticBudget` in `HierarchicalContextOptions` |
| **Episodic** (query-dependent) | Recalled events, predictions, contextual patterns | Hybrid vector + BM25 search | `episodicBudget` in `HierarchicalContextOptions` |
| **Working** (session-scoped) | Turn summaries from the current conversation | Loaded from session state | `workingBudget` in `HierarchicalContextOptions` |

**Structured output format:** The context string is assembled in section order, with each section labeled:

1. `[IDENTITY]` — Semantic tier. Who this user is. Each fact annotated with confidence grade (`verified`, `likely`, `inferred`). Verbatim quotes included when available.
2. `[EMOTIONAL STATE]` — Recent emotional context, sorted by recency. Each entry shows relative time and confidence level. Verbatim user words included inline.
3. `[OPEN THREADS]` — Unresolved topics from the Zeigarnik tracker. Shows tag, content, and duration since thread opened.
4. `[RECALLED CONTEXT -- VERIFIED]` — Episodic facts with confidence >= 0.8. Facts the user explicitly stated.
5. `[RECALLED CONTEXT -- INFERRED]` — Episodic facts with confidence 0.5-0.79, prefixed with hedging language ("May be", "Possibly").
6. `[PROACTIVE CONTEXT]` — Proactive surfacing items. Labeled by trigger type.
7. `[SESSION CONTEXT]` — Working memory tier. Current conversation turn summaries.
8. `[PAST CONVERSATIONS]` — Conversation summaries from previous sessions, with relative dates.

**TypeScript API:**

```typescript
// Build hierarchical context for prompt injection.
const ctx: HierarchicalContext = await memg.buildHierarchicalMemoryContext(
  entityId: string,
  queryText: string,
  opts?: HierarchicalContextOptions
);

interface HierarchicalContextOptions {
  workingBudget?: number;    // Max tokens for working memory tier
  episodicBudget?: number;   // Max tokens for episodic tier
  semanticBudget?: number;   // Max tokens for semantic tier
  includeProactive?: boolean; // Include proactive surfacing (default: true)
  includeEmotional?: boolean; // Include emotional state (default: true)
  confidenceFloor?: number;  // Exclude facts below this confidence
  maxAgeDays?: number;       // Maximum fact age to include (days)
}

interface HierarchicalContext {
  working: string;     // Session context tier text
  episodic: string;    // Recalled context tier text
  semantic: string;    // Identity tier text
  proactive: string;   // Proactive surfacing text
  emotional: string;   // Emotional state text
  totalTokens: number; // Tokens used across all tiers
  formatted: string;   // The full assembled context string
}
```

The `formatted` field contains the complete context string ready for system prompt injection. Individual tier fields are available for applications that need custom assembly.

**Token budgets:** Each tier has an independent budget. The builder tracks `tokensUsed` globally and respects the overall `totalTokens` ceiling. If a tier overflows, facts are truncated by significance. Summary sections use a dedicated sub-budget that is also capped by remaining total budget.

**Injection order rationale:** The ordering places identity first and session context last, exploiting the U-shaped attention curve documented in "Lost in the Middle" (Liu et al., 2023) — LLMs attend most strongly to the beginning and end of context. Identity (always relevant) occupies the high-attention start position; working memory (most recently relevant) occupies the high-attention end position; episodic memories occupy the middle where attention is weakest but semantic relevance compensates.

**Example usage:**

```typescript
const m = new MemG({ dbPath: './memory.db', openaiApiKey: process.env.OPENAI_API_KEY });
await m.init();

// Build structured, confidence-graded memory context
const ctx = await m.buildHierarchicalMemoryContext('user-123', 'How is my career looking?', {
  includeProactive: true,
  includeEmotional: true,
  confidenceFloor: 0.5,
});
console.log(ctx.formatted);
// Output:
// [IDENTITY] Who this user is:
// - Works at TCS (verified)
// - Lives in Mumbai (verified)
//
// [EMOTIONAL STATE] Recent emotional context:
// - Feeling anxious about career change (3 days ago)
//   User said: "I feel stuck in this job"
//
// [OPEN THREADS] Unresolved topics to follow up on:
// - Considering department transfer (still open, started 2 weeks ago)
// ...

// Individual tier text is also available:
console.log(ctx.semantic);   // Identity section only
console.log(ctx.emotional);  // Emotional section only
console.log(ctx.totalTokens); // Token count across all tiers
```

---

#### Emotional Memory Scoring

**Problem solved:** Facts carry informational significance but not emotional significance. "Father passed away" and "Got promoted" both score high on significance but require fundamentally different recall, decay, and interaction behavior. Without emotional metadata, the system treats grief and celebration identically.

**New Fact fields:**

| Field | Type | Range | Purpose |
|---|---|---|---|
| `emotionalWeight` | `number` | 0.0-1.0 | Intensity of emotional significance. 0.0 = neutral/factual, 1.0 = deeply emotional |
| `emotionalValence` | `string` | 9 categories | Dominant emotion category |

**Valid valence categories:** `grief`, `joy`, `anxiety`, `hope`, `love`, `anger`, `fear`, `pride`, `neutral`.

**Extraction:** The extraction prompt (`buildExtractionPrompt()` in `extract.ts`) instructs the LLM to output `emotional_weight` (0.0-1.0) and `emotional_valence` (one of the 9 categories) for each extracted fact. Validation clamps weight to [0, 1] and rejects valences not in the allowed set. Both fields default to `null` when not applicable.

**Search engine boost:** The `HybridEngine.rank()` method applies an additive emotional boost to the hybrid score:

```
score += emotionalWeight * emotionalBoost
```

Default `emotionalBoost` is `0.05`. A fact with `emotionalWeight = 1.0` receives a +0.05 score boost — enough to break ties and surface emotionally significant facts in borderline relevance cases, but not enough to override genuine semantic irrelevance.

**Context builder:** Emotional facts are loaded separately via a `listFactsFiltered` call with `emotionalValences` filter (excluding `neutral`). They appear in the `[EMOTIONAL STATE]` section of hierarchical context, sorted by recency, with relative timestamps and confidence annotations.

---

#### Confidence-Gated Generation

**Problem solved:** All recalled facts are injected with equal authority regardless of extraction confidence. A fact the user explicitly stated (`confidence: 1.0`) and a fact the LLM inferred from context (`confidence: 0.4`) appear side by side in the prompt with no distinction.

**Confidence tiers:**

| Tier | Confidence Range | Context Label | LLM Behavior |
|---|---|---|---|
| Verified | >= 0.8 | `[RECALLED CONTEXT -- VERIFIED]` | State directly: "You mentioned X" |
| Likely | 0.5 - 0.79 | Prefixed with "May be" | Hedge: "I think you mentioned X" |
| Inferred | < 0.5 | Prefixed with "Possibly" | Speculate: "You might have mentioned X" |

**`confidenceFloor` option:** The `HierarchicalContextOptions.confidenceFloor` parameter (default: 0.3) excludes all facts below the threshold from context injection. This prevents very-low-confidence guesses from reaching the LLM at all.

**Search engine confidence adjustment:** The hybrid ranking engine applies a confidence-based score multiplier:

```
if (confidence < 1.0) {
  score *= 0.95 + 0.05 * confidence;
}
```

A fact with `confidence: 0.5` retains 97.5% of its raw score. A fact with `confidence: 0.0` retains 95%. This is a gentle tiebreaker, not a hard filter — low-confidence facts that are highly semantically relevant still surface.

**Context builder behavior:** The `buildHierarchicalContext()` function splits recalled facts into `verified` (>= 0.8) and `inferred` (< 0.8) groups. Verified facts are rendered as direct statements with their confidence score. Inferred facts are rendered with hedging prefixes ("May be", "Possibly") and their confidence score, instructing the LLM to use appropriately tentative language.

---

#### Open Thread Tracking (Zeigarnik Memory)

**Problem solved:** Users mention unresolved situations — pending decisions, ongoing health issues, awaited results — that create psychological tension. The system has no mechanism to track which situations remain open and which have been resolved, so it cannot proactively follow up or surface ongoing concerns.

**New Fact field:**

| Field | Type | Values | Purpose |
|---|---|---|---|
| `threadStatus` | `string \| null` | `'open'`, `'resolved'`, `null` | Tracks whether a situation is unresolved |

**Extraction:** The extraction prompt instructs the LLM to set `thread_status: "open"` for unresolved situations, pending decisions, and open questions. Validation normalizes the value to `'open'` or `null` — any value other than `'open'` is treated as no thread.

**Search engine boost:** Open threads receive a fixed +0.03 additive score boost in hybrid ranking, ensuring they surface slightly more readily than closed facts at equal relevance.

**TypeScript API:**

```typescript
// List all open threads for an entity.
async getOpenThreads(
  entityId: string,
  limit?: number    // default: 20
): Promise<Memory[]>

// Mark a thread as resolved.
async resolveThread(
  entityId: string,
  memoryId: string
): Promise<boolean>
```

`getOpenThreads()` queries the store's `listOpenThreads()` method, which filters `mg_entity_fact` rows where `thread_status = 'open'`. Returns `Memory` objects with full metadata including `threadStatus`, `verbatim`, and `emotionalValence`.

`resolveThread()` sets the fact's `thread_status` to `'resolved'` via `store.updateThreadStatus()`. This removes it from the open threads list and from the `[OPEN THREADS]` section of hierarchical context.

**Context builder:** Open threads appear in the `[OPEN THREADS]` section with their tag, content, and duration since opened (computed from `startedAt` when available). Example output:

```
[OPEN THREADS] Unresolved topics to follow up on:
- Relationship: User is considering breaking up with partner (still open, started 2 weeks ago)
- Work: User awaiting results of job interview (still open, started 5 days ago)
```

---

#### Verbatim Store (Self-Reference Mirroring)

**Problem solved:** Extracted facts are paraphrased summaries of what the user said: "User is grieving" instead of "I can't stop crying about my dad." The paraphrase loses the user's voice, emotional texture, and the specific phrasing that makes recalled memory feel personal rather than clinical.

**New Fact field:**

| Field | Type | Constraints | Purpose |
|---|---|---|---|
| `verbatim` | `string \| null` | Max 300 chars, only when confidence >= 0.8 | User's exact words that led to this fact |

**Extraction:** The extraction prompt instructs: *"verbatim = the user's EXACT words that led to this fact, quoted verbatim. Only include when confidence >= 0.8. null otherwise."* Validation trims whitespace and caps at 300 characters. Empty strings are normalized to `null`.

**Context builder:** When a fact has a `verbatim` field, the hierarchical context builder appends it inline:

```
- User's father recently passed away (verified) | User said: "I can't stop crying about my dad"
```

This appears in both the `[IDENTITY]` section (for conscious facts) and the `[RECALLED CONTEXT -- VERIFIED]` section (for recalled facts). The quoted format signals to the LLM that these are the user's actual words, enabling more empathetic and personalized responses that mirror the user's own language.

---

#### Temporal Semantic Memory

**Problem solved:** Facts record *when they were extracted* (`createdAt`) but not *when the described state began*. A user who says "I've been dealing with anxiety for about three weeks" produces a fact with `createdAt = today`, but the anxiety started three weeks ago. The system has no way to represent or surface this temporal span.

**New Fact field:**

| Field | Type | Format | Purpose |
|---|---|---|---|
| `startedAt` | `string \| null` | ISO 8601 date (YYYY-MM-DD) | When the state or situation described by this fact began |

**Extraction:** The extraction prompt instructs the LLM to compute `started_at` from relative language: *"For relative references like 'for 3 weeks', compute the date: today minus the duration."* Validation ensures ISO date format and rejects unparseable values.

**Context builder:** When `startedAt` is present, the `[OPEN THREADS]` section displays temporal duration:

```
- Health: User experiencing anxiety (still open, started 3 weeks ago)
```

---

#### Proactive Memory Surfacing

**Problem solved:** The memory system is purely reactive — it only recalls facts when the user's query is semantically close. It never proactively surfaces relevant information: it does not follow up on unresolved situations, check in after emotional disclosures, celebrate milestones, or reference meaningful old memories.

**Design:** The `getProactiveContext()` function generates re-engagement triggers by scanning the fact store for specific patterns. Five trigger types are implemented:

| Trigger Type | What It Detects | Priority | Example |
|---|---|---|---|
| `open_thread` | Unresolved situations, older = higher priority | 0.1-1.3 (scaled by significance + age) | "You mentioned considering breaking up — how is that going?" |
| `emotional_checkin` | Negative emotional facts from 1-14 days ago | 0.0-0.5 (scaled by emotional weight) | "You mentioned experiencing grief 5 days ago. How are you feeling about that now?" |
| `milestone` | Fact count milestones (50, 100, 250, 500, 1000) or day anniversaries (30, 90, 180, 365) | 0.8 | "This is a milestone — 100 memories stored together." |
| `nostalgia` | High-significance facts older than 30 days, not recalled in 14+ days | 0.4 | "Remember when you mentioned: 'I got the job!' That was 45 days ago." |
| `prediction_followup` | Prediction-tagged facts aged 7-30 days with open/unset thread status | 0.6 | "You made a prediction 12 days ago: 'I think I'll get the promotion.' Has anything changed?" |

**Variable ratio reinforcement:** To avoid predictability, proactive surfacing uses a hash-based probability gate:

```typescript
const dayHash = now.getDate() + entityUuid.charCodeAt(0);
const shouldSurface = (dayHash % 3) !== 0; // ~66% chance
```

This implements variable ratio reinforcement (Skinner): unpredictable reward timing produces the highest engagement rates. The system surfaces proactive context roughly two-thirds of the time, making it feel natural rather than mechanical.

**TypeScript API:**

```typescript
// Get proactive surfacing items for an entity.
async getProactiveContext(
  entityId: string,
  opts?: { trigger?: string; limit?: number }
): Promise<ProactiveContext[]>

interface ProactiveContext {
  type: 'open_thread' | 'emotional_checkin' | 'milestone' | 'nostalgia' | 'prediction_followup';
  content: string;         // Human-readable prompt for the LLM
  sourceFactId: string;    // UUID of the triggering fact
  priority: number;        // Sorting weight (higher = more important)
  daysSince?: number;      // Days since the source fact was created
}
```

Results are sorted by priority descending and capped at `limit` (default: 3). The `content` field is ready for injection into the `[PROACTIVE CONTEXT]` section of hierarchical context.

**Example usage:**

```typescript
// Get re-engagement triggers for session start
const proactive = await m.getProactiveContext('user-123', { trigger: 'all', limit: 3 });
for (const item of proactive) {
  console.log(`[${item.type}] ${item.content}`);
}
// [open_thread] Considering department transfer (open since 5 days ago)
// [emotional_checkin] You mentioned experiencing anxiety 3 days ago. How are you feeling about that now?
// [milestone] This is a milestone — 100 memories stored together.
```

---

#### Segment-Level Extraction

**Problem solved:** The standard extraction pipeline processes the entire conversation turn as a single unit. For long conversations that span multiple topics, this produces two problems: (1) the LLM extraction call receives a long, multi-topic transcript that dilutes attention on any single topic, and (2) trivial segments (greetings, acknowledgments) waste an extraction call.

**Design:** `runSegmentedExtraction()` processes conversation by pre-segmented topic chunks. Each segment carries optional `topic` and `classification` metadata that are prepended to the transcript as context lines before the standard extraction pipeline runs.

**TypeScript API:**

```typescript
// Run extraction over pre-segmented conversation chunks.
async extractFromSegments(
  entityId: string,
  segments: SegmentInput[]
): Promise<void>

interface SegmentInput {
  messages: Array<{ role: string; content: string }>;
  topic?: string;           // e.g., "career", "relationship"
  classification?: string;  // e.g., "QUESTION", "PREDICTION"
}
```

**Behavior:**
1. Each segment is checked independently for triviality via `isTrivialTurn()`. Trivial segments are skipped entirely — no LLM call.
2. For non-trivial segments, the `topic` and `classification` fields are prepended to the first message as context lines (e.g., "Topic: career\nClassification: PREDICTION\n...").
3. The standard `runExtraction()` pipeline handles the rest: LLM call, parse, validate, embed, dedup, slot conflict resolution, TTL assignment, persist.
4. Results are aggregated: total `inserted` and `reinforced` counts across all segments.

**Benefits:**
- **Fewer extraction calls.** Trivial segments (greetings, filler) are skipped without an LLM call.
- **More coherent facts.** Each extraction call receives a single-topic transcript, so the LLM has full topic context without cross-topic interference.
- **Topic-aware extraction.** The prepended topic and classification metadata guide the LLM toward more accurate fact typing and tagging.

---

#### User-Visible Memory -- Pin & Confirm

**New Fact field:**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `pinned` | `boolean` | `false` | Whether the user has pinned this fact |

**TypeScript API:**

```typescript
// Pin a memory — it never decays.
async pin(entityId: string, memoryId: string): Promise<boolean>

// Unpin a memory — re-enable normal decay.
async unpin(entityId: string, memoryId: string): Promise<boolean>

// Confirm a memory is correct — boosts confidence to 1.0, reinforces.
async confirm(entityId: string, memoryId: string): Promise<boolean>
```

**Pin behavior:** `pinFact()` sets `pinned = 1`, `significance = 10`, and `expires_at = NULL` on the fact row. This means:
- The fact never expires (no TTL).
- The fact qualifies for the semantic tier in hierarchical context (significance >= 8).
- The fact is immune to staleness demotion and routine pruning.

**Unpin behavior:** `unpinFact()` sets `pinned = 0`. The fact retains its current significance but is no longer immune to decay. A new TTL is not automatically assigned — the fact remains without expiry until the next reinforcement or manual significance adjustment.

**Confirm behavior:** `confirmFact()` sets `confidence = 1.0`, increments `reinforced_count`, and updates `reinforced_at` to the current timestamp. This promotes an uncertain fact to verified status, affecting how it appears in confidence-gated context (moved from `[INFERRED]` to `[VERIFIED]` section).

**Search engine boost:** Pinned facts receive a fixed +0.10 additive score boost in hybrid ranking (configurable via `pinnedBoost` option). This is the largest single boost in the ranking pipeline — larger than emotional boost (+0.05), open thread boost (+0.03), or engagement boost (max +0.02).

**Example usage:**

```typescript
// List what the AI knows
const memories = await m.list('user-123');

// User confirms a fact is correct — boosts confidence to 1.0
await m.confirm('user-123', memories[0].id);

// Pin a fact so it never decays
await m.pin('user-123', memories[0].id);

// View and resolve open threads
const threads = await m.getOpenThreads('user-123');
console.log(threads); // [{ content: "Considering department transfer", threadStatus: "open", ... }]
await m.resolveThread('user-123', threads[0].id);
```

---

#### Personalization Throttle

**Problem solved:** Without limits, the context builder can over-personalize — flooding the prompt with identity facts, emotional context, and thread tracking until the LLM becomes so focused on the user's history that it loses the ability to be helpful on the current task. Research shows that personalization follows an inverted U-curve: too little feels generic, too much feels invasive or distracting.

**Configuration options:**

| Option | Type | Default | Purpose |
|---|---|---|---|
| `maxPersonalFacts` | `number` | `15` | Caps identity + emotional + thread facts per context build |
| `diversifyTopics` | `boolean` | `true` | No single tag exceeds 40% of recalled facts |
| `freshnessBias` | `number` | `0.3` | Configurable recency preference (0.0-1.0) |

**`maxPersonalFacts`:** The hierarchical context builder tracks a `personalFactCount` counter. Every fact added to the `[IDENTITY]`, `[EMOTIONAL STATE]`, or `[OPEN THREADS]` sections increments this counter. Once it reaches `maxPersonalFacts`, no more personal facts are added to those sections. Recalled context and session context are not subject to this cap — they are query-driven, not profile-driven.

**`diversifyTopics`:** When enabled, the context builder caps any single tag at 40% of the recalled fact set before rendering the `[RECALLED CONTEXT]` section. The search engine also applies diversification: after ranking, if any tag exceeds 40% of results, excess facts in that tag have their score multiplied by 0.85, pushing them down. This prevents a user with 50 "work" facts and 5 "health" facts from having their entire recalled context be about work.

---

#### New Schema Fields

The `mg_entity_fact` table gains 7 new columns in the v2 migration:

| Column | SQL Type | Default | Nullable | Purpose |
|---|---|---|---|---|
| `emotional_weight` | `REAL` | -- | Yes | Emotional intensity (0.0-1.0) |
| `emotional_valence` | `TEXT` | -- | Yes | Emotion category (one of 9 values) |
| `verbatim` | `TEXT` | -- | Yes | User's exact words (max 300 chars) |
| `started_at` | `TEXT` | -- | Yes | ISO date when the state began |
| `thread_status` | `TEXT` | -- | Yes | `'open'` or `'resolved'` |
| `engagement_score` | `REAL` | `0` | No | Topic engagement level (0.0-1.0) |
| `pinned` | `INTEGER` | `0` | No | User-pinned flag (0 or 1) |

**Indexes added:**

| Index | Columns | Purpose |
|---|---|---|
| `idx_mg_fact_thread` | `(entity_id, thread_status)` | Fast open thread queries |
| `idx_mg_fact_pinned` | `(entity_id, pinned)` | Fast pinned fact queries |
| `idx_mg_fact_emotional` | `(entity_id, emotional_valence)` | Fast emotional fact queries |

**Migration strategy:** The schema uses `ALTER TABLE ... ADD COLUMN` statements that are applied idempotently. The SQLite store wraps each migration statement in a try-catch so that columns that already exist do not cause errors. New columns with nullable defaults are backward-compatible — existing rows have `NULL` values for the new fields, which the `rowToFact()` mapper handles with fallback defaults.

**Full Fact type in TypeScript:**

```typescript
interface Fact {
  uuid: string;
  content: string;
  embedding?: number[];
  createdAt?: string;
  updatedAt?: string;
  factType: 'identity' | 'event' | 'pattern';
  temporalStatus: 'current' | 'historical';
  significance: number;
  contentKey: string;
  referenceTime?: string;
  expiresAt?: string;
  reinforcedAt?: string;
  reinforcedCount: number;
  tag: string;
  slot: string;
  confidence: number;
  embeddingModel: string;
  sourceRole: string;
  recallCount: number;
  lastRecalledAt?: string;
  // v2 fields:
  emotionalWeight?: number;
  emotionalValence?: 'grief' | 'joy' | 'anxiety' | 'hope' | 'love' | 'anger' | 'fear' | 'pride' | 'neutral';
  verbatim?: string;
  startedAt?: string;
  threadStatus?: 'open' | 'resolved' | null;
  engagementScore?: number;
  pinned?: boolean;
}
```

---

#### New Store Interface Methods

The `Store` interface (`store.ts`) gains 9 new methods for the v2 features. All methods return `T | Promise<T>` (the `MaybeAsync<T>` pattern) so both synchronous stores (SQLite) and asynchronous stores (Postgres, MySQL) can implement them.

**Pin & Confirm:**

```typescript
// Pin a fact: sets pinned=1, significance=10, expires_at=NULL.
pinFact(factUuid: string): MaybeAsync<void>;

// Unpin a fact: sets pinned=0.
unpinFact(factUuid: string): MaybeAsync<void>;

// Confirm a fact: sets confidence=1.0, increments reinforced_count,
// updates reinforced_at to now.
confirmFact(factUuid: string): MaybeAsync<void>;
```

**Thread Tracking:**

```typescript
// Set thread_status on a fact ('open' or 'resolved').
updateThreadStatus(factUuid: string, status: string): MaybeAsync<void>;

// List facts with thread_status='open' for an entity, ordered by
// significance DESC, created_at ASC. Returns full Fact objects.
listOpenThreads(entityUuid: string, limit: number): MaybeAsync<Fact[]>;
```

**Emotional & Engagement:**

```typescript
// Update emotional weight and valence on a fact.
updateEmotionalWeight(factUuid: string, weight: number, valence: string): MaybeAsync<void>;

// Update engagement score on a fact.
updateEngagementScore(factUuid: string, score: number): MaybeAsync<void>;
```

**Entity Statistics (for Milestones):**

```typescript
// Count total facts for an entity (used for milestone detection).
countEntityFacts(entityUuid: string): MaybeAsync<number>;

// Get the creation date of the oldest fact for an entity (used for
// anniversary milestone detection). Returns ISO date string or null.
getEntityFirstFactDate(entityUuid: string): MaybeAsync<string | null>;
```

All 9 methods are implemented in `MemGStore` (SQLite), `PostgresStore`, and `MySQLStore`. Custom store implementations must add these methods to support v2 features.

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
