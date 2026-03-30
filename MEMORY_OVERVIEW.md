# MemG -- Memory Overview

## 1. What is MemG

MemG is a pluggable memory layer for language model applications. It gives LLMs persistent memory across conversations by intercepting API calls, recalling relevant knowledge from stored facts, and asynchronously extracting new knowledge from every interaction. The result is an LLM that remembers who it is talking to, what has been discussed before, and what matters most -- without the developer building any of this from scratch. MemG ships as a zero-code reverse proxy, a native SDK (Go, TypeScript, Python), and an MCP server for agent frameworks.

---

## 2. The Problem

Large language models are stateless. Every request is a blank slate -- the model cannot recall anything from previous conversations. If a user says "I'm allergic to peanuts" today, the model has zero knowledge of that tomorrow.

The naive solution is to dump the entire conversation history into every request. This fails for three reasons:

1. **Context windows are finite.** Models accept a limited number of tokens. A few weeks of conversation history exceeds that limit.
2. **Irrelevant context degrades quality.** Stuffing pages of old conversations into the prompt drowns the model in noise and produces worse answers than if it had no context at all.
3. **Not all knowledge is in conversations.** Some facts are derived -- "User tends to prefer concise answers" is a pattern observed across many conversations, not a quote from any single one.

---

## 3. Three Modes of Operation

**Proxy mode** -- A reverse proxy that sits between your application and the LLM API. It intercepts chat completion requests, recalls relevant facts and conversation summaries, injects them into the system prompt, forwards to the real API, and extracts knowledge from the response in the background. Zero code changes required -- change one environment variable and your existing app gets persistent memory.

**Library mode** -- Native SDKs for Go, TypeScript, and Python that provide programmatic APIs. Call `Chat()` or `Stream()` with a user identifier, and MemG handles recall, context building, and extraction automatically. Direct recall APIs are also available for applications that need fine-grained control over what is retrieved and how it is injected.

**MCP mode** -- Exposes memory operations as JSON-RPC tools that agent frameworks can invoke directly. Any MCP-compatible agent can store facts, recall context, manage sessions, and query the knowledge graph through a standard protocol.

---

## 4. How Memory Flows Through the System

The end-to-end lifecycle of a single interaction:

1. **User sends a message.** MemG receives the request via proxy interception, SDK call, or MCP tool invocation.

2. **MemG embeds the query.** The user's message is converted into a dense vector embedding -- a numeric representation of its meaning. This vector is computed once and reused for all recall passes.

3. **Context is assembled from five layers.** Each layer contributes a different kind of memory:
   - **User profile (semantic memory)** -- The user's most important identity facts (name, location, allergies, job) are loaded unconditionally on every request, regardless of what the user asked. The LLM always knows who it is talking to.
   - **Session context (turn-range summaries)** -- Compressed summaries of earlier turns in the current conversation, providing continuity beyond the raw message window.
   - **Recalled facts (episodic memory)** -- Facts retrieved by semantic similarity and keyword matching against the user's query, filtered by a dynamic cutoff that adapts to the score distribution.
   - **Artifacts** -- Code blocks, JSON, SQL, and other structured outputs from earlier in the conversation, retrievable by description or content.
   - **Conversation summaries** -- Narrative summaries of past sessions, recalled by relevance to the current query.

4. **The LLM responds with full context.** The assembled memory is injected into the system prompt alongside the recent conversation history. The model generates its response informed by everything the system knows.

5. **Background pipeline processes the exchange.** After the response is returned to the user, a unified background pipeline kicks off asynchronously:
   - Atomic facts are extracted from the conversation via an LLM call.
   - Structured outputs (code, JSON, SQL) are detected and stored as artifacts.
   - Entity mentions (proper nouns, technical terms) are accumulated for session-level context.
   - If the conversation has grown beyond the working memory window, older turns are compressed into turn-range summaries.

6. **On session expiry, the full conversation is summarized.** When the user goes idle long enough for the session to expire, the complete conversation is summarized by the LLM and stored with an embedding for future recall.

---

## 5. The Five Core Techniques

### Semantic Retrieval (Long-Term Memory)

Every piece of knowledge is stored as a fact -- a short natural language statement like "User is allergic to peanuts" -- paired with a dense vector embedding that represents its meaning. When the user sends a message, MemG embeds the query and measures how similar it is to every stored fact using cosine similarity (semantic matching) and BM25 (keyword matching). The two scores are combined with adaptive weights that shift based on query length. A dynamic cutoff algorithm (Kneedle) analyzes the shape of the score curve to determine where genuinely relevant results end and noise begins, returning only the facts above that knee. This hybrid approach covers each method's blind spots: semantic matching finds facts with different wording but similar meaning, while keyword matching catches exact term matches that embeddings might underweight.

### Session-Scoped Conversation (Short-Term Memory)

MemG groups messages into sessions using sliding inactivity timeouts. Within an active session, the most recent messages are loaded and prepended to the LLM request, giving the model perfect recall of the current conversation without unbounded history growth. When the user goes idle and the session expires, a new session begins with a clean slate. The old conversation is summarized in the background. This creates natural boundaries between unrelated interactions -- a breakfast recipe discussion at 9am does not pollute a Python coding session at 3pm.

### Knowledge Distillation (Learning)

After every LLM response, the conversation is sent to an extraction pipeline that analyzes it and produces structured facts. "I just adopted a golden retriever named Max" yields facts like "User has a dog named Max" and "User's dog is a golden retriever." Each fact is embedded and stored, becoming immediately available for future recall. Extraction runs asynchronously so the user never waits for it. Over time, the system accumulates a richer body of knowledge with each interaction. The extraction pipeline is pluggable -- different applications can define what knowledge matters and how to extract it.

### Memory Lifecycle (Decay and Reinforcement)

Not all facts are equally durable. Each fact carries a type (identity, event, or pattern), a temporal status (current or historical), and a significance level that determines its time-to-live. High-significance facts like allergies persist indefinitely. Low-significance facts like today's lunch expire in days. When a fact is mentioned again in a later conversation, its decay timer resets and its reinforcement count increments -- repeated mention equals persistence. Facts reinforced five or more times are automatically promoted to permanent status. When new information contradicts old information (the user moves from Austin to Seattle), the old fact is reclassified to historical rather than deleted, preserving the full timeline. Slot-based conflict resolution handles this automatically for single-valued attributes like location and job title.

### Conversation Summaries

When a session expires, the full conversation is summarized by the LLM into a concise narrative capturing what was discussed, what was decided, and what remains pending. These summaries are embedded and recalled by relevance just like facts. They capture narrative arc and conversational flow that individual atomic facts cannot -- "We planned a Japan trip, narrowed to Tokyo for 7 days in April, the user still needs to apply for a visa" is a thread the user can pick up naturally. Summaries older than 90 days are pruned to prevent unbounded growth.

---

## 6. Advanced Subsystems

### Chat Boundaries and Working Memory

Sessions use sliding expiry so continuous conversations are never interrupted. A token-aware adaptive window dynamically sizes the number of raw messages loaded based on remaining budget after other context layers. When conversations grow long, older turns are compressed into immutable turn-range summaries that preserve detail permanently. An overview summary consolidates older summaries, keeping the total summary footprint bounded regardless of conversation length.

### Memory Truth and Provenance

Every fact carries provenance metadata: a semantic slot for conflict resolution, an extraction confidence score, the embedding model that produced its vector, and whether it originated from user or assistant utterances. Slot-based conflict resolution automatically reclassifies superseded facts (the old location becomes historical when a new one arrives). Slot names are normalized via embedding similarity against a canonical registry, ensuring consistent conflict detection even when the LLM uses different words for the same concept.

### Retrieval Correctness

The query is embedded once and reused for all recall passes, halving embedding cost. Facts are loaded in significance order rather than creation order, ensuring important old facts are not crowded out by recent trivia. An optional query transformer rewrites vague follow-up queries into standalone retrieval queries. Embedding dimension mismatches are detected and logged rather than silently returning empty results. A re-embedding operation migrates facts to a new model without data loss.

### Runtime Efficiency

A fixed-worker extraction pool with a bounded queue prevents goroutine pile-up under load. A trivial-turn gate skips extraction for greetings and acknowledgments, saving an LLM call per trivial exchange. Metadata-only reads skip the heavy embedding column when only content and metadata are needed. A per-entity conscious context cache with eager invalidation avoids repeated database queries for profile facts.

### Long-Horizon Memory Hygiene

Recall usage is tracked on every fact (how often it was injected into a prompt and when). Mutable identity facts that have not been reinforced or recalled for months are demoted in conscious mode ranking. A background consolidator clusters old event facts into pattern facts, preventing linear growth. Conversation summaries older than 90 days are pruned.

### Concurrency and Resource Bounds

Background work runs in semaphore-bounded goroutine pools. The proxy reuses a single HTTP client for all upstream calls. Database connection pools have explicit limits. Recall candidate loading is capped to prevent unbounded memory consumption. Slot normalization batches all embedding calls into a single request per extraction job.

### Session Intelligence

Rolling turn-range summaries compress older conversation turns with a two-level structure (individual range summaries plus a consolidated overview). An artifact store detects and indexes code blocks, JSON, and SQL from conversations, making them retrievable when the user says "modify that code" twenty turns later. An entity mention accumulator tracks specific proper nouns and technical terms, augmenting vague queries with session-level context when primary recall returns insufficient results. All background work runs as independent stages within a single unified pipeline per entity.

### Unified Recall Entry Point

A single function orchestrates the full recall pipeline: embedding the query, loading profile facts, recalling relevant facts and summaries, loading turn summaries and artifacts, and assembling the final context string with five-layer priority and token budgeting. Any improvement to the recall pipeline automatically propagates to every caller -- proxy, library, and custom integrations alike.

### Hierarchical Memory Architecture

The memory token budget is partitioned into three tiers modeled after the Atkinson-Shiffrin cognitive framework. Semantic memory (always-on identity facts) occupies the beginning of the prompt where transformer attention is strongest. Episodic memory (query-relevant recalled facts and summaries) fills the middle, ranked by an Ebbinghaus-inspired retention function that models exponential forgetting. Working memory (current session context) occupies the end of the prompt where recency bias is highest. Each tier has an independent budget with surplus cascading to the next tier.

### Relational Memory Graph

A knowledge graph of subject-predicate-object triples links facts through typed relationships, enabling multi-hop reasoning that pure vector search cannot achieve. When the user asks "How is my mother doing?", graph traversal connects "mother Priya" to "ill" to "upcoming surgery" to "worried" -- relationships that span multiple facts with low pairwise similarity. Entity resolution merges variant surface forms ("mom", "mother", "Priya") into canonical entities. Graph expansion augments standard recall by pulling in neighboring facts connected through the graph structure.

### Emotional Memory Scoring

Facts carry emotional metadata: valence (positive/negative/neutral), arousal (high/medium/low), and category (grief, joy, anxiety, hope, and others). High-arousal facts decay more slowly, modeled after the flashbulb memory phenomenon where emotionally intense experiences persist longer. Peak emotional moments in conversations are detected and protected from routine pruning. The context builder annotates emotionally significant facts so the LLM can respond with appropriate sensitivity.

### Proactive Memory Surfacing

Rather than waiting for the user to ask, the system evaluates trigger conditions at the start of each session. It can follow up on predictions whose deadlines have passed, check in on emotional conversations that happened days ago, acknowledge milestones like the 100th session, surface nostalgic references to early interactions, deliver pattern insights the user may not have noticed, and greet returning users with continuity from their last conversation. Triggers fire on a variable schedule to prevent habituation.

### Confidence-Gated Generation

Recalled facts are labeled with three confidence tiers -- VERIFIED, INFERRED, and UNCERTAIN -- based on their extraction confidence and source role. These labels are injected alongside the facts so the LLM can calibrate its language: stating verified facts directly, hedging inferred facts, and flagging uncertain facts as possibilities rather than truths. This prevents the system from confidently asserting something it is not sure about.

### User-Visible Memory and Co-Creation

Users can see their full memory profile, correct wrong facts, confirm inferred ones, pin important facts to ensure they are always recalled, add facts manually, and delete facts they want forgotten. A deny list prevents specific content from being re-extracted. Corrections take precedence over future extraction -- if a user corrects a fact, the pipeline will not silently overwrite it. This turns memory from something that happens to the user into something the user actively builds.

---

## 7. Storage and Providers

MemG supports three storage backends out of the box: **PostgreSQL**, **SQLite**, and **MySQL**. Each backend implements the full repository contract including the knowledge graph triple table, artifact store, and turn summary schema.

The system integrates with **10 LLM providers** (OpenAI, Anthropic, Gemini, Ollama, Azure OpenAI, AWS Bedrock, DeepSeek, Groq, Together AI, xAI) and **11 embedding providers** (OpenAI, Gemini, Ollama, HuggingFace, Azure OpenAI, AWS Bedrock, Together AI, Cohere, VoyageAI, in-process ONNX Runtime, and a local sentence-transformers service). Local embedding options require no API keys.

Recall uses hybrid ranking: cosine similarity for semantic matching combined with BM25 for keyword matching, with adaptive query-length weighting and Kneedle dynamic cutoff.

---

## 8. Research Foundation

The architecture is grounded in established research across three domains:

**Cognitive psychology** -- The three-tier memory hierarchy follows the Atkinson-Shiffrin model (sensory/short-term/long-term stores). Episodic decay uses an Ebbinghaus forgetting curve. Emotional memory scoring draws on flashbulb memory research (Brown and Kulik), the peak-end rule (Kahneman), and emotional memory bias findings (Talarico and Rubin). Proactive surfacing applies variable ratio reinforcement (Skinner), the Zeigarnik effect, and the nostalgia effect. User-visible memory leverages the IKEA effect and endowment effect.

**Information retrieval** -- Fact recall combines dense vector search with Okapi BM25 lexical scoring, a hybrid approach standard in modern search systems. The Kneedle algorithm (Satopaa et al., 2011) provides dynamic result cutoff. Slot normalization uses embedding-based semantic similarity for entity resolution.

**AI memory systems** -- The design draws on MemGPT (OS-inspired tiered memory), MemoryBank (Ebbinghaus decay with emotional modulation), Mem0 (graph memory with grounded generation), HippoRAG (knowledge graph plus Personalized PageRank for multi-hop reasoning), A-MEM (Zettelkasten-style associative indexing), AriGraph (semantic-episodic graph for planning), and Generative Agents (relational knowledge for social reasoning). Context injection order follows findings from Lost in the Middle (Liu et al., 2023) on transformer attention patterns.

For full technical details, formulas, and implementation specifics, see [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md).
