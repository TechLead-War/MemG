# MemG — Memory System Overview

This document explains how MemG gives LLMs persistent memory. It covers the key ideas and design decisions at a high level. For implementation details, algorithms, data structures, and production subsystems, see [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md).

---

## The Problem

LLMs are stateless. Every request starts from zero — the model remembers nothing from prior conversations. If a user says "I'm allergic to peanuts" today, the model has no idea tomorrow.

The obvious fix — dump all past conversations into every request — fails because:

- **Context windows are finite.** A few weeks of chat history exceeds even the largest windows.
- **Noise kills quality.** Stuffing irrelevant history into the prompt makes answers worse, not better.
- **Not all knowledge is explicit.** Some facts are patterns observed across conversations, not direct quotes.

MemG solves this with five techniques that work together behind a simple API.

---

## The Five Techniques

```
┌─────────────────────────────────────────────────────────────┐
│                        MemG Memory                          │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────────┐    │
│  │ Long-Term │  │Short-Term │  │     Write Path        │    │
│  │  Recall   │  │  Session  │  │                       │    │
│  │           │  │  History  │  │  Distillation → Facts │    │
│  │ Facts +   │  │           │  │  Summaries → Recaps   │    │
│  │ Summaries │  │  Recent   │  │  Lifecycle → Decay,   │    │
│  │ by        │  │  messages │  │    Reinforce, Evolve  │    │
│  │ relevance │  │  (capped) │  │                       │    │
│  └─────┬─────┘  └─────┬─────┘  └──────────┬───────────┘    │
│        │              │                    │                 │
│        └──────────┬───┘                    │                 │
│                   ▼                        │                 │
│          LLM receives rich,               │                 │
│          relevant context                  │                 │
│                   │                        │                 │
│                   ▼                        │                 │
│            LLM responds ──────────────────►│                 │
│                                    (async extraction)       │
└─────────────────────────────────────────────────────────────┘
```

### 1. Semantic Retrieval — "What do I know that's relevant right now?"

This is the core read path. When the user sends a message, MemG finds the most relevant stored facts — even if the wording is completely different from how they were originally stated.

**How it works conceptually:**

- Every stored fact is converted into a numerical vector (an embedding) that captures its *meaning*, not just its words.
- The user's query is also converted into a vector using the same model.
- MemG finds facts whose vectors point in a similar direction to the query vector — meaning they're semantically related.
- It also does keyword matching (BM25) to catch exact term overlaps that pure semantic search might underweight.
- The two scores are combined. The balance adapts: short queries lean more on keywords, longer queries lean more on meaning.
- A smart cutoff (the Kneedle algorithm) decides how many facts to return based on the score distribution — instead of always returning a fixed number, it stops where scores drop from "clearly relevant" to "noise."

**Why this design:**

- Hybrid search (vectors + keywords) covers each method's blind spots.
- Brute-force search over all facts is used instead of approximate indexes, because per-user fact stores are small enough (hundreds to low thousands) that exact search is both fast and complete.
- Dynamic cutoff means the LLM gets a clean, focused set of facts rather than being flooded with marginally relevant ones.

### 2. Session Tracking — "What just happened in this conversation?"

Within a single conversation, you need continuity. If the user says "My name is Alice" and then asks "What's my name?", the system shouldn't wait for fact extraction — it should remember the conversation.

**How it works conceptually:**

- Messages are grouped into sessions based on activity. A session stays open as long as messages keep flowing (sliding timeout, default 30 minutes).
- The most recent messages (default: 20 turns) are loaded and included in every request, giving the model perfect short-term recall.
- When a session expires due to inactivity, a new one begins as a clean slate. The old session's conversation is summarized in the background (Technique 5).

**Why this design:**

- The sliding timeout creates natural conversation boundaries — you don't carry breakfast recipe context into an afternoon coding session.
- Capping message history prevents unbounded prompt growth while keeping recent context sharp.
- MemG deliberately does not attempt topic segmentation within sessions. Figuring out what "this" refers to when the user switches topics is the LLM's job — MemG is a memory layer, not a reasoning engine.

### 3. Knowledge Distillation — "What should I remember from this conversation?"

Retrieval and sessions are read paths. Distillation is the write path — it creates new knowledge from conversations.

**How it works conceptually:**

- After every LLM response, the conversation turn is sent to an extraction pipeline that runs in the background.
- An LLM analyzes the exchange and extracts structured facts: "User has a dog named Max", "User's dog is a golden retriever", "User recently adopted their dog."
- Each fact is embedded and stored, immediately available for future recall.
- This is asynchronous — the user doesn't wait for extraction. Facts are ready within 1–2 seconds.

**Why this design:**

- Raw messages are noisy and implicit. Distillation produces clean, searchable, atomic pieces of knowledge.
- Extraction stages are user-defined because different applications need fundamentally different strategies — a medical assistant extracts symptoms and medications, a coding assistant extracts tech preferences and architecture decisions. There is no universal extraction prompt.
- Trivial exchanges ("hi", "ok", "thanks") are detected and skipped before calling the extraction LLM, saving cost without losing knowledge.

### 4. Memory Lifecycle — "How should knowledge age, change, and be forgotten?"

Without lifecycle management, every fact is permanent and equal. A lunch from six months ago sits alongside a life-threatening allergy. Old addresses coexist with current ones. The fact store grows without bound.

MemG gives every fact three properties that control how it lives and dies:

**Fact Types — what kind of knowledge is this?**

| Type | What it is | How it evolves |
|---|---|---|
| **Identity** | Enduring truths: "lives in Seattle", "allergic to peanuts" | Replaced — old value becomes historical, new value becomes current |
| **Event** | Point-in-time occurrences: "ate pasta on March 18" | Accumulated — events never supersede each other |
| **Pattern** | Behavioral tendencies: "tends to eat pasta for breakfast" | Strengthened or weakened by evidence over time |

**Temporal Status — is this still true?**

When a user moves from Austin to Seattle, MemG doesn't delete "lives in Austin" — it reclassifies it as *historical*. The system can now answer both "Where do you live?" (Seattle) and "Have you ever lived in Austin?" (yes, previously). Historical facts get a recall penalty but aren't invisible.

**Significance — how important is this?**

| Level | Examples | Lifespan |
|---|---|---|
| High (10) | Allergies, major life events | Never expires |
| Medium (5) | Current book, workplace | 30 days without reinforcement |
| Low (1–4) | Today's lunch, a meeting time | 7 days without reinforcement |

Significance controls *how long* a fact survives, not *how loudly* it surfaces. A peanut allergy doesn't appear in response to "What's the weather?" just because it's high-significance. It still must pass the relevance threshold. Significance determines lifespan, not recall priority.

**Reinforcement — repetition prevents forgetting.**

When a fact is re-extracted (the user mentions it again in a later conversation), its decay timer resets. A fact mentioned in 5+ separate conversations gets promoted to permanent. This mirrors how human memory works — repetition burns things in.

**Decay — forgetting improves quality.**

This is counterintuitive but critical: a smaller, well-pruned fact store produces *better* answers than a larger, noisier one. Expired low-value facts would otherwise fill the retrieval results with marginally relevant noise, drowning out the genuinely useful facts. Forgetting is a recall precision mechanism, not just housekeeping.

### 5. Conversation Summaries — "What happened last time we talked?"

Facts capture atomic knowledge ("budget is $3000", "prefers Airbnb"). But conversations have narrative — decisions made, threads left open, plans in progress. Summaries capture the story that individual facts cannot.

**How it works conceptually:**

- When a session expires and a new one begins, the old session's full conversation is summarized by the LLM.
- The summary is embedded and stored, just like facts.
- During recall, summaries go through the same relevance ranking as facts. They surface only when relevant to the current query.
- Trivial conversations ("What's my email?" → response) produce no summary.
- Summaries are pruned after 90 days — old enough that their narrative context is unlikely to be needed.

**Why both facts and summaries:**

| Question | What answers it |
|---|---|
| "What's my budget?" | A fact: "$3000" |
| "How's my trip planning going?" | A summary: "Planned Japan trip, narrowed to Tokyo, 7 days in April, Airbnb, needs visa, checking flights Friday" |

Facts give precision. Summaries give continuity. Together, the LLM can both answer specific questions and naturally pick up where a previous conversation left off.

---

## How It All Comes Together

Here's what happens during a single request:

```
User: "Can you recommend a restaurant for tonight?"

BEFORE the LLM call:
  1. Embed the query (once, reused for all recall)
  2. Load user profile (top facts by significance — always present)
     → "Allergic to peanuts", "Vegetarian", "Lives in Portland"
  3. Recall relevant facts (semantic + keyword search, dynamic cutoff)
     → "Dislikes loud environments"
  4. Recall relevant summaries (same search, separate layer)
     → [Mar 10] "Discussed wanting to try more plant-based restaurants"
  5. Load recent session messages (last 20 turns)
  6. Assemble everything into the LLM prompt

THE LLM RESPONDS with a personalized recommendation.

AFTER the response (async, in background):
  • Save messages to conversation log
  • Extract new facts from the exchange
  • Track which recalled facts were used
  • (On session expiry: summarize the conversation)
```

Three layers of context — user profile, query-relevant facts, and conversation summaries — ensure the LLM is always informed. The profile provides baseline identity. Recalled facts provide topical relevance. Summaries provide conversational continuity. And the write path continuously learns from every interaction.

---

## Key Design Principles

**Memory, not intelligence.** MemG provides context. It does not resolve pronouns, detect topic shifts, or reason about what the user means. That's the LLM's job.

**Pluggable at every layer.** Storage backends, LLM providers, embedding models, and extraction stages are all swappable interfaces. The core handles lifecycle mechanics; domain-specific decisions belong to the application.

**Small and exact over large and approximate.** Per-user memory is small enough for exact search. No approximate indexes, no external vector databases, no C++ dependencies.

**Forgetting is a feature.** Pruning stale facts directly improves answer quality by keeping the retrieval pool clean.

**Async by default.** Extraction, summarization, recall tracking, and pruning all happen in the background. The user never waits for memory operations.

---

## What's in MEMORY_ARCHITECTURE.md

The deep-dive document covers everything above in full detail, plus:

- **The math** — cosine similarity formula, BM25 scoring, hybrid weight adaptation, Kneedle algorithm mechanics
- **The data model** — complete `Fact` struct with all 20+ fields, defaults, and field-level explanations
- **Polysemy handling** — how contextual embeddings resolve ambiguous words
- **Replacement vs accumulation** — the hard problem of deciding when "I love apple" supersedes "I love mango"
- **Slot-based conflict resolution** — automatic handling of single-valued attributes (location, job, name)
- **Slot normalization** — embedding-based canonical matching so "spouse" and "partner" resolve to the same slot
- **Extraction validation** — filtering empty, oversized, low-confidence, and ungrounded facts
- **Confidence-weighted ranking** — small penalty for uncertain inferences during retrieval
- **Embedding mismatch detection and re-embedding** — handling model changes without silent amnesia
- **Conscious mode** — always-on user profile injection with staleness demotion for mutable facts
- **Fact filtering** — per-call scoping by type, status, tags, significance, and time range
- **Embedding backfill** — healing facts stored during provider outages
- **Context builder internals** — token budgets, cross-component deduplication, priority ordering
- **Runtime efficiency** — fixed-worker pools, bounded queues, metadata-only reads, batch pruning, connection pooling
- **Consolidation** — background clustering of old events into pattern facts
- **Recall usage tracking** — RecallCount / LastRecalledAt feedback signals
- **Concurrency bounds** — semaphore-capped background goroutines, HTTP client reuse, candidate caps
