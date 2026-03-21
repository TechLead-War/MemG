# CLAUDE.md

## Purpose
This repository is **MemG**, a pluggable memory layer for LLM applications, written in Go.

Primary goal:
- enable persistent memory for LLM apps via reverse proxy, Go library, and MCP server
- support hybrid recall, fact lifecycle management, and multi-provider integrations
- remain easy to plug into applications with minimal integration effort

Non-goals:
- do not turn this into a general-purpose database or vector store
- do not rewrite the provider registry pattern
- do not introduce frameworks, ORMs, or heavy abstractions over the existing interfaces

## Critical Workflows
1. **Proxy mode** — `memg proxy` intercepts LLM API calls, injects recalled facts, and asynchronously extracts knowledge
2. **Library mode** — `memg.New(repo, opts...)` provides programmatic APIs like `Chat()` and `Stream()`
3. **MCP mode** — `memg mcp` exposes memory operations as JSON-RPC tools for agent frameworks

## Architecture Invariants
Respect these boundaries:
- **Repository** (`store/store.go`) — persistence contract
- **Provider** (`llm/provider.go`) — LLM contract
- **Embedder** (`embed/embedder.go`) — embedding contract
- **Engine** (`search/engine.go`) — ranking contract
- **Stage** (`memory/augment/stage.go`) — extraction pipeline contract

Do not break interface contracts in:
- `store/store.go`
- `llm/provider.go`
- `embed/embedder.go`
- `search/engine.go`

## Core Patterns
- provider and embedder packages self-register via `init()`
- instance-level config uses `memg.With*()`
- per-call overrides use `memg.For*()` and `llm.With*()`
- SQL backends own their dialect-specific query implementations
- proxy internal calls must avoid self-interception loops

## Package Layout
- `memg.go`, `config.go`, `option.go` — core MemG types and config
- `cmd/memg/` — CLI entrypoints
- `store/` — repository contract and store implementations
- `llm/` — provider interface and provider packages
- `embed/` — embedder interface and provider packages
- `search/` — hybrid ranking logic
- `memory/` — recall, sessions, decay, reinforcement, consolidation
- `memory/augment/` — extraction pipeline
- `proxy/` — reverse proxy
- `mcp/` — MCP server
- `graph/` — knowledge graph triples
- `local/embedder/` — local embedding service

## Source of Truth Order
When facts conflict, trust sources in this order:
1. current code
2. current repo docs
3. current config/defaults in repo
4. user-provided repo context
5. prior assumptions or generic patterns

Never override repo evidence with intuition.

## Fact Verification Rules
Classify every factual claim as one of:
- **repo fact**
- **external current fact**
- **stable general knowledge**

For **repo facts**:
- verify from code or repo docs before stating them
- never guess defaults, timeouts, limits, field names, schema shape, API behavior, or file paths

For **external current facts**:
- verify from authoritative current sources before stating them
- prefer vendor docs, official announcements, or official product pages
- do not answer from memory
- do not treat search snippets as proof; open the source and confirm the claim

If the user challenges a factual claim:
- treat the earlier answer as untrusted
- re-verify from source
- retract unsupported claims plainly
- restart from verified facts only

If verification is incomplete or conflicting:
- say **unverified** or **conflicting**
- do not guess

In answers, separate:
- **Verified**
- **Inferred**
- **Unknown**

## Read Before Write
Before proposing or making a change:
1. read all directly relevant files
2. read neighboring types, helpers, interfaces, and existing tests if present
3. understand the full call path
4. follow patterns already used in the repo
5. do not patch from one isolated file without understanding surrounding flow

## Coding Rules
- do not write code unless explicitly asked; first provide a concrete plan
- prefer small, direct changes over broad rewrites
- match the style of the file being edited
- keep functions focused and explicit
- prefer Go standard library first
- preserve backward compatibility unless explicitly told otherwise
- use error returns, not panics
- wrap errors with `fmt.Errorf("context: %w", err)` where useful
- propagate `context.Context` through I/O, DB, and network paths
- do not add comments unless the logic is genuinely non-obvious
- do not create new abstractions unless required by the task
- do not modify unrelated files

## Architecture Rules
- business logic must not live in handlers/controllers
- raw SQL must stay inside storage implementations
- config must come from flags, config files, or env vars
- respect config resolution order: CLI flags > config file > env vars
- depend on interfaces, not concrete types
- avoid global mutable state except write-once registries

## Debugging Rules
When debugging:
1. reproduce or localize the issue first
2. identify the source of truth: code, config, logs, DB, docs, or infra
3. distinguish verified cause from hypothesis
4. do not propose code changes until the likely root cause is evidenced
5. if the issue may be outside code, say so before suggesting edits

Use `MEMG_DEBUG=1` when debug logging is needed.
Temporary logs must be removed before finalizing unless explicitly useful long-term.

## Failure and Data Integrity Rules
- protect writes above all else
- partial writes are worse than failed writes
- duplicate writes must be idempotent or deduplicated
- preserve ordering invariants
- do not leave the system in a half-written or unrecoverable state
- prefer graceful degradation where appropriate
- distinguish transient failures from permanent failures
- destructive operations are high risk and must be called out explicitly

## Concurrency Rules
- do not assume goroutines are safe; verify shared-state access
- every background goroutine must have a shutdown path or bounded lifecycle
- understand backpressure before using channels in request paths
- verify read-modify-write safety explicitly
- call out lock ordering and deadlock risks when relevant

## Dependency Rules
- prefer existing repo dependencies
- keep dependency surface small
- add new dependencies only when clearly necessary
- explain why stdlib or existing code is insufficient
- prefer pure Go solutions where practical

## Security Rules
- never expose secrets or tokens
- never hardcode credentials
- validate external input at proxy and MCP boundaries
- treat message content and user input as unsafe
- avoid logging API keys, message content, or PII in production paths

## Performance Rules
- do not optimize blindly
- call out repeated I/O, unnecessary allocations, excessive network calls, and N+1 patterns
- preserve readability unless performance is a demonstrated requirement
- do not introduce unbounded goroutines
- respect existing safety caps

## Documentation Rules
If behavior, architecture, or configuration changes, update the relevant docs in the same change when applicable:
- `README.md`
- `MEMORY_ARCHITECTURE.md`

Do not leave docs stale when behavior changes.

## Completion Gate
A task is complete only when all applicable checks available in the environment are satisfied or explicitly marked unavailable:
- code is syntactically valid
- imports resolve
- types match
- changed paths are internally consistent
- existing relevant tests were checked if available
- edge cases were reviewed
- all unverified parts were called out explicitly

Do not create new test files unless explicitly asked.
Use existing tests when available.
If no relevant tests exist, state that verification is limited.

## Output Rules
For repo-specific answers, always separate:
1. **Verified**
2. **Inferred**
3. **Unknown**

For implementation tasks, report:
1. **Plan** — what will be built and in which parts of the system
2. **Logic** — how the solution works end-to-end
3. **Why this approach** — why this design is correct and preferred over obvious alternatives
4. **Risks / Edge Cases** — what can break, fail, or behave unexpectedly
5. **Verification Status** — what was verified by code inspection, tests, execution, or is still unverified

Do not focus on line-by-line code changes unless explicitly asked.
Focus on system behavior, decision flow, invariants, and correctness.
Be concise. Do not dump unrelated suggestions.
Do not claim something works unless it was tested or clearly marked unverified.

## When to Ask
Ask for clarification only when blocked by:
- ambiguous business logic
- destructive schema or contract changes
- missing credentials or config
- multiple valid directions with materially different tradeoffs

Otherwise, make the best grounded call and label assumptions clearly.

## What to Never Do
- do not fabricate behavior, test results, file contents, or verification
- do not guess repository facts
- do not rewrite large parts of the system unless explicitly asked
- do not break public or interface contracts casually
- do not introduce a new storage backend without implementing the full repository contract
- do not add provider packages that skip self-registration
- do not present hypotheses as facts


While any coding, or other task you have full access to do operations over DB, chatgpt, web or any other tool you use, just justify to yourself why you need.
Before giving any number, fact, code logic, or anything upfront to me, and this is the most important above all that you need to verify that, you have 
access to all the things, DB, search tools, or whatever you need but you cant give me false things, assumptions, or incorrect information.