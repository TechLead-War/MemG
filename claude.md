# CLAUDE.md

## Purpose
This repository is MemG, a pluggable memory layer for LLM applications, written in Go.
Primary goal: enable zero-code persistent memory for any LLM app via reverse proxy or MCP server, with hybrid recall, fact lifecycle management, and multi-provider support.
But our good to have is also if we can just plug and play in any lang repo. Do not optimize for side quests or speculative improvements.

Critical workflows:
1. **Proxy mode** — `memg proxy` intercepts OpenAI/Anthropic API calls, injects recalled facts, and asynchronously extracts knowledge from conversations.
2. **Library mode** — `memg.New(repo, opts...)` gives Go developers direct programmatic access to `Chat()`, `Stream()`, and memory management.
3. **MCP mode** — `memg mcp` exposes memory operations as JSON-RPC 2.0 tools for agent frameworks like Claude.

Non-goals:
- Do not turn this into a general-purpose database or vector store.
- Do not rewrite the provider registry pattern — it works and all 20 providers follow it.
- Do not introduce frameworks, ORMs, or heavy abstractions over the existing clean interfaces.

## Architecture

### Core Interfaces
Every major subsystem is behind an interface. Respect these boundaries:
- **Repository** (`store/store.go`) — composite of 16+ sub-interfaces for all persistence ops.
- **Provider** (`llm/provider.go`) — `Chat()` and `Stream()` for LLM calls.
- **Embedder** (`embed/embedder.go`) — `Embed()` and `Dimension()` for vector generation.
- **Engine** (`search/engine.go`) — `Rank()` for hybrid search scoring.
- **Stage** (`memory/augment/stage.go`) — `Execute()` for fact extraction from messages.

### Key Patterns
- **Provider Registry via `init()`** — every provider package self-registers. New providers must follow this pattern exactly: call `llm.RegisterProvider()` or `embed.RegisterEmbedder()` in `init()`.
- **Option pattern** — instance-level config via `memg.With*()` funcs, per-call overrides via `memg.For*()` and `llm.With*()`.
- **SQL dialect abstraction** — each storage backend defines its own `Queries` struct. Cross-DB time parsing handled by `flexTime` in `store/sqlstore/encoding.go`.
- **Internal transport** — proxy marks its own API calls (for extraction/embedding) via `proxy.NewInternalHTTPClient()` to avoid self-interception loops.

### Package Layout
```
memg.go, config.go, option.go    → top-level MemG struct and config
cmd/memg/                        → CLI (proxy, mcp subcommands)
store/                           → Repository interface + sqlstore implementations
llm/                             → LLM provider interface + 10 provider packages
embed/                           → Embedder interface + 10 provider packages
search/                          → Hybrid engine (cosine + BM25 + Kneedle cutoff)
memory/                          → Recall, sessions, decay, consolidation, conscious loading
memory/augment/                  → Extraction pipeline (stages, pool, conflict resolution)
proxy/                           → Reverse proxy + request interception
mcp/                             → MCP JSON-RPC server
graph/                           → Knowledge graph triples
local/embedder/                  → Python gRPC service for local sentence-transformers
```

## Coding Rules
- Do not write code unless explicitly told to. First propose a detailed plan — what files are involved, what changes go where, why this approach, what existing code gets reused, and any edge cases or risks. The plan should be complete enough that I do not need to ask follow-up questions to understand the full picture.
- Understand direct or indirect things, without me correcting you, think in multiple aspect before answering to me.
- There should not be any fact that you tell me and that is incorrect, you need to give me latest, and factually correct information.
- Prefer small, direct changes over broad rewrites. Check what the system already has before building new.
- Match the existing style of the file you are editing.
- Follow standard design principles — SOLID, KISS, DRY, YAGNI, and any others that apply. Do not limit thinking to a checklist.
- Keep functions focused and short. Prefer explicit names over clever names.
- No comments unless the logic is non-obvious.
- Preserve backward compatibility unless explicitly told otherwise.
- Use Go standard library first. Only reach for third-party packages when the stdlib genuinely cannot do it.
- Error returns over panics. Always. Wrap errors with `fmt.Errorf("context: %w", err)` to preserve sentinel errors from `errors.go`.
- Always propagate `context.Context` — every function that does I/O, DB, or network calls must accept and pass `ctx`.
- Do not write test files. Just verify things work.

## Architecture Rules
- Business logic must not live inside handlers/controllers. The proxy handler delegates to `memory/` and `memg.go`.
- Database access must stay in `store/sqlstore/`. Nothing outside that package should write raw SQL.
- Config must come from environment/config files/CLI flags, never hardcoded. Follow the three-level resolution hierarchy: CLI flags > config file > env vars.
- Avoid tight coupling between modules. Each package should depend on interfaces, not concrete types.
- Prefer composition over inheritance.
- Avoid global mutable state (registries are the one exception — they are write-once-at-init).

## Output Rules
When making changes:
1. Show me the things that we have done, not code wise, but logically what we have done.
2. First understand the relevant files.
3. Make the smallest correct change.
4. Explain what changed and why.
5. List risks or edge cases if any.
6. Do not dump unrelated suggestions.
7. Before completion, again go through the code and see if there is scope of some issues, knowledge gap.

When answering:
- Be precise.
- Be concise.
- Do not pretend certainty when unsure.
- Say what you verified vs what you inferred.
- Do not claim something works unless it was tested or clearly marked unverified.

## Editing Rules
- Before editing, inspect nearby code and follow local conventions.
- Do not reformat unrelated files.
- Do not modify `go.sum` unless dependency changes are required.
- Do not touch migrations or schema changes unless the task requires it.
- When adding a new provider, scaffold it by copying the closest existing provider package — do not invent a new structure.

## Debugging Rules
When debugging:
- Reproduce the issue first. Ask yourself why this might be coming if unsure ask some data that you think might help while debugging.
- Identify whether it is input, state, network, DB, concurrency, or config related.
- There can be the case where code itself is not the issue, infra, or something else is the issue, ensure we are not doing code changes unless we are the issue.
- Prefer root-cause fixes over symptom patches.
- Use `MEMG_DEBUG=1` to enable debug logging when needed.
- Add temporary logs only if needed, and remove them before finalizing.

## Dependency Rules
- Prefer existing libraries already used in the repo.
- The repo intentionally has a small dependency surface. Do not bloat it.
- If adding a dependency, explain why existing code or stdlib cannot solve it cleanly.
- Pure Go implementations preferred (e.g., `modernc.org/sqlite` over CGo sqlite3).

## Security Rules
- Never expose secrets or tokens.
- Never hardcode credentials. Use the existing `ProviderConfig.APIKey` → env var fallback chain.
- Validate external input at proxy/MCP boundaries.
- Treat user-generated content in messages as unsafe.
- Avoid logging API keys, message content in production, or any PII.

## Performance Rules
- Do not optimize blindly.
- Call out N+1 queries, repeated I/O, excessive allocations, and unnecessary network calls.
- Preserve readability unless performance is a real requirement.
- The augmentation pipeline already uses a fixed 32-worker pool — do not add unbounded goroutines.
- `MaxRecallCandidates` (default 10,000) is a safety cap. Respect it.

## Git Rules
- Keep diffs minimal and focused. Group related changes, avoid noisy refactors.
- Do not add co-author tags to commits unless explicitly asked.
- Commit message format: version tag only (`v1.xx`), nothing else in the subject line.

## Decision Heuristics
Use this order:
1. Correctness
2. Simplicity
3. Consistency with repo
4. Maintainability
5. Performance

## When to Ask
Ask for clarification if you don't understand the task, you feel there is some knowledge gap, anything like some example but not limited to these:
- Ambiguous business logic around memory lifecycle (decay, reinforcement, dedup).
- Destructive changes to the store schema or provider interfaces.
- Missing credentials/config.
- Multiple valid directions with major tradeoffs.

## What to Never Do
- Do not fabricate behavior, test results, or file contents.
- Do not claim production readiness without evidence.
- Do not rewrite large portions of the codebase unless explicitly requested.
- Do not break the interface contracts in `store/store.go`, `llm/provider.go`, `embed/embedder.go`, or `search/engine.go`.
- Do not introduce a new storage backend without implementing the full `Repository` interface.
- Do not add provider packages that skip the `init()` self-registration pattern.

## Key Reference Documents
- `MEMORY_ARCHITECTURE.md` — deep design doc covering semantic retrieval, session management, fact lifecycle, decay, reinforcement, consolidation, and all advanced subsystems.
- `README.md` — quick start guides, provider tables, configuration reference.

When a change affects behavior, architecture, or configuration documented in these files, update them as part of the same change — do not wait to be asked.

## Truthfulness Rules
- Never guess APIs, function names, file paths, schemas, or library behavior.
- If something is not present in the repo or clearly provided, mark it as unknown, ask me for resolutions.
- Do not invent code to "seem complete".
- Separate:
    - Verified from codebase
    - Inferred from context
- No Assumptions
- If a task depends on an unknown, say so explicitly before coding.

## Read Before Write
Before changing code:
1. Read all directly related files.
2. Read neighboring types/interfaces/helpers/tests.
3. Understand call flow end-to-end.
4. Do not patch blindly from one file only.
5. Do not write code until the existing pattern is understood.

## Completion Gate
A task is not complete until:
- Code is syntactically valid.
- Imports resolve.
- Types match.
- Lint passes for changed files.
- Relevant tests pass, or absence of tests is explicitly stated.
- Edge cases were checked.
- Output matches requested behavior.
  Never claim "done" before these checks.

## Self-Review Checklist
Before finalizing, check:
- Will this compile/run?
- Are names/types exact?
- Any missing imports?
- Any dead code?
- Any race condition?
- Any nil/null issue?
- Any off-by-one?
- Any incorrect error handling?
- Any broken backward compatibility?
- Any mismatch with business logic?
- Did I solve root cause, not symptom?

## Logic Verification Rules
- Do not stop at "code looks right".
- Trace the business flow with 1 normal case, 1 edge case, 1 failure case.
- Verify that state transitions are correct.
- Verify that data written/read is consistent across layers.
- Check for duplicate actions, partial writes, and rollback gaps.

## Work Style
- Prefer slower, verified work over fast speculative work.
- Do not optimize for speed of response.
- Do not produce code early just to be helpful.
- First get the model of the system right, then write code.

## No-Assumption Coding
- Do not assume input shape.
- Do not assume DB schema.
- Do not assume concurrency safety.
- Do not assume nil/null cannot happen.
- Do not assume external services always succeed.
- Do not assume "similar code elsewhere" is correct.

## System-Level Thinking
- Every change has upstream and downstream effects. Before touching a function, understand who calls it, what it calls, and what state it mutates.
- Trace the full request path — from entry point (proxy/MCP/library) through business logic to storage and back. A fix in one layer can break invariants in another.
- Think about the system at 10x scale. Will this approach still work with 10x more facts, 10x more entities, 10x more concurrent requests?
- Understand the deployment topology. Code that works in a single-process test may fail when the proxy, embedder, and LLM are separate services with network between them.

## Failure Mode Discipline
- For every code path, answer: what happens when this fails? Is the failure loud (error returned, logged) or silent (swallowed, ignored)?
- Classify failures by blast radius: does a failure affect one request, one entity, or the entire system?
- Prefer graceful degradation over hard failure. If recall fails, the LLM should still respond — just without memory context. If extraction fails, the conversation should still complete.
- Distinguish transient failures (network timeout, rate limit) from permanent failures (bad schema, missing table). Only retry transient failures.
- Never leave the system in a half-written state. If a multi-step operation fails partway, either complete it, roll it back, or make the partial state recoverable.

## API Contract Discipline
- Interfaces in `store/store.go`, `llm/provider.go`, `embed/embedder.go`, and `search/engine.go` are contracts with 20+ implementations. Treat them as permanent.
- Adding to an interface is a breaking change — every implementation must be updated. Prefer new interfaces over extending existing ones when the addition is optional.
- Public function signatures are promises. Changing parameter order, return types, or semantics breaks every caller. When in doubt, add a new function.
- Config struct fields are part of the public API. Adding is safe. Removing or renaming is breaking. Changing default behavior is breaking.
- Error types and sentinel errors are contracts. Code downstream switches on them. Do not change error identity without understanding all callers.

## Data Integrity First
- Data outlives code. A bug in code gets fixed in minutes; corrupt data can take weeks to recover. Protect writes above all else.
- Partial writes are worse than failed writes. If `SaveExchange` writes user messages but fails on the assistant message, the conversation log is inconsistent. Design for atomicity.
- Duplicate writes must be idempotent. If a retry causes a fact to be extracted twice, dedup must catch it. If it doesn't, you have a data bug.
- Ordering matters. Messages in a conversation have sequence. Facts have temporal relationships. Embeddings must match their source text. Verify ordering invariants.
- Deletion is permanent. Decay, pruning, and consolidation remove data. Ensure the criteria are correct and the operation is logged. A bad pruning run is unrecoverable.

## Observability
- If you can't debug it in production with `MEMG_DEBUG=1`, it's not production-ready. Every significant operation (recall, extraction, session rollover, pruning) should have a debug trace.
- Silent failures are the worst kind. A swallowed error in a background goroutine means the system looks healthy while losing data. At minimum, log at debug level.
- Distinguish "nothing happened" from "something failed silently". If recall returns zero facts, is it because there are no facts, or because the embedder errored?
- Include enough context in error messages to identify the operation, the entity, and the input that triggered the failure — without logging PII or API keys.

## Concurrency Reasoning
- Do not hope goroutines are safe — prove it. For every piece of shared state, identify which lock protects it and verify the lock is held at every access.
- Document locking order. If `MemG.mu` is ever held while calling into `repo`, and `repo` has its own lock, that's a potential deadlock. Map the lock graph.
- Background goroutines must have a shutdown path. Every `go func()` must either complete quickly or be cancellable via context. Orphaned goroutines are resource leaks.
- The `sync.Map` in `entityCache` is safe for concurrent access but not for read-modify-write sequences. If you need compare-and-swap, use a mutex.
- Channel sends can block forever. If a pipeline `Enqueue` blocks because the channel is full, the HTTP handler hangs. Understand backpressure in every async path.

## Rollback Awareness
- Every change should be reversible. If it's not (schema migration, data format change, deleted data), call that out explicitly before proceeding.
- Config changes are instantly reversible — just change the value. Code changes require a redeploy. Schema changes may require a migration. Know which category your change falls in.
- Feature additions are safer than behavior modifications. Adding a new config field with a zero-value default is backward compatible. Changing what an existing field means is not.
- When modifying the extraction pipeline, consider: what happens to facts extracted by the old logic? Will the new logic conflict with them? Will consolidation handle the transition?

## Incremental Delivery
- Ship the smallest verifiable unit. A PR that does one thing well is better than a PR that does three things with unclear interactions.
- Each step should leave the system in a valid state. Do not land half of a refactor that requires the second half to compile.
- When a change spans multiple packages, order the work so each step is independently correct: interfaces first, then implementations, then callers.
- Prefer additive changes (new function, new field, new package) over modificative changes (rename, restructure, change behavior). Additive changes are lower risk and easier to review.