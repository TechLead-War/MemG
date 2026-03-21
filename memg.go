package memg

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"memg/embed"
	"memg/llm"
	"memg/memory"
	"memg/memory/augment"
	"memg/search"
	"memg/store"
)

// ChatOption modifies a single Chat call at the MemG level.
type ChatOption func(*chatConfig)

// chatConfig holds per-call overrides.
type chatConfig struct {
	entityID   string            // override entity for this call
	factFilter *store.FactFilter // optional filter for recall queries
}

// ForEntity overrides the entity for a single Chat call. This allows one
// MemG instance to serve multiple users without creating separate instances.
func ForEntity(id string) ChatOption {
	return func(c *chatConfig) { c.entityID = id }
}

// WithFactFilter applies a filter to recall queries for this chat call.
func WithFactFilter(filter store.FactFilter) ChatOption {
	return func(c *chatConfig) { c.factFilter = &filter }
}

// MemG is the primary entry point for the memory-augmented LLM system.
// It intercepts language model calls to inject relevant recalled facts,
// tracks conversations, and asynchronously extracts new knowledge.
type MemG struct {
	mu               sync.RWMutex
	cfg              *Config
	repo             store.Repository
	provider         llm.Provider
	embedder         embed.Embedder
	engine           search.Engine
	pipeline         *augment.Pipeline
	pruner           *memory.Pruner
	consciousCache   *memory.ConsciousCache
	queryTransformer memory.QueryTransformer

	// entityCache maps external entity IDs to internal UUIDs for fast
	// per-request resolution without hitting the database every time.
	entityCache sync.Map // string -> string

	entityUUID  string
	processUUID string
	sessionUUID string

	// bgSem bounds the number of concurrent background goroutines
	// (recall tracking, session summarization, post-response work).
	bgSem chan struct{}

	baseCtx    context.Context
	cancelBase context.CancelFunc
	closed     bool
}

// New creates a MemG instance connected to the given repository.
func New(repo store.Repository, opts ...Option) (*MemG, error) {
	if repo == nil {
		return nil, ErrNoRepository
	}
	cfg := DefaultConfig()
	for _, o := range opts {
		o(cfg)
	}

	ctx, cancel := context.WithCancel(context.Background())
	g := &MemG{
		cfg:        cfg,
		repo:       repo,
		engine:     search.NewHybrid(),
		pipeline:   augment.NewPipeline(repo),
		bgSem:      make(chan struct{}, 64),
		baseCtx:    ctx,
		cancelBase: cancel,
	}

	if cfg.LLMProvider != "" {
		prov, err := llm.NewProvider(cfg.LLMProvider, cfg.LLMConfig)
		if err != nil {
			return nil, fmt.Errorf("memg: llm provider %q: %w", cfg.LLMProvider, err)
		}
		g.provider = prov
	}

	if cfg.EmbedProvider != "" {
		emb, err := embed.NewEmbedder(cfg.EmbedProvider, cfg.EmbedConfig)
		if err != nil {
			return nil, fmt.Errorf("memg: embed provider %q: %w", cfg.EmbedProvider, err)
		}
		g.embedder = emb
		g.cfg.EmbedDimension = emb.Dimension()
		g.pipeline.Embedder = emb
	}

	pruner := memory.NewPruner(repo, cfg.PruneInterval)
	pruner.Start()
	g.pruner = pruner

	g.consciousCache = memory.NewConsciousCache(cfg.ConsciousCacheTTL)
	g.pipeline.OnFactInserted = func(entityUUID string, fact *store.Fact) {
		g.consciousCache.Invalidate(entityUUID)
	}
	g.pipeline.OnFactStatusChanged = func(entityUUID string, factUUID string) {
		g.consciousCache.Invalidate(entityUUID)
	}

	return g, nil
}

// Migrate creates or updates the database schema.
func (g *MemG) Migrate(ctx context.Context) error {
	return g.repo.Migrate(ctx)
}

// SetProvider configures the language model provider used by Chat and Stream.
func (g *MemG) SetProvider(p llm.Provider) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.provider = p
}

// SetEmbedder configures the embedding service used for recall queries.
func (g *MemG) SetEmbedder(e embed.Embedder) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.embedder = e
	g.cfg.EmbedDimension = e.Dimension()
	g.pipeline.Embedder = e
}

// AddStage registers a custom augmentation stage.
func (g *MemG) AddStage(s augment.Stage) {
	g.pipeline.AddStage(s)
}

// SetQueryTransformer configures an optional query transformer for recall.
func (g *MemG) SetQueryTransformer(qt memory.QueryTransformer) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.queryTransformer = qt
}

// Chat sends messages to the language model with automatic memory injection.
// Recalled facts are prepended to the system prompt, and conversation history
// is loaded from the active session. After the response arrives, the exchange
// is persisted and the augmentation pipeline is triggered.
//
// Use ForEntity("user-id") to scope the call to a specific user. This allows
// one MemG instance to serve multiple users:
//
//	g.Chat(ctx, messages, memg.ForEntity("user-alice"))
//	g.Chat(ctx, messages, memg.ForEntity("user-bob"))
//
// If ForEntity is not passed, the instance-level entity (from WithEntity) is used.
func (g *MemG) Chat(ctx context.Context, messages []*llm.Message, opts ...any) (*llm.Response, error) {
	g.mu.RLock()
	prov := g.provider
	emb := g.embedder
	g.mu.RUnlock()

	if prov == nil {
		return nil, ErrNoProvider
	}

	var cc chatConfig
	var llmOpts []llm.CallOption
	for _, o := range opts {
		switch v := o.(type) {
		case ChatOption:
			v(&cc)
		case llm.CallOption:
			llmOpts = append(llmOpts, v)
		}
	}

	entityUUID := ""
	if cc.entityID != "" {
		uuid, err := g.resolveEntityUUID(ctx, cc.entityID)
		if err != nil {
			return nil, fmt.Errorf("memg: resolve entity %q: %w", cc.entityID, err)
		}
		entityUUID = uuid
	} else {
		if err := g.resolveIdentities(ctx); err != nil {
			return nil, fmt.Errorf("memg: resolve identities: %w", err)
		}
		entityUUID = g.entityUUID
	}

	sessionUUID := ""
	if entityUUID != "" && g.cfg.SessionTimeout > 0 {
		sess, isNew, err := g.repo.EnsureSession(ctx, entityUUID, g.processUUID, g.cfg.SessionTimeout)
		if err == nil {
			sessionUUID = sess.UUID
			if isNew {
				g.goBackground(func() { g.summarizeClosedSession(entityUUID, sessionUUID) })
			}
		}
	} else if cc.entityID == "" {
		sessionUUID = g.sessionUUID
	}

	req := llm.NewRequest(messages, llmOpts...)
	retrievalMessages := memory.NormalizeConversationMessages(messages)
	if sessionUUID != "" {
		history, err := memory.LoadRecentHistory(ctx, g.repo, sessionUUID, g.cfg.WorkingMemoryTurns)
		if err == nil && len(history) > 0 {
			req.InjectHistory(memory.MissingHistory(history, messages))
			retrievalMessages = memory.MergeHistory(history, messages)
		}
	}

	if entityUUID != "" && emb != nil {
		ctxInput := memory.ContextInput{
			Budget: memory.ContextBudget{
				TotalTokens:   g.cfg.MemoryTokenBudget,
				SummaryTokens: g.cfg.SummaryTokenBudget,
			},
		}

		if g.cfg.ConsciousMode {
			conscious, err := memory.LoadConsciousContextCached(ctx, g.consciousCache, g.repo, entityUUID, g.cfg.ConsciousLimit)
			if err == nil {
				ctxInput.ConsciousFacts = conscious
			}
		}

		if q := lastUserContent(retrievalMessages); q != "" {
			if g.queryTransformer != nil {
				var history []string
				for _, m := range retrievalMessages {
					history = append(history, m.Role+": "+m.Content)
				}
				if transform, tErr := g.queryTransformer.TransformQuery(ctx, q, history); tErr == nil && transform != nil && transform.RewrittenQuery != "" {
					q = transform.RewrittenQuery
				}
			}

			vectors, embedErr := emb.Embed(ctx, []string{q})
			if embedErr == nil && len(vectors) > 0 {
				queryVec := vectors[0]
				queryModel := embed.ModelNameOf(emb)

				var recallFilters []store.FactFilter
				if cc.factFilter != nil {
					recallFilters = append(recallFilters, *cc.factFilter)
				}
				facts, err := memory.RecallWithVector(ctx, g.engine, g.repo, queryVec, queryModel, q, entityUUID, g.cfg.RecallFactsLimit, g.cfg.RecallThreshold, g.cfg.MaxRecallCandidates, recallFilters...)
				if err == nil {
					ctxInput.RecalledFacts = facts
				}

				summaries, err := memory.RecallSummariesWithVector(ctx, g.engine, g.repo, queryVec, q, entityUUID, 5, g.cfg.RecallThreshold)
				if err == nil {
					ctxInput.Summaries = summaries
				}
			}
		}

		contextText := memory.BuildContext(ctxInput)
		if contextText != "" {
			req.PrependSystem(contextText)
		}
		if len(ctxInput.RecalledFacts) > 0 {
			g.goBackground(func() { g.trackRecallUsage(ctxInput.RecalledFacts) })
		}
	}

	resp, err := prov.Chat(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("memg: provider error: %w", err)
	}

	g.goBackground(func() { g.afterResponseForEntity(entityUUID, sessionUUID, messages, resp) })
	return resp, nil
}

// Stream sends messages and returns an incremental response reader.
// Memory injection happens before the stream starts, and the completed
// exchange is persisted after the stream finishes successfully.
//
// Use ForEntity("user-id") to scope the call to a specific user, and
// WithFactFilter to constrain recall queries, just like Chat().
func (g *MemG) Stream(ctx context.Context, messages []*llm.Message, opts ...any) (*llm.StreamReader, error) {
	g.mu.RLock()
	prov := g.provider
	emb := g.embedder
	g.mu.RUnlock()

	if prov == nil {
		return nil, ErrNoProvider
	}

	var cc chatConfig
	var llmOpts []llm.CallOption
	for _, o := range opts {
		switch v := o.(type) {
		case ChatOption:
			v(&cc)
		case llm.CallOption:
			llmOpts = append(llmOpts, v)
		}
	}

	entityUUID := ""
	if cc.entityID != "" {
		uuid, err := g.resolveEntityUUID(ctx, cc.entityID)
		if err != nil {
			return nil, fmt.Errorf("memg: resolve entity %q: %w", cc.entityID, err)
		}
		entityUUID = uuid
	} else {
		if err := g.resolveIdentities(ctx); err != nil {
			return nil, fmt.Errorf("memg: resolve identities: %w", err)
		}
		entityUUID = g.entityUUID
	}

	sessionUUID := ""
	if entityUUID != "" && g.cfg.SessionTimeout > 0 {
		sess, isNew, err := g.repo.EnsureSession(ctx, entityUUID, g.processUUID, g.cfg.SessionTimeout)
		if err == nil {
			sessionUUID = sess.UUID
			if isNew {
				g.goBackground(func() { g.summarizeClosedSession(entityUUID, sessionUUID) })
			}
		}
	} else if cc.entityID == "" {
		sessionUUID = g.sessionUUID
	}

	req := llm.NewRequest(messages, llmOpts...)
	retrievalMessages := memory.NormalizeConversationMessages(messages)
	if sessionUUID != "" {
		history, err := memory.LoadRecentHistory(ctx, g.repo, sessionUUID, g.cfg.WorkingMemoryTurns)
		if err == nil && len(history) > 0 {
			req.InjectHistory(memory.MissingHistory(history, messages))
			retrievalMessages = memory.MergeHistory(history, messages)
		}
	}

	if entityUUID != "" && emb != nil {
		ctxInput := memory.ContextInput{
			Budget: memory.ContextBudget{
				TotalTokens:   g.cfg.MemoryTokenBudget,
				SummaryTokens: g.cfg.SummaryTokenBudget,
			},
		}

		if g.cfg.ConsciousMode {
			conscious, err := memory.LoadConsciousContextCached(ctx, g.consciousCache, g.repo, entityUUID, g.cfg.ConsciousLimit)
			if err == nil {
				ctxInput.ConsciousFacts = conscious
			}
		}

		if q := lastUserContent(retrievalMessages); q != "" {
			if g.queryTransformer != nil {
				var history []string
				for _, m := range retrievalMessages {
					history = append(history, m.Role+": "+m.Content)
				}
				if transform, tErr := g.queryTransformer.TransformQuery(ctx, q, history); tErr == nil && transform != nil && transform.RewrittenQuery != "" {
					q = transform.RewrittenQuery
				}
			}

			vectors, embedErr := emb.Embed(ctx, []string{q})
			if embedErr == nil && len(vectors) > 0 {
				queryVec := vectors[0]
				queryModel := embed.ModelNameOf(emb)

				var recallFilters []store.FactFilter
				if cc.factFilter != nil {
					recallFilters = append(recallFilters, *cc.factFilter)
				}
				facts, err := memory.RecallWithVector(ctx, g.engine, g.repo, queryVec, queryModel, q, entityUUID, g.cfg.RecallFactsLimit, g.cfg.RecallThreshold, g.cfg.MaxRecallCandidates, recallFilters...)
				if err == nil {
					ctxInput.RecalledFacts = facts
				}

				summaries, err := memory.RecallSummariesWithVector(ctx, g.engine, g.repo, queryVec, q, entityUUID, 5, g.cfg.RecallThreshold)
				if err == nil {
					ctxInput.Summaries = summaries
				}
			}
		}

		contextText := memory.BuildContext(ctxInput)
		if contextText != "" {
			req.PrependSystem(contextText)
		}
		if len(ctxInput.RecalledFacts) > 0 {
			g.goBackground(func() { g.trackRecallUsage(ctxInput.RecalledFacts) })
		}
	}

	upstream, err := prov.Stream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("memg: provider error: %w", err)
	}

	ch := make(chan llm.StreamEvent)
	go func() {
		defer close(ch)
		for upstream.Next() {
			ch <- llm.StreamEvent{
				Delta:  upstream.Delta(),
				Done:   upstream.Done(),
				Finish: upstream.FinishReason(),
			}
		}
		if err := upstream.Err(); err != nil {
			ch <- llm.StreamEvent{Err: err}
			return
		}
		content := strings.TrimSpace(upstream.Text())
		if content == "" {
			return
		}
		g.afterResponseForEntity(entityUUID, sessionUUID, messages, &llm.Response{
			Role:         llm.RoleAssistant,
			Content:      content,
			FinishReason: upstream.FinishReason(),
		})
	}()

	return llm.NewStreamReader(ch), nil
}

// ReEmbedFacts re-embeds all facts for the given entity using the current
// embedding model. Use this after changing the embedding provider.
func (g *MemG) ReEmbedFacts(ctx context.Context, entityID string) (int, error) {
	g.mu.RLock()
	emb := g.embedder
	g.mu.RUnlock()
	if emb == nil {
		return 0, ErrNoEmbedder
	}

	entityUUID, err := g.resolveEntityUUID(ctx, entityID)
	if err != nil {
		return 0, fmt.Errorf("memg: resolve entity: %w", err)
	}

	modelName := g.cfg.EmbedProvider
	if g.cfg.EmbedConfig.Model != "" {
		modelName = g.cfg.EmbedConfig.Model
	}

	return memory.ReEmbedFacts(ctx, g.repo, emb, entityUUID, modelName, 50)
}

// Close shuts down background workers and releases the repository connection.
func (g *MemG) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return ErrAlreadyClosed
	}
	g.closed = true
	g.cancelBase()
	g.pruner.Stop()
	g.pipeline.Shutdown()
	return g.repo.Close()
}

// resolveEntityUUID maps an external entity ID to an internal UUID, using
// a cache to avoid hitting the database on every request.
func (g *MemG) resolveEntityUUID(ctx context.Context, externalID string) (string, error) {
	if externalID == "" {
		return "", nil
	}
	if cached, ok := g.entityCache.Load(externalID); ok {
		return cached.(string), nil
	}
	uuid, err := g.repo.UpsertEntity(ctx, externalID)
	if err != nil {
		return "", err
	}
	g.entityCache.Store(externalID, uuid)
	return uuid, nil
}

// resolveIdentities maps external IDs to internal UUIDs on first access.
func (g *MemG) resolveIdentities(ctx context.Context) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.cfg.EntityID != "" && g.entityUUID == "" {
		id, err := g.repo.UpsertEntity(ctx, g.cfg.EntityID)
		if err != nil {
			return fmt.Errorf("resolve identities: upsert entity: %w", err)
		}
		g.entityUUID = id
	}
	if g.cfg.ProcessID != "" && g.processUUID == "" {
		id, err := g.repo.UpsertProcess(ctx, g.cfg.ProcessID)
		if err != nil {
			return fmt.Errorf("resolve identities: upsert process: %w", err)
		}
		g.processUUID = id
	}
	if g.cfg.SessionID != "" {
		oldSessionUUID := g.sessionUUID
		sess, isNew, err := g.repo.EnsureSession(ctx, g.entityUUID, g.processUUID, g.cfg.SessionTimeout)
		if err != nil {
			return fmt.Errorf("resolve identities: ensure session: %w", err)
		}
		if isNew && oldSessionUUID != "" && sess.UUID != oldSessionUUID {
			g.goBackground(func() { g.summarizeClosedSession(g.entityUUID, sess.UUID) })
		}
		g.sessionUUID = sess.UUID
	}
	return nil
}

// summarizeClosedSession finds the most recent unsummarized conversation for an
// entity that is not part of the current active session and summarizes it.
func (g *MemG) summarizeClosedSession(entityUUID, currentSessionUUID string) {
	ctx := g.baseCtx

	g.mu.RLock()
	prov := g.provider
	emb := g.embedder
	g.mu.RUnlock()

	if prov == nil || emb == nil {
		return
	}

	conv, err := g.repo.FindUnsummarizedConversation(ctx, entityUUID, currentSessionUUID)
	if err != nil || conv == nil {
		return
	}

	if conv.Summary != "" {
		return
	}

	if err := memory.GenerateAndStoreSummary(ctx, prov, emb, g.repo, conv.UUID); err != nil {
		if g.cfg.Debug {
			fmt.Printf("memg: generate summary: %v\n", err)
		}
	}
}

func (g *MemG) afterResponseForEntity(entityUUID, sessionUUID string, input []*llm.Message, resp *llm.Response) {
	ctx := g.baseCtx
	if sessionUUID != "" {
		if err := memory.SaveExchange(ctx, g.repo, sessionUUID, entityUUID, input, resp); err != nil {
			if g.cfg.Debug {
				fmt.Printf("memg: save exchange: %v\n", err)
			}
		}
	}
	if entityUUID != "" || g.processUUID != "" {
		all := memory.NormalizeConversationMessages(input)
		if content := strings.TrimSpace(resp.Content); content != "" {
			all = append(all, llm.AssistantMessage(content))
		}
		if len(all) == 0 {
			return
		}
		g.pipeline.Enqueue(&augment.Job{
			EntityID:  entityUUID,
			ProcessID: g.processUUID,
			Messages:  all,
		})
	}
}

// trackRecallUsage updates recall stats for facts that were injected into the prompt.
func (g *MemG) trackRecallUsage(facts []*memory.RecalledFact) {
	if len(facts) == 0 {
		return
	}
	ids := make([]string, len(facts))
	for i, f := range facts {
		ids[i] = f.ID
	}
	ctx := g.baseCtx
	if err := g.repo.UpdateRecallUsage(ctx, ids); err != nil {
		if g.cfg.Debug {
			fmt.Printf("memg: update recall usage: %v\n", err)
		}
	}
}

// goBackground runs fn in a goroutine, bounded by the background semaphore.
// If the semaphore is full, fn runs synchronously to prevent unbounded
// goroutine growth.
func (g *MemG) goBackground(fn func()) {
	select {
	case g.bgSem <- struct{}{}:
		go func() {
			defer func() { <-g.bgSem }()
			fn()
		}()
	default:
		fn()
	}
}

func lastUserContent(msgs []*llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == llm.RoleUser {
			return msgs[i].Content
		}
	}
	return ""
}

func renderFacts(facts []*memory.RecalledFact) string {
	var b strings.Builder
	b.WriteString("Relevant context from memory:\n")
	for _, f := range facts {
		b.WriteString("- ")
		if f.TemporalStatus == "historical" {
			b.WriteString("[historical] ")
		}
		b.WriteString(f.Content)
		b.WriteByte('\n')
	}
	return b.String()
}

func renderConscious(facts []*memory.ConsciousFact) string {
	var b strings.Builder
	b.WriteString("User profile:\n")
	for _, f := range facts {
		b.WriteString("- ")
		b.WriteString(f.Content)
		b.WriteByte('\n')
	}
	return b.String()
}

func renderSummaries(summaries []*memory.RecalledSummary) string {
	var b strings.Builder
	b.WriteString("Relevant past conversations:\n")
	for _, s := range summaries {
		b.WriteString("- [")
		b.WriteString(s.CreatedAt.Format("Jan 2, 2006"))
		b.WriteString("] ")
		b.WriteString(s.Summary)
		b.WriteByte('\n')
	}
	return b.String()
}
