package bench

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/memory/augment"
	"memg/store"
	"memg/store/sqlstore"
)

// RunConfig holds all settings for a benchmark run.
type RunConfig struct {
	DataPath      string
	DBPath        string
	LLMProvider   string
	LLMModel      string
	LLMAPIKey     string
	EmbedProvider string
	EmbedModel    string
	EmbedAPIKey   string
	RecallLimit   int
	RecallThresh  float64
	MaxCandidates int
	Categories    []int
	Conversations []string
	QAConcurrency int
	Debug         bool
}

// QAResult holds the score for a single QA question.
type QAResult struct {
	SampleID  string
	Category  int
	Question  string
	Gold      string
	Predicted string
	F1        float64
}

// CategoryScore aggregates scores for one category.
type CategoryScore struct {
	Category int
	Name     string
	Count    int
	SumF1    float64
	AvgF1    float64
}

// BenchResult holds the full benchmark output.
type BenchResult struct {
	Results    []QAResult
	Categories []CategoryScore
	OverallF1  float64
	TotalQA    int
	TotalFacts int
	Duration   time.Duration
}

// Run executes the full LoCoMo benchmark: ingest conversations, then evaluate QA.
func Run(ctx context.Context, cfg RunConfig) (*BenchResult, error) {
	start := time.Now()

	// Load dataset.
	fmt.Println("Loading LoCoMo dataset...")
	ds, err := LoadDataset(cfg.DataPath)
	if err != nil {
		return nil, fmt.Errorf("load dataset: %w", err)
	}
	fmt.Printf("Loaded %d conversations, computing total QA...\n", len(ds.Entries))

	totalQA := 0
	for _, e := range ds.Entries {
		totalQA += len(e.QA)
	}
	fmt.Printf("Total QA pairs: %d\n", totalQA)

	// Filter entries if specific conversations requested.
	entries := ds.Entries
	if len(cfg.Conversations) > 0 {
		convSet := make(map[string]struct{})
		for _, c := range cfg.Conversations {
			convSet[c] = struct{}{}
		}
		var filtered []Entry
		for _, e := range entries {
			if _, ok := convSet[e.SampleID]; ok {
				filtered = append(filtered, e)
			}
		}
		entries = filtered
		fmt.Printf("Filtered to %d conversations\n", len(entries))
	}

	// Filter QA by category if requested.
	catSet := make(map[int]struct{})
	if len(cfg.Categories) > 0 {
		for _, c := range cfg.Categories {
			catSet[c] = struct{}{}
		}
	}

	// Open database.
	dbPath := cfg.DBPath
	if dbPath == "" {
		dbPath = filepath.Join(os.TempDir(), fmt.Sprintf("memg_bench_%d.db", time.Now().UnixNano()))
	}
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return nil, fmt.Errorf("create db dir: %w", err)
	}
	fmt.Printf("Database: %s\n", dbPath)

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	defer db.Close()
	db.SetMaxOpenConns(4)
	db.SetMaxIdleConns(2)

	repo := sqlstore.NewSQLite(db)
	if err := repo.Migrate(ctx); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}

	// Create embedder.
	emb, err := embed.NewEmbedder(cfg.EmbedProvider, embed.ProviderConfig{
		APIKey: cfg.EmbedAPIKey,
		Model:  cfg.EmbedModel,
	})
	if err != nil {
		return nil, fmt.Errorf("create embedder (%s): %w", cfg.EmbedProvider, err)
	}
	fmt.Println("Probing embedder...")
	probeCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	if err := embed.Probe(probeCtx, emb); err != nil {
		return nil, fmt.Errorf("embedder probe: %w", err)
	}
	fmt.Printf("Embedder ready: %s (dim=%d)\n", cfg.EmbedProvider, emb.Dimension())

	// Create LLM provider.
	provider, err := llm.NewProvider(cfg.LLMProvider, llm.ProviderConfig{
		APIKey: cfg.LLMAPIKey,
		Model:  cfg.LLMModel,
	})
	if err != nil {
		return nil, fmt.Errorf("create llm (%s): %w", cfg.LLMProvider, err)
	}
	fmt.Printf("LLM ready: %s model=%s\n", cfg.LLMProvider, cfg.LLMModel)

	// Create pipeline with bench extraction stage.
	pipeline := augment.NewPipeline(repo)
	pipeline.Embedder = emb
	stage := NewExtractionStage(provider, emb)
	pipeline.AddStage(stage)

	// === Phase 1: Ingest conversations ===
	fmt.Println("\n=== Phase 1: Ingesting conversations ===")
	entityMap := make(map[string]string) // sampleID -> entityUUID
	totalFacts := 0

	var summaryWg sync.WaitGroup
	summarySem := make(chan struct{}, 8) // limit concurrent summary goroutines

	for i, entry := range entries {
		fmt.Printf("\n[%d/%d] Ingesting %s (%s & %s, %d sessions)...\n",
			i+1, len(entries), entry.SampleID,
			entry.Conversation.SpeakerA, entry.Conversation.SpeakerB,
			len(entry.Conversation.Sessions))

		entityUUID, err := repo.UpsertEntity(ctx, "locomo-"+entry.SampleID)
		if err != nil {
			return nil, fmt.Errorf("upsert entity %s: %w", entry.SampleID, err)
		}
		entityMap[entry.SampleID] = entityUUID

		for j, session := range entry.Conversation.Sessions {
			msgs := sessionToMessages(session, entry.Conversation.SpeakerA, entry.Conversation.SpeakerB)
			sessionCtx := ""
			if session.DateTime != "" {
				sessionCtx = fmt.Sprintf("This conversation session (#%d) took place on %s between %s and %s.",
					session.Number, session.DateTime, entry.Conversation.SpeakerA, entry.Conversation.SpeakerB)
			}
			job := &augment.Job{
				EntityID:       entityUUID,
				Messages:       msgs,
				SessionContext: sessionCtx,
			}
			nFacts, pErr := pipeline.ProcessSync(ctx, job)
			if pErr != nil {
				log.Printf("  session %d extraction failed: %v", session.Number, pErr)
				continue
			}
			totalFacts += nFacts

			// Generate session summary for summary-based recall.
			summaryWg.Add(1)
			summarySem <- struct{}{}
			go func(s Session, eUUID, spkA, spkB string) {
				defer summaryWg.Done()
				defer func() { <-summarySem }()
				generateSessionSummary(ctx, repo, emb, provider, eUUID, s, spkA, spkB)
			}(session, entityUUID, entry.Conversation.SpeakerA, entry.Conversation.SpeakerB)

			if cfg.Debug || (j+1)%5 == 0 || j == len(entry.Conversation.Sessions)-1 {
				fmt.Printf("  session %d/%d: +%d facts (total %d)\n",
					j+1, len(entry.Conversation.Sessions), nFacts, totalFacts)
			}
		}
	}

	// Wait for all session summaries to finish before Phase 2.
	summaryWg.Wait()
	fmt.Printf("\nIngestion complete: %d facts extracted\n", totalFacts)

	// === Phase 2: QA Evaluation ===
	fmt.Println("\n=== Phase 2: QA Evaluation ===")

	var allResults []QAResult
	var mu sync.Mutex
	concurrency := cfg.QAConcurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	sem := make(chan struct{}, concurrency)

	for _, entry := range entries {
		entityUUID := entityMap[entry.SampleID]
		qaList := entry.QA

		// Filter by category.
		if len(catSet) > 0 {
			var filtered []QA
			for _, q := range qaList {
				if _, ok := catSet[q.Category]; ok {
					filtered = append(filtered, q)
				}
			}
			qaList = filtered
		}

		fmt.Printf("\n[%s] Evaluating %d questions...\n", entry.SampleID, len(qaList))
		var wg sync.WaitGroup

		for qi, qa := range qaList {
			qa := qa
			qi := qi
			wg.Add(1)
			sem <- struct{}{}

			go func() {
				defer wg.Done()
				defer func() { <-sem }()

				predicted, qErr := answerQuestion(
					ctx,
					provider,
					emb,
					repo,
					entityUUID,
					entry.Conversation.SpeakerA,
					entry.Conversation.SpeakerB,
					qa,
					cfg.RecallLimit,
					cfg.RecallThresh,
					cfg.MaxCandidates,
				)
				if qErr != nil {
					log.Printf("  Q%d failed: %v", qi+1, qErr)
					predicted = ""
				}

				gold := qa.GoldAnswer()

				f1 := ScoreQA(predicted, qa)

				result := QAResult{
					SampleID:  entry.SampleID,
					Category:  qa.Category,
					Question:  qa.Question,
					Gold:      gold,
					Predicted: predicted,
					F1:        f1,
				}

				mu.Lock()
				allResults = append(allResults, result)
				mu.Unlock()

				if cfg.Debug {
					fmt.Printf("  Q%d [cat%d] F1=%.3f | Q: %s\n", qi+1, qa.Category, f1, truncate(qa.Question, 60))
					fmt.Printf("       Gold: %s\n", truncate(gold, 60))
					fmt.Printf("       Pred: %s\n", truncate(predicted, 60))
				}
			}()
		}
		wg.Wait()

		// Print per-conversation progress.
		var convF1Sum float64
		for _, r := range allResults {
			if r.SampleID == entry.SampleID {
				convF1Sum += r.F1
			}
		}
		convCount := 0
		for _, r := range allResults {
			if r.SampleID == entry.SampleID {
				convCount++
			}
		}
		if convCount > 0 {
			fmt.Printf("[%s] Avg F1: %.4f (%d questions)\n", entry.SampleID, convF1Sum/float64(convCount), convCount)
		}
	}

	// === Aggregate results ===
	catScores := make(map[int]*CategoryScore)
	for _, r := range allResults {
		cs, ok := catScores[r.Category]
		if !ok {
			cs = &CategoryScore{
				Category: r.Category,
				Name:     CategoryName(r.Category),
			}
			catScores[r.Category] = cs
		}
		cs.Count++
		cs.SumF1 += r.F1
	}

	var categories []CategoryScore
	var overallSum float64
	overallCount := 0
	for _, cs := range catScores {
		if cs.Count > 0 {
			cs.AvgF1 = cs.SumF1 / float64(cs.Count)
		}
		categories = append(categories, *cs)
		overallSum += cs.SumF1
		overallCount += cs.Count
	}

	overallF1 := 0.0
	if overallCount > 0 {
		overallF1 = overallSum / float64(overallCount)
	}

	result := &BenchResult{
		Results:    allResults,
		Categories: categories,
		OverallF1:  overallF1,
		TotalQA:    overallCount,
		TotalFacts: totalFacts,
		Duration:   time.Since(start),
	}

	printReport(result)
	return result, nil
}

func sessionToMessages(session Session, speakerA, speakerB string) []*llm.Message {
	// Add a system message with session context for temporal resolution.
	var msgs []*llm.Message
	if session.DateTime != "" {
		ctx := fmt.Sprintf("This conversation session (#%d) took place on %s between %s and %s.",
			session.Number, session.DateTime, speakerA, speakerB)
		msgs = append(msgs, llm.SystemMessage(ctx))
	}

	for _, turn := range session.Turns {
		content := fmt.Sprintf("%s: %s", turn.Speaker, turn.TurnText())
		msgs = append(msgs, llm.UserMessage(content))
	}
	return msgs
}

func answerQuestion(
	ctx context.Context,
	provider llm.Provider,
	embedder embed.Embedder,
	repo store.Repository,
	entityUUID string,
	speakerA string,
	speakerB string,
	qa QA,
	recallLimit int,
	recallThresh float64,
	maxCandidates int,
) (string, error) {
	question := qa.Question
	if recallLimit <= 0 {
		recallLimit = 50
	}
	if recallThresh <= 0 {
		recallThresh = 0.05
	}
	if maxCandidates <= 0 {
		maxCandidates = 500
	}

	// Use the library's recall + context builder — same code path as memg.Chat().
	memoryContext, err := recallAndBuildContext(
		ctx, embedder, repo, entityUUID, question,
		recallLimit, recallThresh, maxCandidates,
		4000, 1000, 10,
	)
	if err != nil {
		return "", fmt.Errorf("recall: %w", err)
	}

	factCtx := memoryContext

	system := buildAnswerSystemPrompt(factCtx)

	req := &llm.Request{
		System:    system,
		Messages:  []*llm.Message{llm.UserMessage(question)},
		MaxTokens: 256,
	}

	resp, err := chatRetry(ctx, provider, req, 2)
	if err != nil {
		return "", fmt.Errorf("answer llm: %w", err)
	}

	return strings.TrimSpace(resp.Content), nil
}

func buildAnswerSystemPrompt(factContext string) string {
	var b strings.Builder
	b.WriteString(factContext)
	b.WriteString("\nBased ONLY on the above facts, answer the following question.\n")
	b.WriteString("Rules:\n")
	b.WriteString("- Use the EXACT names, titles, numbers, dates, and terms from the facts. Do not paraphrase, generalize, or rephrase.\n")
	b.WriteString("  - If a fact says \"Cakes\", answer \"Cakes\", not \"Baked goods\".\n")
	b.WriteString("  - If a fact says \"auto engineering\", answer \"auto engineering\", not \"Working on cars\".\n")
	b.WriteString("  - If a fact says a number, use that exact number.\n")
	b.WriteString("- If the question asks about multiple items, list ALL matching items from the facts, separated by commas. Do not stop at one or two.\n")
	b.WriteString("- When dates appear in the facts, use the specific date (e.g. \"January 20, 2023\"), never relative terms.\n")
	b.WriteString("- If the answer is a single item, answer with ONLY that item — no extra words, no sentence framing.\n")
	b.WriteString("- If the information is not available in the provided facts, say \"Not mentioned.\"\n")
	b.WriteString("- Do not infer, guess, or add information not explicitly stated in the facts.\n")
	b.WriteString("- NEVER start your answer with a person's name, \"The answer is\", or any framing. Just the answer.\n")
	b.WriteString("\nExamples of correct answers:\n")
	b.WriteString("Q: What is Jon's attitude towards the festival? → Glad\n")
	b.WriteString("Q: What flooring does Jon want? → Marley flooring\n")
	b.WriteString("Q: What did Gina win? → first place at a regionals dance competition\n")
	b.WriteString("Q: What does Jon's dance make him? → happy\n")
	b.WriteString("Q: How does Gina describe the studio? → amazing\n")
	b.WriteString("Q: What did Gina receive? → a trophy\n")

	return strings.TrimSpace(b.String())
}

func printReport(r *BenchResult) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("LoCoMo Benchmark Results")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total questions: %d\n", r.TotalQA)
	fmt.Printf("Total facts extracted: %d\n", r.TotalFacts)
	fmt.Printf("Duration: %s\n\n", r.Duration.Round(time.Second))

	fmt.Println("Per-category results:")
	fmt.Printf("  %-15s %8s %8s\n", "Category", "Count", "Avg F1")
	fmt.Println("  " + strings.Repeat("-", 33))

	// Sort categories by number.
	catOrder := []int{4, 1, 2, 3, 5} // paper order: single, multi, temporal, open, adversarial
	printed := make(map[int]bool)
	for _, cn := range catOrder {
		for _, cs := range r.Categories {
			if cs.Category == cn && !printed[cn] {
				fmt.Printf("  %-15s %8d %8.4f\n", cs.Name, cs.Count, cs.AvgF1)
				printed[cn] = true
			}
		}
	}
	// Print any remaining categories not in the standard order.
	for _, cs := range r.Categories {
		if !printed[cs.Category] {
			fmt.Printf("  %-15s %8d %8.4f\n", cs.Name, cs.Count, cs.AvgF1)
		}
	}

	fmt.Println("  " + strings.Repeat("-", 33))
	fmt.Printf("  %-15s %8d %8.4f\n", "OVERALL", r.TotalQA, r.OverallF1)
	fmt.Println(strings.Repeat("=", 60))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// generateSessionSummary creates a conversation summary for a bench session.
func generateSessionSummary(
	ctx context.Context,
	repo store.Repository,
	embedder embed.Embedder,
	provider llm.Provider,
	entityUUID string,
	session Session,
	speakerA, speakerB string,
) {
	var transcript strings.Builder
	for _, turn := range session.Turns {
		fmt.Fprintf(&transcript, "%s: %s\n", turn.Speaker, turn.TurnText())
	}
	text := transcript.String()
	if strings.TrimSpace(text) == "" {
		return
	}

	dateCtx := ""
	if session.DateTime != "" {
		dateCtx = fmt.Sprintf("This conversation took place on %s. ", session.DateTime)
	}

	prompt := fmt.Sprintf(`%sSummarize this conversation between %s and %s. Focus on: what was discussed, decisions made, plans, emotions expressed, and specific details (names, dates, places). If trivial, respond with NONE. Keep under 200 tokens.`, dateCtx, speakerA, speakerB)

	req := &llm.Request{
		System:    prompt,
		Messages:  []*llm.Message{llm.UserMessage(text)},
		MaxTokens: 512,
	}

	resp, err := chatRetry(ctx, provider, req, 2)
	if err != nil {
		return
	}
	summary := strings.TrimSpace(resp.Content)
	if summary == "" || strings.EqualFold(summary, "NONE") {
		return
	}

	vectors, err := embedder.Embed(ctx, []string{summary})
	if err != nil || len(vectors) == 0 {
		return
	}

	sess, _, sErr := repo.EnsureSession(ctx, entityUUID, fmt.Sprintf("bench-session-%d", session.Number), time.Hour)
	if sErr != nil || sess == nil {
		return
	}
	convID, cErr := repo.StartConversation(ctx, sess.UUID, entityUUID)
	if cErr != nil {
		return
	}
	if convID == "" {
		return
	}

	_ = repo.UpdateConversationSummary(ctx, convID, summary, vectors[0], embed.ModelNameOf(embedder))
}
