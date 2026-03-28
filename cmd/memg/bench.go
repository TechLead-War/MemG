package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"memg/bench"

	// Register LLM providers.
	_ "memg/llm/anthropic"
	_ "memg/llm/azureopenai"
	_ "memg/llm/bedrock"
	_ "memg/llm/deepseek"
	_ "memg/llm/gemini"
	_ "memg/llm/groq"
	_ "memg/llm/ollama"
	_ "memg/llm/openai"
	_ "memg/llm/togetherai"
	_ "memg/llm/xai"

	// Register embedding providers.
	_ "memg/embed/azureopenai"
	_ "memg/embed/bedrock"
	_ "memg/embed/cohere"
	_ "memg/embed/gemini"
	_ "memg/embed/huggingface"
	_ "memg/embed/ollama"
	_ "memg/embed/openai"
	_ "memg/embed/openaicompat"
	_ "memg/embed/togetherai"
	_ "memg/embed/voyageai"

	// SQLite driver.
	_ "modernc.org/sqlite"
)

func runBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)

	dataPath := fs.String("data", "", "path to locomo10.json (default: bench/testdata/locomo10.json)")
	dbPath := fs.String("db", "", "path to SQLite database (default: temp file)")
	llmProvider := fs.String("llm-provider", "openai", "LLM provider for extraction and QA")
	llmModel := fs.String("llm-model", "gpt-4o-mini", "LLM model")
	llmAPIKey := fs.String("llm-api-key", "", "LLM API key (default: env OPENAI_API_KEY)")
	embedProvider := fs.String("embed-provider", "openai", "embedding provider")
	embedModel := fs.String("embed-model", "", "embedding model (default: provider default)")
	embedAPIKey := fs.String("embed-api-key", "", "embed API key (default: same as LLM or env)")
	recallLimit := fs.Int("recall-limit", 50, "max facts recalled per question")
	recallThresh := fs.Float64("recall-threshold", 0.05, "minimum score for recalled facts")
	maxCandidates := fs.Int("max-candidates", 500, "max candidate facts for recall scan")
	categories := fs.String("categories", "", "comma-separated category numbers to evaluate (default: all)")
	conversations := fs.String("conversations", "", "comma-separated sample IDs to evaluate (default: all)")
	concurrency := fs.Int("concurrency", 4, "number of concurrent QA evaluations")
	outputJSON := fs.String("output", "", "write results to JSON file")
	debug := fs.Bool("debug", false, "verbose logging")

	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	// Resolve data path.
	dp := *dataPath
	if dp == "" {
		// Try relative to working directory first, then relative to binary.
		candidates := []string{
			"bench/testdata/locomo10.json",
			filepath.Join(filepath.Dir(os.Args[0]), "bench", "testdata", "locomo10.json"),
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				dp = c
				break
			}
		}
		if dp == "" {
			log.Fatal("locomo10.json not found. Use -data flag or place it at bench/testdata/locomo10.json")
		}
	}

	// Parse category filter.
	var catFilter []int
	if *categories != "" {
		for _, s := range strings.Split(*categories, ",") {
			s = strings.TrimSpace(s)
			n, err := strconv.Atoi(s)
			if err != nil {
				log.Fatalf("invalid category %q: %v", s, err)
			}
			catFilter = append(catFilter, n)
		}
	}

	// Parse conversation filter.
	var convFilter []string
	if *conversations != "" {
		for _, s := range strings.Split(*conversations, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				convFilter = append(convFilter, s)
			}
		}
	}

	// Resolve API keys.
	lKey := *llmAPIKey
	if lKey == "" {
		lKey = os.Getenv("OPENAI_API_KEY")
	}
	eKey := *embedAPIKey
	if eKey == "" {
		eKey = lKey
	}

	cfg := bench.RunConfig{
		DataPath:      dp,
		DBPath:        *dbPath,
		LLMProvider:   *llmProvider,
		LLMModel:      *llmModel,
		LLMAPIKey:     lKey,
		EmbedProvider: *embedProvider,
		EmbedModel:    *embedModel,
		EmbedAPIKey:   eKey,
		RecallLimit:   *recallLimit,
		RecallThresh:  *recallThresh,
		MaxCandidates: *maxCandidates,
		Categories:    catFilter,
		Conversations: convFilter,
		QAConcurrency: *concurrency,
		Debug:         *debug,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nInterrupted, shutting down...")
		cancel()
	}()

	fmt.Println("MemG LoCoMo Benchmark")
	fmt.Println(strings.Repeat("-", 40))

	result, err := bench.Run(ctx, cfg)
	if err != nil {
		log.Fatalf("benchmark failed: %v", err)
	}

	if *outputJSON != "" && result != nil {
		data, jErr := json.MarshalIndent(result, "", "  ")
		if jErr != nil {
			log.Printf("failed to marshal results: %v", jErr)
		} else if wErr := os.WriteFile(*outputJSON, data, 0o644); wErr != nil {
			log.Printf("failed to write results: %v", wErr)
		} else {
			fmt.Printf("\nResults written to %s\n", *outputJSON)
		}
	}
}
