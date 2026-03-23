package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"memg"
	"memg/embed"
	"memg/llm"
	"memg/memory/augment"
	"memg/proxy"
	"memg/store/sqlstore"

	// Register default providers — import for side effects.
	_ "memg/embed/gemini"
	_ "memg/embed/local"
	_ "memg/embed/onnx"
	_ "memg/embed/openai"
	_ "memg/llm/anthropic"
	_ "memg/llm/gemini"
	_ "memg/llm/openai"

	// SQLite driver.
	_ "modernc.org/sqlite"
)

// proxyConfig holds all resolved settings for the proxy subcommand.
type proxyConfig struct {
	Port                   int
	Target                 string
	Entity                 string
	DB                     string
	EmbedProvider          string
	EmbedModel             string
	EmbedBaseURL           string
	EmbedAPIKey            string
	LLMProvider            string
	LLMModel               string
	LLMBaseURL             string
	LLMAPIKey              string
	SessionTimeout         time.Duration
	WorkingMemoryTurns     int
	MemoryTokenBudget      int
	SummaryTokenBudget     int
	ConsciousMode          bool
	ConsciousLimit         int
	ConsciousCacheTTL      time.Duration
	RecallFactsLimit       int
	RecallFactThreshold    float64
	RecallSummaryLimit     int
	RecallSummaryThreshold float64
	MaxRecallCandidates    int
	PruneInterval          time.Duration
	Debug                  bool
}

// runProxy implements the "memg proxy" subcommand.
func runProxy(args []string) {
	fs := flag.NewFlagSet("proxy", flag.ExitOnError)

	configPath := fs.String("config", "", "path to config file (default: auto-detect)")
	port := fs.Int("port", 0, "port to listen on")
	target := fs.String("target", "", "upstream LLM API base URL")
	entity := fs.String("entity", "", "single-entity mode entity ID")
	dbPath := fs.String("db", "", "path to SQLite database")
	embedProvider := fs.String("embed-provider", "", "embedding provider name")
	embedModel := fs.String("embed-model", "", "embedding model")
	llmProvider := fs.String("llm-provider", "", "LLM provider for extraction calls")
	llmModel := fs.String("llm-model", "", "LLM model for extraction")
	debug := fs.Bool("debug", false, "enable verbose logging")

	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	// Load config file (auto-detect or explicit path).
	var fc *memg.FileConfig
	var cfgPath string
	var err error
	if *configPath != "" {
		fc, cfgPath, err = memg.LoadConfigFileFrom([]string{*configPath})
	} else {
		fc, cfgPath, err = memg.LoadConfigFile()
	}
	if err != nil {
		log.Fatalf("memg proxy: %v", err)
	}
	if cfgPath != "" {
		fmt.Printf("Loaded config from %s\n", cfgPath)
	}

	// Resolve values: CLI flag > config file > default.
	// A zero/empty CLI flag means "not set by user".
	cfg := proxyConfig{
		Port:                   fc.ProxyPort(8787),
		Target:                 fc.ProxyTarget("https://api.openai.com"),
		Entity:                 fc.ProxyEntity(""),
		DB:                     fc.ProxyDB("~/.memg/memory.db"),
		EmbedProvider:          fc.EmbedProviderName("openai"),
		EmbedModel:             fc.EmbedModelName(""),
		EmbedBaseURL:           fc.EmbedBaseURL(""),
		EmbedAPIKey:            fc.Embed.APIKey,
		LLMProvider:            fc.LLMProviderName("openai"),
		LLMModel:               fc.LLMModelName(""),
		LLMBaseURL:             fc.LLMBaseURL(""),
		LLMAPIKey:              fc.LLM.APIKey,
		SessionTimeout:         fc.SessionTimeoutDuration(30 * time.Minute),
		WorkingMemoryTurns:     fc.WorkingMemoryTurns(20),
		MemoryTokenBudget:      fc.MemoryTokenBudget(4000),
		SummaryTokenBudget:     fc.SummaryTokenBudget(1000),
		ConsciousMode:          fc.ConsciousMode(true),
		ConsciousLimit:         fc.ConsciousLimitVal(10),
		ConsciousCacheTTL:      fc.ConsciousCacheTTLDuration(30 * time.Second),
		RecallFactsLimit:       fc.RecallLimit(100),
		RecallFactThreshold:    fc.RecallThresholdVal(0.10),
		RecallSummaryLimit:     fc.RecallSummaryLimitVal(5),
		RecallSummaryThreshold: fc.RecallSummaryThresholdVal(0.30),
		MaxRecallCandidates:    10000,
		PruneInterval:          fc.PruneIntervalDuration(5 * time.Minute),
		Debug:                  fc.Debug,
	}

	// CLI flags override config file values.
	if *port != 0 {
		cfg.Port = *port
	}
	if *target != "" {
		cfg.Target = *target
	}
	if *entity != "" {
		cfg.Entity = *entity
	}
	if *dbPath != "" {
		cfg.DB = *dbPath
	}
	if *embedProvider != "" {
		cfg.EmbedProvider = *embedProvider
	}
	if *embedModel != "" {
		cfg.EmbedModel = *embedModel
	}
	if *llmProvider != "" {
		cfg.LLMProvider = *llmProvider
	}
	if *llmModel != "" {
		cfg.LLMModel = *llmModel
	}
	if *debug {
		cfg.Debug = true
	}

	if err := startProxy(cfg); err != nil {
		log.Fatalf("memg proxy: %v", err)
	}
}

func startProxy(cfg proxyConfig) error {
	ctx := context.Background()

	// Expand ~ in the database path.
	dbPath := expandHome(cfg.DB)

	// Ensure the parent directory exists.
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return fmt.Errorf("create db directory: %w", err)
	}

	// Open SQLite database with connection pool limits.
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	db.SetMaxOpenConns(4)
	db.SetMaxIdleConns(2)
	db.SetConnMaxLifetime(30 * time.Minute)

	// Create repository and run migrations.
	repo := sqlstore.NewSQLite(db)
	if err := repo.Migrate(ctx); err != nil {
		return fmt.Errorf("migrate database: %w", err)
	}

	// Create an HTTP client that marks requests as internal to prevent
	// the proxy from intercepting its own extraction/embedding calls.
	internalHTTP := proxy.NewInternalHTTPClient()

	// Create embedder.
	emb, err := embed.NewEmbedder(cfg.EmbedProvider, embed.ProviderConfig{
		Model:      cfg.EmbedModel,
		BaseURL:    cfg.EmbedBaseURL,
		APIKey:     cfg.EmbedAPIKey,
		HTTPClient: internalHTTP,
	})
	if err != nil {
		return fmt.Errorf("create embedder (%s): %w", cfg.EmbedProvider, err)
	}

	// Create LLM provider for extraction calls.
	provider, err := llm.NewProvider(cfg.LLMProvider, llm.ProviderConfig{
		Model:      cfg.LLMModel,
		BaseURL:    cfg.LLMBaseURL,
		APIKey:     cfg.LLMAPIKey,
		HTTPClient: internalHTTP,
	})
	if err != nil {
		return fmt.Errorf("create llm provider (%s): %w", cfg.LLMProvider, err)
	}

	// Create augmentation pipeline and register the built-in extraction stage.
	pipeline := augment.NewPipeline(repo)
	extractionStage := proxy.NewDefaultExtractionStage(provider, emb)
	pipeline.AddStage(extractionStage)

	// Parse target URL.
	targetURL, err := url.Parse(cfg.Target)
	if err != nil {
		return fmt.Errorf("parse target URL %q: %w", cfg.Target, err)
	}

	// Create proxy server.
	srv := proxy.NewServer(proxy.Config{
		Target:                 targetURL,
		Repo:                   repo,
		Embedder:               emb,
		Provider:               provider,
		Pipeline:               pipeline,
		DefaultEntity:          cfg.Entity,
		SessionTimeout:         cfg.SessionTimeout,
		WorkingMemoryTurns:     cfg.WorkingMemoryTurns,
		MemoryTokenBudget:      cfg.MemoryTokenBudget,
		SummaryTokenBudget:     cfg.SummaryTokenBudget,
		ConsciousMode:          cfg.ConsciousMode,
		ConsciousLimit:         cfg.ConsciousLimit,
		ConsciousCacheTTL:      cfg.ConsciousCacheTTL,
		RecallFactsLimit:       cfg.RecallFactsLimit,
		RecallSummaryLimit:     cfg.RecallSummaryLimit,
		RecallFactThreshold:    cfg.RecallFactThreshold,
		RecallSummaryThreshold: cfg.RecallSummaryThreshold,
		MaxRecallCandidates:    cfg.MaxRecallCandidates,
		Debug:                  cfg.Debug,
	})

	// Start the server in a goroutine.
	addr := fmt.Sprintf(":%d", cfg.Port)
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Start(addr)
	}()

	// Print startup information.
	fmt.Printf("MemG proxy listening on %s, forwarding to %s\n", addr, cfg.Target)
	fmt.Printf("Set OPENAI_BASE_URL=http://localhost:%d/v1 to add memory to your app\n", cfg.Port)
	if cfg.Entity != "" {
		fmt.Printf("Single-entity mode: all requests attributed to %q\n", cfg.Entity)
	}
	if cfg.Debug {
		fmt.Println("Debug logging enabled")
	}

	// Wait for interrupt signal or server error.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		fmt.Printf("\nReceived %s, shutting down...\n", sig)
	case err := <-errCh:
		if err != nil {
			return fmt.Errorf("server error: %w", err)
		}
	}

	// Graceful shutdown with a 10-second timeout.
	shutdownCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("memg proxy: shutdown error: %v", err)
	}

	// Drain the augmentation pipeline.
	pipeline.Shutdown()

	// Close the repository (and underlying database).
	if err := repo.Close(); err != nil {
		log.Printf("memg proxy: close repo: %v", err)
	}

	fmt.Println("Shut down complete.")
	return nil
}

// expandHome replaces a leading ~ with the user's home directory.
func expandHome(path string) string {
	if len(path) == 0 || path[0] != '~' {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, path[1:])
}
