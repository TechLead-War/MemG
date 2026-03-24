package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"memg/embed"
	"memg/llm"
	"memg/mcp"
	"memg/memory/augment"
	"memg/proxy"
	"memg/store/sqlstore"

	// Register default providers.
	_ "memg/embed/onnx"
	_ "memg/embed/openai"
	_ "memg/llm/anthropic"
	_ "memg/llm/gemini"
	_ "memg/llm/openai"

	// SQLite driver.
	_ "modernc.org/sqlite"
)

// runMCP implements the "memg mcp" subcommand.
func runMCP(args []string) {
	fs := flag.NewFlagSet("mcp", flag.ExitOnError)

	port := fs.Int("port", 8686, "port to listen on")
	dbPath := fs.String("db", "~/.memg/memory.db", "path to SQLite database")
	embedProvider := fs.String("embed-provider", "openai", "embedding provider name")
	embedModel := fs.String("embed-model", "", "embedding model (empty = provider default)")
	llmProvider := fs.String("llm-provider", "", "LLM provider for extraction (enables extract_from_messages)")
	llmModel := fs.String("llm-model", "", "LLM model for extraction")
	debug := fs.Bool("debug", false, "enable verbose logging")

	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	if err := startMCP(*port, *dbPath, *embedProvider, *embedModel, *llmProvider, *llmModel, *debug); err != nil {
		log.Fatalf("memg mcp: %v", err)
	}
}

func startMCP(port int, dbPath, embedProviderName, embedModel, llmProviderName, llmModelName string, debug bool) error {
	ctx := context.Background()

	dbPath = expandHome(dbPath)

	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return fmt.Errorf("create db directory: %w", err)
	}

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}

	repo := sqlstore.NewSQLite(db)
	if err := repo.Migrate(ctx); err != nil {
		return fmt.Errorf("migrate database: %w", err)
	}

	emb, err := embed.NewEmbedder(embedProviderName, embed.ProviderConfig{
		Model: embedModel,
	})
	if err != nil {
		return fmt.Errorf("create embedder (%s): %w", embedProviderName, err)
	}
	probeCtx, cancelProbe := context.WithTimeout(ctx, 15*time.Second)
	defer cancelProbe()
	if err := embed.Probe(probeCtx, emb); err != nil {
		return fmt.Errorf("embedder health check failed: %w", err)
	}

	mcpCfg := mcp.Config{
		Repo:     repo,
		Embedder: emb,
		Debug:    debug,
	}

	// If an LLM provider is configured, set up the extraction pipeline so
	// that extract_from_messages is available to SDK clients.
	if llmProviderName != "" {
		provider, provErr := llm.NewProvider(llmProviderName, llm.ProviderConfig{
			Model: llmModelName,
		})
		if provErr != nil {
			return fmt.Errorf("create llm provider (%s): %w", llmProviderName, provErr)
		}

		pipeline := augment.NewPipeline(repo)
		pipeline.Embedder = emb
		extractionStage := proxy.NewDefaultExtractionStage(provider, emb)
		pipeline.AddStage(extractionStage)
		mcpCfg.Pipeline = pipeline

		fmt.Printf("Extraction pipeline enabled (provider: %s)\n", llmProviderName)
	}

	srv := mcp.NewServer(mcpCfg)

	addr := fmt.Sprintf(":%d", port)
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Start(addr)
	}()

	fmt.Printf("MemG MCP server listening on %s\n", addr)
	fmt.Printf("MCP endpoint: http://localhost:%d/mcp/\n", port)
	if debug {
		fmt.Println("Debug logging enabled")
	}

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

	shutdownCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("memg mcp: shutdown error: %v", err)
	}

	if err := repo.Close(); err != nil {
		log.Printf("memg mcp: close repo: %v", err)
	}

	fmt.Println("Shut down complete.")
	return nil
}
