package main

import (
	_ "embed"
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

//go:embed dashboard.html
var dashboardHTML string

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Scenario struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Group       string   `json:"group"`
	Args        []string `json:"args"`
}

type RunStatus string

const (
	StatusRunning RunStatus = "running"
	StatusDone    RunStatus = "done"
	StatusFailed  RunStatus = "failed"
)

type CatScore struct {
	Category int     `json:"Category"`
	Name     string  `json:"Name"`
	Count    int     `json:"Count"`
	SumF1    float64 `json:"SumF1"`
	AvgF1    float64 `json:"AvgF1"`
}

type BenchResult struct {
	Categories []CatScore `json:"Categories"`
	OverallF1  float64    `json:"OverallF1"`
	TotalQA    int        `json:"TotalQA"`
	TotalFacts int        `json:"TotalFacts"`
	Duration   int64      `json:"Duration"`
}

type Run struct {
	ID         string       `json:"id"`
	ScenarioID string       `json:"scenario_id"`
	Scenario   string       `json:"scenario"`
	Status     RunStatus    `json:"status"`
	StartedAt  time.Time    `json:"started_at"`
	FinishedAt *time.Time   `json:"finished_at,omitempty"`
	DurationS  string       `json:"duration,omitempty"`
	Result     *BenchResult `json:"result,omitempty"`
	Error      string       `json:"error,omitempty"`
	Args       []string     `json:"args"`

	mu        sync.Mutex
	output    []string
	listeners []chan string
	cancel    context.CancelFunc
}

func (r *Run) Subscribe() ([]string, chan string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	snap := make([]string, len(r.output))
	copy(snap, r.output)
	if r.Status == StatusDone || r.Status == StatusFailed {
		return snap, nil
	}
	ch := make(chan string, 256)
	r.listeners = append(r.listeners, ch)
	return snap, ch
}

func (r *Run) appendLine(line string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.output = append(r.output, line)
	for _, ch := range r.listeners {
		select {
		case ch <- line:
		default:
		}
	}
}

func (r *Run) finish(status RunStatus, result *BenchResult, errMsg string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	r.Status = status
	r.FinishedAt = &now
	r.DurationS = now.Sub(r.StartedAt).Round(time.Second).String()
	r.Result = result
	r.Error = errMsg
	for _, ch := range r.listeners {
		close(ch)
	}
	r.listeners = nil
}

// ---------------------------------------------------------------------------
// RunManager
// ---------------------------------------------------------------------------

type RunManager struct {
	mu         sync.RWMutex
	runs       map[string]*Run
	seq        atomic.Int64
	binPath    string
	projectDir string
}

func NewRunManager(binPath, projectDir string) *RunManager {
	return &RunManager{
		runs:       make(map[string]*Run),
		binPath:    binPath,
		projectDir: projectDir,
	}
}

func (m *RunManager) Start(scenarioID, scenarioName string, args []string) *Run {
	id := fmt.Sprintf("run-%d", m.seq.Add(1))
	r := &Run{
		ID:         id,
		ScenarioID: scenarioID,
		Scenario:   scenarioName,
		Status:     StatusRunning,
		StartedAt:  time.Now(),
		Args:       args,
	}
	m.mu.Lock()
	m.runs[id] = r
	m.mu.Unlock()

	go m.execute(r)
	return r
}

func (m *RunManager) execute(r *Run) {
	resultFile := filepath.Join(os.TempDir(), fmt.Sprintf("memg_bench_%s.json", r.ID))
	args := append([]string{"bench", "-output", resultFile}, r.Args...)

	ctx, cancel := context.WithCancel(context.Background())
	r.mu.Lock()
	r.cancel = cancel
	r.mu.Unlock()

	cmd := exec.CommandContext(ctx, m.binPath, args...)
	cmd.Dir = m.projectDir
	cmd.Env = os.Environ()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		r.finish(StatusFailed, nil, err.Error())
		cancel()
		return
	}
	cmd.Stderr = cmd.Stdout

	if err := cmd.Start(); err != nil {
		r.finish(StatusFailed, nil, err.Error())
		cancel()
		return
	}

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 256*1024)
	for scanner.Scan() {
		r.appendLine(scanner.Text())
	}

	waitErr := cmd.Wait()
	cancel()

	if waitErr != nil {
		r.finish(StatusFailed, nil, waitErr.Error())
		return
	}

	var result *BenchResult
	if data, readErr := os.ReadFile(resultFile); readErr == nil {
		var br BenchResult
		if json.Unmarshal(data, &br) == nil {
			result = &br
		}
		os.Remove(resultFile)
	}
	r.finish(StatusDone, result, "")
}

func (m *RunManager) Get(id string) *Run {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.runs[id]
}

func (m *RunManager) List() []*Run {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]*Run, 0, len(m.runs))
	for _, r := range m.runs {
		out = append(out, r)
	}
	return out
}

func (m *RunManager) Stop(id string) bool {
	r := m.Get(id)
	if r == nil {
		return false
	}
	r.mu.Lock()
	c := r.cancel
	r.mu.Unlock()
	if c != nil {
		c()
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

var scenarios = []Scenario{
	// Category tests (fast, on conv-30)
	{ID: "quick-smoke", Name: "Quick Smoke Test", Description: "conv-30, all categories", Group: "Category Tests",
		Args: []string{"-conversations", "conv-30"}},
	{ID: "cat-single", Name: "Single-Hop", Description: "Category 4 — basic fact recall (conv-30)", Group: "Category Tests",
		Args: []string{"-conversations", "conv-30", "-categories", "4"}},
	{ID: "cat-multi", Name: "Multi-Hop", Description: "Category 1 — multi-fact reasoning (conv-30)", Group: "Category Tests",
		Args: []string{"-conversations", "conv-30", "-categories", "1"}},
	{ID: "cat-temporal", Name: "Temporal", Description: "Category 2 — time-sensitive recall (conv-30)", Group: "Category Tests",
		Args: []string{"-conversations", "conv-30", "-categories", "2"}},
	{ID: "cat-open", Name: "Open-Domain", Description: "Category 3 — broad knowledge (conv-30)", Group: "Category Tests",
		Args: []string{"-conversations", "conv-30", "-categories", "3"}},
	{ID: "cat-adversarial", Name: "Adversarial", Description: "Category 5 — hallucination resistance (conv-30)", Group: "Category Tests",
		Args: []string{"-conversations", "conv-30", "-categories", "5"}},

	// Recall tuning (conv-30)
	{ID: "recall-10", Name: "Recall Limit 10", Description: "max 10 facts recalled per question", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-recall-limit", "10"}},
	{ID: "recall-25", Name: "Recall Limit 25", Description: "max 25 facts recalled per question", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-recall-limit", "25"}},
	{ID: "recall-50", Name: "Recall Limit 50", Description: "default recall limit", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-recall-limit", "50"}},
	{ID: "recall-100", Name: "Recall Limit 100", Description: "max 100 facts recalled per question", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-recall-limit", "100"}},
	{ID: "thresh-low", Name: "Threshold 0.01", Description: "very permissive recall threshold", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-recall-threshold", "0.01"}},
	{ID: "thresh-high", Name: "Threshold 0.20", Description: "strict recall threshold", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-recall-threshold", "0.20"}},
	{ID: "cand-500", Name: "Max Candidates 500", Description: "narrow candidate scan", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-max-candidates", "500"}},
	{ID: "cand-5000", Name: "Max Candidates 5000", Description: "wider candidate scan", Group: "Recall Tuning",
		Args: []string{"-conversations", "conv-30", "-max-candidates", "5000"}},

	// Scale tests
	{ID: "three-conv", Name: "Three Conversations", Description: "conv-30 + conv-26 + conv-44", Group: "Scale Tests",
		Args: []string{"-conversations", "conv-30,conv-26,conv-44"}},
	{ID: "full", Name: "Full Benchmark", Description: "All 10 conversations, all categories", Group: "Scale Tests",
		Args: []string{}},
}

// ---------------------------------------------------------------------------
// HTTP Handlers
// ---------------------------------------------------------------------------

type server struct {
	mgr *RunManager
}

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(dashboardHTML))
}

func (s *server) handleScenarios(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(scenarios)
}

func (s *server) handleStartRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}

	var body struct {
		ScenarioID    string `json:"scenario_id"`
		Provider      string `json:"provider"`
		Model         string `json:"model"`
		APIKey        string `json:"api_key"`
		EmbedProvider string `json:"embed_provider"`
		EmbedModel    string `json:"embed_model"`
		EmbedAPIKey   string `json:"embed_api_key"`
		DataPath      string `json:"data_path"`
		ExtraArgs     string `json:"extra_args"`
		Debug         bool   `json:"debug"`
	}

	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Find scenario.
	var sc *Scenario
	for i := range scenarios {
		if scenarios[i].ID == body.ScenarioID {
			sc = &scenarios[i]
			break
		}
	}

	name := "custom"
	var baseArgs []string
	if sc != nil {
		name = sc.Name
		baseArgs = append(baseArgs, sc.Args...)
	}

	// Add global settings as flags.
	dp := body.DataPath
	if dp == "" {
		dp = "bench/testdata/locomo10.json"
	}
	args := []string{"-data", dp}

	if body.Provider != "" {
		args = append(args, "-llm-provider", body.Provider)
	}
	if body.Model != "" {
		args = append(args, "-llm-model", body.Model)
	}
	if body.APIKey != "" {
		args = append(args, "-llm-api-key", body.APIKey)
	}
	if body.EmbedProvider != "" {
		args = append(args, "-embed-provider", body.EmbedProvider)
	}
	if body.EmbedModel != "" {
		args = append(args, "-embed-model", body.EmbedModel)
	}
	if body.EmbedAPIKey != "" {
		args = append(args, "-embed-api-key", body.EmbedAPIKey)
	}
	if body.Debug {
		args = append(args, "-debug")
	}

	args = append(args, baseArgs...)

	// Parse extra args.
	if extra := strings.TrimSpace(body.ExtraArgs); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}

	run := s.mgr.Start(body.ScenarioID, name, args)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": run.ID, "scenario": name})
}

func (s *server) handleListRuns(w http.ResponseWriter, r *http.Request) {
	runs := s.mgr.List()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(runs)
}

func (s *server) handleGetRun(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/run/")
	if id == "" {
		http.Error(w, "missing run id", http.StatusBadRequest)
		return
	}
	run := s.mgr.Get(id)
	if run == nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(run)
}

func (s *server) handleStream(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/stream/")
	run := s.mgr.Get(id)
	if run == nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	existing, ch := run.Subscribe()

	for _, line := range existing {
		fmt.Fprintf(w, "data: %s\n\n", line)
	}
	flusher.Flush()

	if ch == nil {
		// Run already finished — send final status.
		run.mu.Lock()
		status := run.Status
		run.mu.Unlock()
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", status)
		flusher.Flush()
		return
	}

	ctx := r.Context()
	for {
		select {
		case line, ok := <-ch:
			if !ok {
				run.mu.Lock()
				status := run.Status
				run.mu.Unlock()
				fmt.Fprintf(w, "event: done\ndata: %s\n\n", status)
				flusher.Flush()
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", line)
			flusher.Flush()
		case <-ctx.Done():
			return
		}
	}
}

func (s *server) handleStopRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/api/stop/")
	if s.mgr.Stop(id) {
		w.Write([]byte(`{"ok":true}`))
	} else {
		http.Error(w, "not found or already stopped", http.StatusNotFound)
	}
}

// ---------------------------------------------------------------------------
// Build + main
// ---------------------------------------------------------------------------

func buildBinary(projectDir string) (string, error) {
	binPath := filepath.Join(os.TempDir(), "memg_bench_dashboard_bin")
	if runtime.GOOS == "windows" {
		binPath += ".exe"
	}

	log.Printf("Building memg binary...")
	cmd := exec.Command("go", "build", "-o", binPath, "./cmd/memg")
	cmd.Dir = projectDir
	cmd.Env = os.Environ()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("go build failed:\n%s\n%w", out, err)
	}
	log.Printf("Built: %s", binPath)
	return binPath, nil
}

func main() {
	projectDir, _ := os.Getwd()
	if _, err := os.Stat(filepath.Join(projectDir, "go.mod")); err != nil {
		log.Fatal("Run this from the MemG project root (where go.mod is)")
	}

	binPath, err := buildBinary(projectDir)
	if err != nil {
		log.Fatal(err)
	}

	mgr := NewRunManager(binPath, projectDir)
	srv := &server{mgr: mgr}

	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.handleIndex)
	mux.HandleFunc("/api/scenarios", srv.handleScenarios)
	mux.HandleFunc("/api/run/start", srv.handleStartRun)
	mux.HandleFunc("/api/runs", srv.handleListRuns)
	mux.HandleFunc("/api/run/", srv.handleGetRun)
	mux.HandleFunc("/api/stream/", srv.handleStream)
	mux.HandleFunc("/api/stop/", srv.handleStopRun)

	ln, err := net.Listen("tcp", "127.0.0.1:8177")
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()
	url := "http://" + addr

	log.Printf("Dashboard: %s", url)

	if runtime.GOOS == "darwin" {
		exec.Command("open", url).Start()
	} else if runtime.GOOS == "linux" {
		exec.Command("xdg-open", url).Start()
	}

	httpSrv := &http.Server{Handler: mux}
	go httpSrv.Serve(ln)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	log.Println("Shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	httpSrv.Shutdown(ctx)
}
