// Package onnx provides an in-process ONNX Runtime embedding provider that
// runs BERT-family models locally without external services.
//
// Register as provider "onnx" via init(). The ONNX Runtime shared library
// (libonnxruntime.dylib / .so / .dll) must be available at runtime.
//
// Configuration via embed.ProviderConfig:
//
//   - BaseURL: path to model directory containing model.onnx and vocab.txt
//     (default: ~/.memg/models/all-MiniLM-L6-v2)
//   - Model: model name for identification (default: all-MiniLM-L6-v2)
//   - Dimension: override auto-detected dimension (0 = auto)
//
// The ONNX Runtime library path is resolved from:
//  1. ONNX_RUNTIME_LIB env var (full path to shared library)
//  2. Common system paths (platform-dependent)
package onnx

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"memg/embed"
)

const (
	providerName   = "onnx"
	defaultModel   = "all-MiniLM-L6-v2"
	defaultSeqLen  = 256
	defaultDim     = 384
	maxBatchSize   = 64
)

var (
	ortOnce sync.Once
	ortErr  error
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Embedder implements embed.Embedder using ONNX Runtime for in-process inference.
type Embedder struct {
	tokenizer *BertTokenizer
	modelPath string
	modelName string
	dimension int
	seqLen    int
	mu        sync.Mutex // serialize inference calls
}

// New creates an ONNX embedding provider. It initializes the ONNX Runtime,
// loads the tokenizer vocabulary, and validates the model file exists.
func New(cfg embed.ProviderConfig) (*Embedder, error) {
	if err := initRuntime(); err != nil {
		return nil, fmt.Errorf("onnx: init runtime: %w", err)
	}

	modelDir := cfg.BaseURL
	if modelDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("onnx: resolve home dir: %w", err)
		}
		modelDir = filepath.Join(home, ".memg", "models", defaultModel)
	}

	modelPath := filepath.Join(modelDir, "model.onnx")
	vocabPath := filepath.Join(modelDir, "vocab.txt")

	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("onnx: model file not found at %s: %w", modelPath, err)
	}
	if _, err := os.Stat(vocabPath); err != nil {
		return nil, fmt.Errorf("onnx: vocab file not found at %s: %w", vocabPath, err)
	}

	tokenizer, err := NewBertTokenizer(vocabPath, defaultSeqLen)
	if err != nil {
		return nil, err
	}

	modelName := cfg.Model
	if modelName == "" {
		modelName = defaultModel
	}

	dimension := cfg.Dimension
	if dimension == 0 {
		dimension = defaultDim
	}

	return &Embedder{
		tokenizer: tokenizer,
		modelPath: modelPath,
		modelName: modelName,
		dimension: dimension,
		seqLen:    defaultSeqLen,
	}, nil
}

// Embed converts texts into normalized embedding vectors using the ONNX model.
func (e *Embedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, 0, len(texts))

	// Process in batches to bound memory usage.
	for i := 0; i < len(texts); i += maxBatchSize {
		end := i + maxBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		vecs, err := e.embedBatch(batch)
		if err != nil {
			return nil, fmt.Errorf("onnx: embed batch %d-%d: %w", i, end, err)
		}
		results = append(results, vecs...)
	}

	return results, nil
}

// Dimension returns the embedding vector length.
func (e *Embedder) Dimension() int { return e.dimension }

// ModelName returns the model identifier.
func (e *Embedder) ModelName() string { return e.modelName }

func (e *Embedder) embedBatch(texts []string) ([][]float32, error) {
	batchSize := len(texts)
	inputIDs, attentionMask, tokenTypeIDs := e.tokenizer.EncodeBatch(texts)

	idShape := ort.NewShape(int64(batchSize), int64(e.seqLen))
	outShape := ort.NewShape(int64(batchSize), int64(e.seqLen), int64(e.dimension))

	inputIDsTensor, err := ort.NewTensor(idShape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(idShape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	tokenTypeIDsTensor, err := ort.NewTensor(idShape, tokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("create token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDsTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Serialize session creation and inference. ONNX Runtime sessions are
	// thread-safe for Run(), but we create short-lived sessions per batch
	// to avoid holding model memory when idle.
	e.mu.Lock()
	defer e.mu.Unlock()

	session, err := ort.NewAdvancedSession(e.modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"token_embeddings"},
		[]ort.Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run inference: %w", err)
	}

	rawOutput := outputTensor.GetData()

	// Mean pool + L2 normalize each embedding in the batch.
	vectors := make([][]float32, batchSize)
	tokensPerSample := e.seqLen * e.dimension
	for i := 0; i < batchSize; i++ {
		sampleStart := i * tokensPerSample
		sampleEnd := sampleStart + tokensPerSample
		sampleEmbeddings := rawOutput[sampleStart:sampleEnd]

		maskStart := i * e.seqLen
		maskEnd := maskStart + e.seqLen
		sampleMask := attentionMask[maskStart:maskEnd]

		pooled := meanPool(sampleEmbeddings, sampleMask, e.seqLen, e.dimension)
		vectors[i] = l2Normalize(pooled)
	}

	return vectors, nil
}

// initRuntime initializes the ONNX Runtime environment exactly once.
func initRuntime() error {
	ortOnce.Do(func() {
		libPath := resolveRuntimeLib()
		if libPath == "" {
			ortErr = fmt.Errorf("ONNX Runtime shared library not found. " +
				"Set ONNX_RUNTIME_LIB env var to the full path, or install it to a standard location")
			return
		}
		ort.SetSharedLibraryPath(libPath)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// resolveRuntimeLib finds the ONNX Runtime shared library path.
func resolveRuntimeLib() string {
	// 1. Explicit env var.
	if p := os.Getenv("ONNX_RUNTIME_LIB"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// 2. Search common paths per platform.
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
		// Also check ONNX_RUNTIME_DIR for extracted tarballs.
		if dir := os.Getenv("ONNX_RUNTIME_DIR"); dir != "" {
			candidates = append([]string{filepath.Join(dir, "lib", "libonnxruntime.dylib")}, candidates...)
		}
	case "linux":
		candidates = []string{
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
		}
		if dir := os.Getenv("ONNX_RUNTIME_DIR"); dir != "" {
			candidates = append([]string{filepath.Join(dir, "lib", "libonnxruntime.so")}, candidates...)
		}
	case "windows":
		candidates = []string{
			`C:\Program Files\onnxruntime\lib\onnxruntime.dll`,
		}
		if dir := os.Getenv("ONNX_RUNTIME_DIR"); dir != "" {
			candidates = append([]string{filepath.Join(dir, "lib", "onnxruntime.dll")}, candidates...)
		}
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	return ""
}
