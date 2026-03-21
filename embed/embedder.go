// Package embed defines the contract for text-to-vector embedding services.
package embed

import "context"

// Embedder converts text into dense vector representations.
type Embedder interface {
	// Embed produces one embedding vector per input text.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimension returns the length of each embedding vector.
	Dimension() int
}

// Modeler is implemented by embedders that can identify the model that
// produced their vectors.
type Modeler interface {
	ModelName() string
}

// ModelNameOf returns the embedder's model name when available.
func ModelNameOf(embedder Embedder) string {
	if modeler, ok := embedder.(Modeler); ok {
		return modeler.ModelName()
	}
	return ""
}
