// Package local provides an embedding client that connects to the local
// Python gRPC embedding service powered by sentence-transformers.
package local

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"memg/embed"
	"memg/embed/local/pb"
)

const (
	defaultAddress = "localhost:50051"
	providerName   = "local"
)

func init() {
	embed.RegisterEmbedder(providerName, func(cfg embed.ProviderConfig) (embed.Embedder, error) {
		return New(cfg)
	})
}

// Client implements embed.Embedder by calling the local gRPC embedding service.
type Client struct {
	conn      *grpc.ClientConn
	client    pb.EmbedderServiceClient
	model     string
	dimension int
}

// New creates a local embedding client. It connects to the gRPC service and
// auto-detects the model dimension via the Info() call.
func New(cfg embed.ProviderConfig) (*Client, error) {
	address := cfg.BaseURL
	if address == "" {
		address = defaultAddress
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("local: connect to %s: %w", address, err)
	}

	client := pb.NewEmbedderServiceClient(conn)

	// Auto-detect model and dimension from the running service.
	info, err := client.Info(ctx, &pb.InfoRequest{})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("local: get model info: %w", err)
	}

	dimension := int(info.Dimension)
	if cfg.Dimension > 0 {
		dimension = cfg.Dimension
	}

	return &Client{
		conn:      conn,
		client:    client,
		model:     info.Model,
		dimension: dimension,
	}, nil
}

// ModelName returns the detected model identifier from the local service.
func (c *Client) ModelName() string { return c.model }

// Embed sends texts to the local embedding service and returns the vectors.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	resp, err := c.client.Embed(ctx, &pb.EmbedRequest{Texts: texts})
	if err != nil {
		return nil, fmt.Errorf("local: embed: %w", err)
	}

	vectors := make([][]float32, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		vectors[i] = emb.Values
	}
	return vectors, nil
}

// Dimension returns the embedding vector length reported by the service.
func (c *Client) Dimension() int { return c.dimension }

// Close shuts down the gRPC connection.
func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
