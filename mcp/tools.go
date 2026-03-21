package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"memg/memory"
	"memg/store"
)

// toolDefinitions returns the MCP tool definitions exposed by this server.
func toolDefinitions() []toolDefinition {
	return []toolDefinition{
		{
			Name:        "add_memories",
			Description: "Store new memories (facts) for an entity. Each memory is embedded and persisted with optional metadata for type, significance, and tagging.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"entity_id": map[string]any{
						"type":        "string",
						"description": "External identifier for the entity (e.g. user ID, agent ID).",
					},
					"memories": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"content": map[string]any{
									"type":        "string",
									"description": "The memory content to store.",
								},
								"type": map[string]any{
									"type":        "string",
									"enum":        []string{"identity", "event", "pattern"},
									"description": "Kind of knowledge: identity (enduring attribute), event (point-in-time), pattern (behavioral tendency). Defaults to identity.",
								},
								"significance": map[string]any{
									"type":        "string",
									"enum":        []string{"low", "medium", "high"},
									"description": "Controls TTL: low (~1 week), medium (~1 month), high (never expires). Defaults to medium.",
								},
								"tag": map[string]any{
									"type":        "string",
									"description": "Optional category label for filtering (e.g. 'preference', 'skill', 'medical').",
								},
							},
							"required": []string{"content"},
						},
						"description": "List of memories to store.",
					},
				},
				"required": []string{"entity_id", "memories"},
			},
		},
		{
			Name:        "search_memories",
			Description: "Search memories for an entity using semantic hybrid search (dense vector + BM25 lexical scoring with adaptive Kneedle cutoff).",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"entity_id": map[string]any{
						"type":        "string",
						"description": "External identifier for the entity.",
					},
					"query": map[string]any{
						"type":        "string",
						"description": "Natural language search query.",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "Maximum number of results to return. Defaults to 10.",
					},
				},
				"required": []string{"entity_id", "query"},
			},
		},
		{
			Name:        "list_memories",
			Description: "List stored memories for an entity, optionally filtered by type or tag. Returns memories ordered by creation time (newest first).",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"entity_id": map[string]any{
						"type":        "string",
						"description": "External identifier for the entity.",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "Maximum number of memories to return. Defaults to 50.",
					},
					"type": map[string]any{
						"type":        "string",
						"enum":        []string{"identity", "event", "pattern"},
						"description": "Filter by fact type.",
					},
					"tag": map[string]any{
						"type":        "string",
						"description": "Filter by tag.",
					},
				},
				"required": []string{"entity_id"},
			},
		},
		{
			Name:        "delete_memory",
			Description: "Delete a specific memory by its ID.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"entity_id": map[string]any{
						"type":        "string",
						"description": "External identifier for the entity.",
					},
					"memory_id": map[string]any{
						"type":        "string",
						"description": "UUID of the memory to delete.",
					},
				},
				"required": []string{"entity_id", "memory_id"},
			},
		},
		{
			Name:        "delete_all_memories",
			Description: "Delete all memories for an entity. This action is irreversible.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"entity_id": map[string]any{
						"type":        "string",
						"description": "External identifier for the entity.",
					},
				},
				"required": []string{"entity_id"},
			},
		},
	}
}

// executeTool routes a tool call to its implementation.
func (s *Server) executeTool(ctx context.Context, name string, args json.RawMessage) (any, error) {
	switch name {
	case "add_memories":
		return s.toolAddMemories(ctx, args)
	case "search_memories":
		return s.toolSearchMemories(ctx, args)
	case "list_memories":
		return s.toolListMemories(ctx, args)
	case "delete_memory":
		return s.toolDeleteMemory(ctx, args)
	case "delete_all_memories":
		return s.toolDeleteAllMemories(ctx, args)
	default:
		return nil, fmt.Errorf("unknown tool: %s", name)
	}
}

// --- add_memories ---

type addMemoriesArgs struct {
	EntityID string          `json:"entity_id"`
	Memories []memoryPayload `json:"memories"`
}

type memoryPayload struct {
	Content      string `json:"content"`
	Type         string `json:"type"`
	Significance string `json:"significance"`
	Tag          string `json:"tag"`
}

func (s *Server) toolAddMemories(ctx context.Context, args json.RawMessage) (any, error) {
	var p addMemoriesArgs
	if err := json.Unmarshal(args, &p); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	if p.EntityID == "" {
		return nil, fmt.Errorf("entity_id is required")
	}
	if len(p.Memories) == 0 {
		return nil, fmt.Errorf("at least one memory is required")
	}

	entityUUID, err := s.entities.getOrCreateUUID(ctx, p.EntityID)
	if err != nil {
		return nil, fmt.Errorf("entity resolution failed: %w", err)
	}

	// Collect contents for batch embedding.
	contents := make([]string, len(p.Memories))
	for i, m := range p.Memories {
		if m.Content == "" {
			return nil, fmt.Errorf("memory at index %d has empty content", i)
		}
		contents[i] = m.Content
	}

	// Embed all contents in one batch.
	var embeddings [][]float32
	if s.cfg.Embedder != nil {
		var err error
		embeddings, err = s.cfg.Embedder.Embed(ctx, contents)
		if err != nil {
			return nil, fmt.Errorf("embedding failed: %w", err)
		}
	}

	// Build facts.
	facts := make([]*store.Fact, len(p.Memories))
	for i, m := range p.Memories {
		f := &store.Fact{
			Content:        m.Content,
			Type:           parseFactType(m.Type),
			TemporalStatus: store.TemporalCurrent,
			Significance:   parseSignificance(m.Significance),
			ContentKey:     store.DefaultContentKey(m.Content),
			Tag:            m.Tag,
		}
		if embeddings != nil && i < len(embeddings) {
			f.Embedding = embeddings[i]
		}

		// Set TTL based on significance.
		f.ExpiresAt = store.TTLForSignificance(f.Significance)

		facts[i] = f
	}

	// Dedup: check each fact against existing content keys.
	var inserted int
	for _, f := range facts {
		existing, err := s.cfg.Repo.FindFactByKey(ctx, entityUUID, f.ContentKey)
		if err != nil {
			return nil, fmt.Errorf("dedup check failed: %w", err)
		}
		if existing != nil {
			// Reinforce the existing fact.
			if err := s.cfg.Repo.ReinforceFact(ctx, existing.UUID, f.ExpiresAt); err != nil {
				return nil, fmt.Errorf("reinforce failed: %w", err)
			}
			continue
		}
		if err := s.cfg.Repo.InsertFact(ctx, entityUUID, f); err != nil {
			return nil, fmt.Errorf("insert failed: %w", err)
		}
		inserted++
	}

	return map[string]any{
		"inserted":   inserted,
		"reinforced": len(facts) - inserted,
	}, nil
}

// --- search_memories ---

type searchMemoriesArgs struct {
	EntityID string `json:"entity_id"`
	Query    string `json:"query"`
	Limit    int    `json:"limit"`
}

func (s *Server) toolSearchMemories(ctx context.Context, args json.RawMessage) (any, error) {
	var p searchMemoriesArgs
	if err := json.Unmarshal(args, &p); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	if p.EntityID == "" {
		return nil, fmt.Errorf("entity_id is required")
	}
	if p.Query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if p.Limit <= 0 {
		p.Limit = 10
	}

	if s.cfg.Embedder == nil {
		return nil, fmt.Errorf("embedder not configured — search requires an embedding provider")
	}

	entityUUID, err := s.entities.getOrCreateUUID(ctx, p.EntityID)
	if err != nil {
		return nil, fmt.Errorf("entity resolution failed: %w", err)
	}

	results, err := memory.Recall(ctx, s.cfg.Embedder, s.engine, s.cfg.Repo, p.Query, entityUUID, p.Limit, 0.1, 10000)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	memories := make([]map[string]any, len(results))
	for i, r := range results {
		memories[i] = map[string]any{
			"id":              r.ID,
			"content":         r.Content,
			"score":           r.Score,
			"temporal_status": r.TemporalStatus,
			"significance":    r.Significance,
			"created_at":      r.CreatedAt.Format(time.RFC3339),
		}
	}

	return map[string]any{
		"memories": memories,
		"count":    len(memories),
	}, nil
}

// --- list_memories ---

type listMemoriesArgs struct {
	EntityID string `json:"entity_id"`
	Limit    int    `json:"limit"`
	Type     string `json:"type"`
	Tag      string `json:"tag"`
}

func (s *Server) toolListMemories(ctx context.Context, args json.RawMessage) (any, error) {
	var p listMemoriesArgs
	if err := json.Unmarshal(args, &p); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	if p.EntityID == "" {
		return nil, fmt.Errorf("entity_id is required")
	}
	if p.Limit <= 0 {
		p.Limit = 50
	}

	entityUUID, err := s.entities.getOrCreateUUID(ctx, p.EntityID)
	if err != nil {
		return nil, fmt.Errorf("entity resolution failed: %w", err)
	}

	filter := store.FactFilter{
		ExcludeExpired: true,
	}
	if p.Type != "" {
		filter.Types = []store.FactType{store.FactType(p.Type)}
	}
	if p.Tag != "" {
		filter.Tags = []string{p.Tag}
	}

	facts, err := s.cfg.Repo.ListFactsFiltered(ctx, entityUUID, filter, p.Limit)
	if err != nil {
		return nil, fmt.Errorf("list failed: %w", err)
	}

	memories := make([]map[string]any, len(facts))
	for i, f := range facts {
		m := map[string]any{
			"id":              f.UUID,
			"content":         f.Content,
			"type":            string(f.Type),
			"temporal_status": string(f.TemporalStatus),
			"significance":    significanceLabel(f.Significance),
			"created_at":      f.CreatedAt.Format(time.RFC3339),
		}
		if f.Tag != "" {
			m["tag"] = f.Tag
		}
		if f.ReinforcedCount > 0 {
			m["reinforced_count"] = f.ReinforcedCount
		}
		memories[i] = m
	}

	return map[string]any{
		"memories": memories,
		"count":    len(memories),
	}, nil
}

// --- delete_memory ---

type deleteMemoryArgs struct {
	EntityID string `json:"entity_id"`
	MemoryID string `json:"memory_id"`
}

func (s *Server) toolDeleteMemory(ctx context.Context, args json.RawMessage) (any, error) {
	var p deleteMemoryArgs
	if err := json.Unmarshal(args, &p); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	if p.EntityID == "" {
		return nil, fmt.Errorf("entity_id is required")
	}
	if p.MemoryID == "" {
		return nil, fmt.Errorf("memory_id is required")
	}

	entityUUID, err := s.entities.getOrCreateUUID(ctx, p.EntityID)
	if err != nil {
		return nil, fmt.Errorf("entity resolution failed: %w", err)
	}

	if err := s.cfg.Repo.DeleteFact(ctx, entityUUID, p.MemoryID); err != nil {
		return nil, fmt.Errorf("delete failed: %w", err)
	}

	return map[string]any{
		"deleted": true,
	}, nil
}

// --- delete_all_memories ---

type deleteAllMemoriesArgs struct {
	EntityID string `json:"entity_id"`
}

func (s *Server) toolDeleteAllMemories(ctx context.Context, args json.RawMessage) (any, error) {
	var p deleteAllMemoriesArgs
	if err := json.Unmarshal(args, &p); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	if p.EntityID == "" {
		return nil, fmt.Errorf("entity_id is required")
	}

	entityUUID, err := s.entities.getOrCreateUUID(ctx, p.EntityID)
	if err != nil {
		return nil, fmt.Errorf("entity resolution failed: %w", err)
	}

	count, err := s.cfg.Repo.DeleteEntityFacts(ctx, entityUUID)
	if err != nil {
		return nil, fmt.Errorf("delete all failed: %w", err)
	}

	return map[string]any{
		"deleted": count,
	}, nil
}

// --- helpers ---

func parseFactType(s string) store.FactType {
	switch s {
	case "event":
		return store.FactTypeEvent
	case "pattern":
		return store.FactTypePattern
	default:
		return store.FactTypeIdentity
	}
}

func parseSignificance(s string) store.Significance {
	switch s {
	case "low":
		return store.SignificanceLow
	case "high":
		return store.SignificanceHigh
	default:
		return store.SignificanceMedium
	}
}

func significanceLabel(s store.Significance) string {
	switch s {
	case store.SignificanceLow:
		return "low"
	case store.SignificanceHigh:
		return "high"
	default:
		return "medium"
	}
}

