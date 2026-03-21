package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// JSON-RPC 2.0 message types.

type jsonrpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Standard JSON-RPC error codes.
const (
	errParse          = -32700
	errInvalidRequest = -32600
	errMethodNotFound = -32601
	errInvalidParams  = -32602
	errInternal       = -32603
)

// MCP protocol constants.
const (
	protocolVersion = "2025-03-26"
	serverName      = "memg"
	serverVersion   = "1.0.0"
)

// dispatch routes a JSON-RPC message to the appropriate handler.
func (s *Server) dispatch(ctx context.Context, msg *jsonrpcMessage) (any, *jsonrpcError) {
	switch msg.Method {
	case "initialize":
		return s.handleInitialize(msg.Params)
	case "ping":
		return map[string]any{}, nil
	case "tools/list":
		return s.handleToolsList()
	case "tools/call":
		return s.handleToolsCall(ctx, msg.Params)
	default:
		return nil, &jsonrpcError{Code: errMethodNotFound, Message: fmt.Sprintf("unknown method: %s", msg.Method)}
	}
}

// handleNotification processes JSON-RPC notifications (messages without an ID).
func (s *Server) handleNotification(msg jsonrpcMessage) {
	switch msg.Method {
	case "notifications/initialized":
		if s.cfg.Debug {
			fmt.Println("memg mcp: client initialized")
		}
	case "notifications/cancelled":
		// Client cancelled a request — nothing to do for now.
	}
}

// --- initialize ---

type initializeParams struct {
	ProtocolVersion string `json:"protocolVersion"`
	ClientInfo      struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"clientInfo"`
}

func (s *Server) handleInitialize(params json.RawMessage) (any, *jsonrpcError) {
	var p initializeParams
	if params != nil {
		if err := json.Unmarshal(params, &p); err != nil {
			return nil, &jsonrpcError{Code: errInvalidParams, Message: "invalid initialize params"}
		}
	}

	if s.cfg.Debug {
		fmt.Printf("memg mcp: client connected: %s %s (protocol %s)\n",
			p.ClientInfo.Name, p.ClientInfo.Version, p.ProtocolVersion)
	}

	return map[string]any{
		"protocolVersion": protocolVersion,
		"capabilities": map[string]any{
			"tools": map[string]any{},
		},
		"serverInfo": map[string]any{
			"name":    serverName,
			"version": serverVersion,
		},
	}, nil
}

// --- tools/list ---

type toolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	InputSchema any    `json:"inputSchema"`
}

func (s *Server) handleToolsList() (any, *jsonrpcError) {
	return map[string]any{
		"tools": toolDefinitions(),
	}, nil
}

// --- tools/call ---

type toolCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type toolResult struct {
	Content []toolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

type toolContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (s *Server) handleToolsCall(ctx context.Context, params json.RawMessage) (any, *jsonrpcError) {
	var p toolCallParams
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &jsonrpcError{Code: errInvalidParams, Message: "invalid tools/call params"}
	}

	result, err := s.executeTool(ctx, p.Name, p.Arguments)
	if err != nil {
		return &toolResult{
			Content: []toolContent{{Type: "text", Text: err.Error()}},
			IsError: true,
		}, nil
	}

	text, _ := json.Marshal(result)
	return &toolResult{
		Content: []toolContent{{Type: "text", Text: string(text)}},
	}, nil
}

// --- JSON-RPC response helpers ---

func writeJSONRPCResult(w http.ResponseWriter, id json.RawMessage, result any) {
	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func writeJSONRPCError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &jsonrpcError{Code: code, Message: message},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
