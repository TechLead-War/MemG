package proxy

import (
	"bufio"
	"fmt"
	"net/http"
	"strings"

	"memg/llm"
)

// handleStreaming forwards the modified request body to the upstream LLM and
// proxies the SSE stream to the client while accumulating the response content
// for background memory processing.
func (s *Server) handleStreaming(
	w http.ResponseWriter,
	r *http.Request,
	body []byte,
	format Format,
	entityUUID, sessionUUID string,
	messages []*llm.Message,
) {
	resp, err := s.forwardToUpstream(r, body)
	if err != nil {
		http.Error(w, "memg proxy: upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Copy any other upstream headers, except those we've already set.
	for key, values := range resp.Header {
		lk := strings.ToLower(key)
		if lk == "content-type" || lk == "cache-control" || lk == "connection" {
			continue
		}
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	flusher, canFlush := w.(http.Flusher)

	var accumulated strings.Builder
	var currentEventType string

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		fmt.Fprint(w, line+"\n")
		if canFlush {
			flusher.Flush()
		}

		if strings.HasPrefix(line, "event: ") {
			currentEventType = strings.TrimPrefix(line, "event: ")
			currentEventType = strings.TrimSpace(currentEventType)
		} else if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			delta, done := format.AccumulateStreamData(currentEventType, data)
			if delta != "" {
				accumulated.WriteString(delta)
			}
			if done {
				// Stream is complete; we could break, but we continue
				// forwarding any remaining lines from the upstream.
			}
		} else if line == "" {
			// Empty line is an SSE event boundary — reset event type.
			currentEventType = ""
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("memg proxy: stream scan error: %v\n", err)
	}

	if resp.StatusCode == http.StatusOK {
		content := accumulated.String()
		if content != "" {
			go s.afterResponse(entityUUID, sessionUUID, messages, content)
		}
	}
}
