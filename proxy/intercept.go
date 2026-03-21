package proxy

import (
	"fmt"
	"io"
	"net/http"

	"memg/llm"
)

// handleNonStreaming forwards the modified request body to the upstream LLM,
// copies the full response to the client, and triggers background memory
// processing on success.
func (s *Server) handleNonStreaming(
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

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "memg proxy: failed to read upstream response", http.StatusBadGateway)
		return
	}

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)

	if resp.StatusCode == http.StatusOK {
		content, err := format.ExtractResponseContent(respBody)
		if err != nil {
			fmt.Printf("memg proxy: extract response content failed (%s): %v\n", format.Name(), err)
			return
		}
		if content != "" {
			go s.afterResponse(entityUUID, sessionUUID, messages, content)
		}
	}
}
