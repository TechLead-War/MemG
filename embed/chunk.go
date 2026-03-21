package embed

import "strings"

// ChunkText splits a long text into overlapping segments that fit within a
// token budget. It splits on word boundaries.
func ChunkText(text string, maxTokens, overlap int) []string {
	words := strings.Fields(text)
	if len(words) <= maxTokens {
		return []string{text}
	}

	var chunks []string
	start := 0
	for start < len(words) {
		end := start + maxTokens
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)
		step := maxTokens - overlap
		if step < 1 {
			step = 1
		}
		start += step
	}
	return chunks
}
