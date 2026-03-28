package memory

import (
	"encoding/json"
	"regexp"
	"strings"

	"memg/llm"
)

// DetectedArtifact represents a code/structured output found in messages.
type DetectedArtifact struct {
	Content      string
	ArtifactType string // "code", "json", "sql", "markdown"
	Language     string // "go", "python", "typescript", etc.
	SourceRole   string // "user" or "assistant"
}

var (
	codeFenceRe = regexp.MustCompile("(?s)```(\\w*)\\n([\\s\\S]*?)```")
	sqlPrefixRe = regexp.MustCompile(`(?i)^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b`)
	jsonBlockRe = regexp.MustCompile(`(?s)\{[\s\S]*?\}`)
)

// DetectArtifacts scans messages for code blocks, JSON objects, SQL statements.
func DetectArtifacts(messages []*llm.Message) []DetectedArtifact {
	seen := make(map[string]struct{})
	var result []DetectedArtifact

	add := func(a DetectedArtifact) {
		if _, dup := seen[a.Content]; dup {
			return
		}
		seen[a.Content] = struct{}{}
		result = append(result, a)
	}

	for _, msg := range messages {
		if msg == nil {
			continue
		}
		role := msg.Role
		if role != llm.RoleUser && role != llm.RoleAssistant {
			continue
		}
		content := msg.Content

		// 1. Code fences.
		for _, match := range codeFenceRe.FindAllStringSubmatch(content, -1) {
			lang := strings.TrimSpace(match[1])
			body := strings.TrimSpace(match[2])
			if body == "" {
				continue
			}
			add(DetectedArtifact{
				Content:      body,
				ArtifactType: "code",
				Language:     lang,
				SourceRole:   role,
			})
		}

		// 2. Inline code blocks: consecutive lines starting with 4+ spaces or tab, > 3 lines.
		stripped := codeFenceRe.ReplaceAllString(content, "")
		lines := strings.Split(stripped, "\n")
		var block []string
		for _, line := range lines {
			if len(line) > 0 && (strings.HasPrefix(line, "    ") || line[0] == '\t') {
				block = append(block, line)
			} else {
				if len(block) > 3 {
					body := strings.TrimSpace(strings.Join(block, "\n"))
					if body != "" {
						add(DetectedArtifact{
							Content:      body,
							ArtifactType: "code",
							SourceRole:   role,
						})
					}
				}
				block = block[:0]
			}
		}
		if len(block) > 3 {
			body := strings.TrimSpace(strings.Join(block, "\n"))
			if body != "" {
				add(DetectedArtifact{
					Content:      body,
					ArtifactType: "code",
					SourceRole:   role,
				})
			}
		}

		// 3. JSON objects.
		for _, match := range jsonBlockRe.FindAllString(stripped, -1) {
			trimmed := strings.TrimSpace(match)
			var js json.RawMessage
			if json.Unmarshal([]byte(trimmed), &js) == nil {
				add(DetectedArtifact{
					Content:      trimmed,
					ArtifactType: "json",
					SourceRole:   role,
				})
			}
		}

		// 4. SQL statements.
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if sqlPrefixRe.MatchString(trimmed) {
				add(DetectedArtifact{
					Content:      trimmed,
					ArtifactType: "sql",
					SourceRole:   role,
				})
			}
		}
	}

	return result
}
