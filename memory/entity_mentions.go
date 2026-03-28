package memory

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"memg/store"
)

var stopWords = map[string]struct{}{
	"the": {}, "and": {}, "for": {}, "are": {}, "but": {}, "not": {},
	"you": {}, "all": {}, "can": {}, "had": {}, "her": {}, "was": {},
	"one": {}, "our": {}, "out": {}, "has": {}, "have": {}, "been": {},
	"with": {}, "this": {}, "that": {}, "from": {}, "they": {}, "will": {},
	"would": {}, "there": {}, "their": {}, "what": {}, "about": {}, "which": {},
	"when": {}, "make": {}, "like": {}, "just": {}, "over": {}, "such": {},
	"take": {}, "than": {}, "them": {}, "very": {}, "some": {}, "could": {},
	"into": {}, "also": {}, "then": {}, "does": {}, "more": {}, "other": {},
	"user": {}, "said": {}, "each": {}, "tell": {}, "should": {}, "because": {},
}

// ExtractEntityMentions pulls proper nouns and specific concepts from
// extracted fact contents. Returns up to maxMentions unique mentions.
func ExtractEntityMentions(facts []*store.Fact, maxMentions int) []string {
	if len(facts) == 0 || maxMentions <= 0 {
		return nil
	}

	seen := make(map[string]struct{})
	var mentions []string

	for i := len(facts) - 1; i >= 0; i-- {
		tokens := tokenize(facts[i].Content)
		for _, tok := range tokens {
			if len(tok) < 3 {
				continue
			}
			lower := strings.ToLower(tok)
			if _, stop := stopWords[lower]; stop {
				continue
			}
			if !isCandidate(tok) {
				continue
			}
			if _, dup := seen[lower]; dup {
				continue
			}
			seen[lower] = struct{}{}
			mentions = append(mentions, tok)
			if len(mentions) >= maxMentions {
				return mentions
			}
		}
	}

	return mentions
}

func tokenize(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		if r == '@' || r == '.' || r == '/' {
			return false
		}
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func isCandidate(tok string) bool {
	if len(tok) == 0 {
		return false
	}
	r, _ := utf8.DecodeRuneInString(tok)
	if unicode.IsUpper(r) {
		return true
	}
	for _, r := range tok {
		if unicode.IsDigit(r) || r == '@' || r == '.' || r == '/' {
			return true
		}
	}
	return false
}
