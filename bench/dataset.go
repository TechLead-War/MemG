package bench

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Dataset holds all parsed LoCoMo conversations and QA pairs.
type Dataset struct {
	Entries []Entry
}

// Entry is one LoCoMo conversation with its QA annotations.
type Entry struct {
	SampleID     string
	Conversation Conversation
	QA           []QA
}

// Conversation holds the multi-session dialogue between two speakers.
type Conversation struct {
	SpeakerA string
	SpeakerB string
	Sessions []Session
}

// Session is a single conversation session with a date and ordered turns.
type Session struct {
	Number   int
	DateTime string
	Turns    []Turn
}

// Turn is a single utterance in a session.
type Turn struct {
	Speaker     string
	DiaID       string
	Text        string
	BlipCaption string // image caption, if an image was shared
}

// flexString unmarshals from both JSON strings and numbers.
type flexString string

func (f *flexString) UnmarshalJSON(data []byte) error {
	// Try string first.
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*f = flexString(s)
		return nil
	}
	// Try number.
	var n json.Number
	if err := json.Unmarshal(data, &n); err == nil {
		*f = flexString(n.String())
		return nil
	}
	// Fallback: treat raw bytes as string.
	*f = flexString(strings.Trim(string(data), `"`))
	return nil
}

// QA is a single question-answer pair with category and evidence pointers.
type QA struct {
	Question          string     `json:"question"`
	Answer            flexString `json:"answer"`
	AdversarialAnswer flexString `json:"adversarial_answer"`
	Category          int        `json:"category"`
	Evidence          []string   `json:"evidence"`
}

// GoldAnswer returns the gold-standard answer string for scoring.
func (q QA) GoldAnswer() string {
	if q.Category == 5 {
		return string(q.AdversarialAnswer)
	}
	return string(q.Answer)
}

// CategoryName returns a human-readable label for a LoCoMo category number.
func CategoryName(cat int) string {
	switch cat {
	case 1:
		return "multi-hop"
	case 2:
		return "temporal"
	case 3:
		return "open-domain"
	case 4:
		return "single-hop"
	case 5:
		return "adversarial"
	default:
		return fmt.Sprintf("unknown-%d", cat)
	}
}

// LoadDataset reads and parses a locomo10.json file.
func LoadDataset(path string) (*Dataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}

	var rawEntries []json.RawMessage
	if err := json.Unmarshal(data, &rawEntries); err != nil {
		return nil, fmt.Errorf("parse dataset array: %w", err)
	}

	entries := make([]Entry, 0, len(rawEntries))
	for _, raw := range rawEntries {
		entry, err := parseEntry(raw)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}

	return &Dataset{Entries: entries}, nil
}

func parseEntry(data json.RawMessage) (Entry, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return Entry{}, fmt.Errorf("parse entry: %w", err)
	}

	var entry Entry
	if err := json.Unmarshal(obj["sample_id"], &entry.SampleID); err != nil {
		return Entry{}, fmt.Errorf("parse sample_id: %w", err)
	}

	conv, err := parseConversation(obj["conversation"])
	if err != nil {
		return Entry{}, fmt.Errorf("parse conversation %s: %w", entry.SampleID, err)
	}
	entry.Conversation = conv

	if err := json.Unmarshal(obj["qa"], &entry.QA); err != nil {
		return Entry{}, fmt.Errorf("parse qa %s: %w", entry.SampleID, err)
	}

	return entry, nil
}

func parseConversation(data json.RawMessage) (Conversation, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return Conversation{}, fmt.Errorf("parse conversation object: %w", err)
	}

	var conv Conversation
	if err := json.Unmarshal(obj["speaker_a"], &conv.SpeakerA); err != nil {
		return Conversation{}, fmt.Errorf("parse speaker_a: %w", err)
	}
	if err := json.Unmarshal(obj["speaker_b"], &conv.SpeakerB); err != nil {
		return Conversation{}, fmt.Errorf("parse speaker_b: %w", err)
	}

	// Discover session numbers from keys like "session_1", "session_2", etc.
	sessionNums := make(map[int]struct{})
	for key := range obj {
		if !strings.HasPrefix(key, "session_") || strings.HasSuffix(key, "_date_time") {
			continue
		}
		numStr := strings.TrimPrefix(key, "session_")
		if n, err := strconv.Atoi(numStr); err == nil {
			sessionNums[n] = struct{}{}
		}
	}

	nums := make([]int, 0, len(sessionNums))
	for n := range sessionNums {
		nums = append(nums, n)
	}
	sort.Ints(nums)

	for _, n := range nums {
		sessionKey := fmt.Sprintf("session_%d", n)
		dateKey := fmt.Sprintf("session_%d_date_time", n)

		var dateTime string
		if dt, ok := obj[dateKey]; ok {
			_ = json.Unmarshal(dt, &dateTime)
		}

		var rawTurns []struct {
			Speaker     string   `json:"speaker"`
			DiaID       string   `json:"dia_id"`
			Text        string   `json:"text"`
			BlipCaption string   `json:"blip_caption"`
			ImgURL      []string `json:"img_url"`
		}
		if err := json.Unmarshal(obj[sessionKey], &rawTurns); err != nil {
			return Conversation{}, fmt.Errorf("parse session_%d: %w", n, err)
		}

		session := Session{Number: n, DateTime: dateTime}
		for _, rt := range rawTurns {
			session.Turns = append(session.Turns, Turn{
				Speaker:     rt.Speaker,
				DiaID:       rt.DiaID,
				Text:        rt.Text,
				BlipCaption: rt.BlipCaption,
			})
		}
		conv.Sessions = append(conv.Sessions, session)
	}

	return conv, nil
}

// TurnText returns the full text of a turn, including image captions.
func (t Turn) TurnText() string {
	if t.BlipCaption != "" {
		return t.Text + " [Shared image: " + t.BlipCaption + "]"
	}
	return t.Text
}
