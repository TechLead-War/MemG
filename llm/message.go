package llm

import (
	"strings"
	"sync"
)

// Standard role constants.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message represents one turn in a conversation.
type Message struct {
	Role    string
	Content string
}

// UserMessage creates a user-role message.
func UserMessage(content string) *Message {
	return &Message{Role: RoleUser, Content: content}
}

// SystemMessage creates a system-role message.
func SystemMessage(content string) *Message {
	return &Message{Role: RoleSystem, Content: content}
}

// AssistantMessage creates an assistant-role message.
func AssistantMessage(content string) *Message {
	return &Message{Role: RoleAssistant, Content: content}
}

// Request bundles everything needed for a single LLM call.
type Request struct {
	Model       string
	System      string
	Messages    []*Message
	MaxTokens   int
	Temperature *float64
}

// NewRequest builds a Request from messages and optional call-level overrides.
func NewRequest(messages []*Message, opts ...CallOption) *Request {
	r := &Request{Messages: messages}
	for _, o := range opts {
		o(r)
	}
	return r
}

// PrependSystem adds context text before the existing system prompt.
func (r *Request) PrependSystem(text string) {
	if text == "" {
		return
	}
	if r.System == "" {
		r.System = text
	} else {
		r.System = text + "\n\n" + r.System
	}
}

// PrependMessages inserts messages at the beginning of the message list.
func (r *Request) PrependMessages(msgs []*Message) {
	r.Messages = append(msgs, r.Messages...)
}

// InjectHistory inserts conversation history after any leading system messages.
func (r *Request) InjectHistory(msgs []*Message) {
	if len(msgs) == 0 {
		return
	}
	insertAt := 0
	for insertAt < len(r.Messages) && r.Messages[insertAt] != nil && r.Messages[insertAt].Role == RoleSystem {
		insertAt++
	}
	updated := make([]*Message, 0, len(r.Messages)+len(msgs))
	updated = append(updated, r.Messages[:insertAt]...)
	updated = append(updated, msgs...)
	updated = append(updated, r.Messages[insertAt:]...)
	r.Messages = updated
}

// Response holds the result of a non-streaming LLM call.
type Response struct {
	Content      string
	Role         string
	FinishReason string
	Usage        Usage
}

// ToMessages converts the response into a message slice for storage.
func (r *Response) ToMessages() []*Message {
	return []*Message{{Role: r.Role, Content: r.Content}}
}

// Usage tracks token consumption.
type Usage struct {
	PromptTokens int
	OutputTokens int
	TotalTokens  int
}

// StreamEvent carries one chunk of a streaming response.
type StreamEvent struct {
	Delta  string
	Done   bool
	Err    error
	Finish string
}

// StreamReader delivers an LLM response incrementally.
type StreamReader struct {
	mu        sync.Mutex
	ch        <-chan StreamEvent
	buf       strings.Builder
	lastDelta string
	lastDone  bool
	done      bool
	err       error
	finish    string
}

// NewStreamReader wraps a channel of events into a StreamReader.
func NewStreamReader(ch <-chan StreamEvent) *StreamReader {
	return &StreamReader{ch: ch}
}

// Next advances the reader. Returns false when the stream is exhausted.
func (s *StreamReader) Next() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return false
	}
	ev, ok := <-s.ch
	if !ok {
		s.done = true
		return false
	}
	if ev.Err != nil {
		s.err = ev.Err
		s.done = true
		return false
	}
	s.lastDelta = ev.Delta
	s.lastDone = ev.Done
	s.buf.WriteString(ev.Delta)
	if ev.Done {
		s.finish = ev.Finish
		s.done = true
	}
	return true
}

// Text returns the content accumulated so far.
func (s *StreamReader) Text() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

// Err returns the first error encountered during streaming.
func (s *StreamReader) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// FinishReason returns the model's finish reason (available after the stream ends).
func (s *StreamReader) FinishReason() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.finish
}

// Delta returns the delta from the most recent successful Next call.
func (s *StreamReader) Delta() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastDelta
}

// Done returns whether the most recent successful Next call delivered the final event.
func (s *StreamReader) Done() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastDone
}
