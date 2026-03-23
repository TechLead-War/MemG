// Package onnx provides an in-process ONNX Runtime embedding provider.
// This file implements a pure Go BERT WordPiece tokenizer compatible with
// the all-MiniLM-L6-v2 model and other uncased BERT-family models.
package onnx

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const (
	clsToken   = "[CLS]"
	sepToken   = "[SEP]"
	padToken   = "[PAD]"
	unkToken   = "[UNK]"
	wpPrefix   = "##"
	maxWordLen = 200
)

// BertTokenizer performs BERT-style WordPiece tokenization.
type BertTokenizer struct {
	vocab     map[string]int32
	maxSeqLen int
	clsID     int32
	sepID     int32
	padID     int32
	unkID     int32
}

// TokenizerOutput holds the encoded result ready for ONNX input tensors.
type TokenizerOutput struct {
	InputIDs      []int64
	AttentionMask []int64
	TokenTypeIDs  []int64
}

// NewBertTokenizer loads a BERT vocabulary from vocab.txt and returns a tokenizer.
// maxSeqLen controls padding/truncation length (256 for all-MiniLM-L6-v2).
func NewBertTokenizer(vocabPath string, maxSeqLen int) (*BertTokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("onnx: open vocab: %w", err)
	}
	defer f.Close()

	vocab := make(map[string]int32, 32000)
	scanner := bufio.NewScanner(f)
	var id int32
	for scanner.Scan() {
		token := scanner.Text()
		vocab[token] = id
		id++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("onnx: read vocab: %w", err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("onnx: vocab is empty")
	}

	lookup := func(tok string) int32 {
		if v, ok := vocab[tok]; ok {
			return v
		}
		return 0
	}

	return &BertTokenizer{
		vocab:     vocab,
		maxSeqLen: maxSeqLen,
		clsID:     lookup(clsToken),
		sepID:     lookup(sepToken),
		padID:     lookup(padToken),
		unkID:     lookup(unkToken),
	}, nil
}

// Encode tokenizes a single text into padded/truncated ONNX-ready tensors.
func (t *BertTokenizer) Encode(text string) TokenizerOutput {
	tokens := t.tokenize(text)

	// Truncate to maxSeqLen - 2 to leave room for [CLS] and [SEP].
	maxTokens := t.maxSeqLen - 2
	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}

	// Build token IDs: [CLS] + tokens + [SEP] + [PAD]...
	ids := make([]int64, t.maxSeqLen)
	mask := make([]int64, t.maxSeqLen)
	typeIDs := make([]int64, t.maxSeqLen) // all zeros for single-sentence

	ids[0] = int64(t.clsID)
	mask[0] = 1

	for i, tok := range tokens {
		pos := i + 1
		if v, ok := t.vocab[tok]; ok {
			ids[pos] = int64(v)
		} else {
			ids[pos] = int64(t.unkID)
		}
		mask[pos] = 1
	}

	sepPos := len(tokens) + 1
	ids[sepPos] = int64(t.sepID)
	mask[sepPos] = 1

	// Remaining positions stay 0 (padID=0, mask=0, typeIDs=0).

	return TokenizerOutput{
		InputIDs:      ids,
		AttentionMask: mask,
		TokenTypeIDs:  typeIDs,
	}
}

// EncodeBatch tokenizes multiple texts, returning flat slices for batched
// ONNX input tensors of shape [batch_size, max_seq_len].
func (t *BertTokenizer) EncodeBatch(texts []string) (inputIDs, attentionMask, tokenTypeIDs []int64) {
	total := len(texts) * t.maxSeqLen
	inputIDs = make([]int64, 0, total)
	attentionMask = make([]int64, 0, total)
	tokenTypeIDs = make([]int64, 0, total)

	for _, text := range texts {
		enc := t.Encode(text)
		inputIDs = append(inputIDs, enc.InputIDs...)
		attentionMask = append(attentionMask, enc.AttentionMask...)
		tokenTypeIDs = append(tokenTypeIDs, enc.TokenTypeIDs...)
	}
	return
}

// tokenize runs the full BERT tokenization pipeline: normalize → basic
// tokenize → WordPiece.
func (t *BertTokenizer) tokenize(text string) []string {
	text = t.normalize(text)
	words := t.basicTokenize(text)

	var tokens []string
	for _, word := range words {
		wp := t.wordPieceTokenize(word)
		tokens = append(tokens, wp...)
	}
	return tokens
}

// normalize lowercases and strips accents (NFD decomposition, remove Mn).
func (t *BertTokenizer) normalize(text string) string {
	text = strings.ToLower(text)
	// NFD decompose then remove combining marks (Mn category).
	decomposed := norm.NFD.String(text)
	var b strings.Builder
	b.Grow(len(decomposed))
	for _, r := range decomposed {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// basicTokenize splits on whitespace and punctuation, inserting spaces
// around punctuation characters.
func (t *BertTokenizer) basicTokenize(text string) []string {
	var b strings.Builder
	b.Grow(len(text) + 64)

	for _, r := range text {
		if isWhitespace(r) {
			b.WriteRune(' ')
		} else if isPunctuation(r) {
			b.WriteRune(' ')
			b.WriteRune(r)
			b.WriteRune(' ')
		} else if isControl(r) {
			continue
		} else {
			b.WriteRune(r)
		}
	}

	return strings.Fields(b.String())
}

// wordPieceTokenize greedily matches the longest subword prefix in the
// vocabulary. Unknown subwords produce [UNK].
func (t *BertTokenizer) wordPieceTokenize(word string) []string {
	if len(word) > maxWordLen {
		return []string{unkToken}
	}

	var tokens []string
	runes := []rune(word)
	start := 0

	for start < len(runes) {
		end := len(runes)
		found := false

		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = wpPrefix + substr
			}
			if _, ok := t.vocab[substr]; ok {
				tokens = append(tokens, substr)
				found = true
				break
			}
			end--
		}

		if !found {
			tokens = append(tokens, unkToken)
			start++
		} else {
			start = end
		}
	}
	return tokens
}

func isWhitespace(r rune) bool {
	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

func isPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.Is(unicode.P, r)
}

func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.Is(unicode.Cc, r)
}
