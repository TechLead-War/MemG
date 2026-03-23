package onnx

import "math"

// meanPool computes the mean of token embeddings weighted by the attention mask.
// tokenEmbeddings is a flat slice of shape [seqLen * dim], attentionMask is [seqLen].
// Returns a single vector of length dim.
func meanPool(tokenEmbeddings []float32, attentionMask []int64, seqLen, dim int) []float32 {
	result := make([]float32, dim)
	var maskSum float32

	for i := 0; i < seqLen; i++ {
		if attentionMask[i] == 0 {
			continue
		}
		maskSum++
		offset := i * dim
		for j := 0; j < dim; j++ {
			result[j] += tokenEmbeddings[offset+j]
		}
	}

	if maskSum > 0 {
		for j := range result {
			result[j] /= maskSum
		}
	}

	return result
}

// l2Normalize normalizes a vector to unit length (L2 norm).
func l2Normalize(vec []float32) []float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sum))
	if norm == 0 {
		return vec
	}
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = v / norm
	}
	return out
}
