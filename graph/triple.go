// Package graph provides a lightweight knowledge graph based on
// subject-predicate-object triples extracted from conversations.
package graph

// Triple is a single directed relationship in the knowledge graph.
type Triple struct {
	Subject   string
	Predicate string
	Object    string
}
