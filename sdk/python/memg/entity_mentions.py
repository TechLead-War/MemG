"""Entity mention extraction: pull proper nouns and specific concepts from facts."""

from __future__ import annotations

import unicodedata
from typing import List

_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "with", "this",
    "that", "from", "they", "will", "would", "there", "their", "what", "about",
    "which", "when", "make", "like", "just", "over", "such", "take", "than",
    "them", "very", "some", "could", "into", "also", "then", "does", "more",
    "other", "user", "said", "each", "tell", "should", "because",
}


def _tokenize(s: str) -> List[str]:
    """Split on whitespace and punctuation, keeping @, ., / within tokens."""
    tokens = []
    current: List[str] = []
    for ch in s:
        if ch in ("@", ".", "/"):
            current.append(ch)
        elif unicodedata.category(ch).startswith("L") or unicodedata.category(ch).startswith("N"):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _is_candidate(tok: str) -> bool:
    """Check if a token is a proper noun or specific concept."""
    if not tok:
        return False
    if tok[0].isupper():
        return True
    for ch in tok:
        if ch.isdigit() or ch in ("@", ".", "/"):
            return True
    return False


def extract_entity_mentions(facts: list, max_mentions: int = 15) -> List[str]:
    """Extract proper nouns and specific concepts from fact contents.

    facts: list of objects with .content attribute.
    Returns up to max_mentions unique mentions.
    """
    if not facts or max_mentions <= 0:
        return []

    seen: set = set()
    mentions: List[str] = []

    for i in range(len(facts) - 1, -1, -1):
        tokens = _tokenize(facts[i].content)
        for tok in tokens:
            if len(tok) < 3:
                continue
            lower = tok.lower()
            if lower in _STOP_WORDS:
                continue
            if not _is_candidate(tok):
                continue
            if lower in seen:
                continue
            seen.add(lower)
            mentions.append(tok)
            if len(mentions) >= max_mentions:
                return mentions

    return mentions
