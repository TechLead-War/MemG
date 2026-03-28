/**
 * Entity mention extraction from fact contents.
 * Port of Go memory/entity_mentions.go.
 */

const STOP_WORDS: Set<string> = new Set([
  'the', 'and', 'for', 'are', 'but', 'not',
  'you', 'all', 'can', 'had', 'her', 'was',
  'one', 'our', 'out', 'has', 'have', 'been',
  'with', 'this', 'that', 'from', 'they', 'will',
  'would', 'there', 'their', 'what', 'about', 'which',
  'when', 'make', 'like', 'just', 'over', 'such',
  'take', 'than', 'them', 'very', 'some', 'could',
  'into', 'also', 'then', 'does', 'more', 'other',
  'user', 'said', 'each', 'tell', 'should', 'because',
]);

function tokenize(s: string): string[] {
  // Split on characters that are not letters, digits, @, ., or /.
  return s.split(/[^\p{L}\p{N}@./]+/u).filter(Boolean);
}

function isCandidate(tok: string): boolean {
  if (tok.length === 0) return false;
  // Starts with uppercase.
  const first = tok.codePointAt(0);
  if (first !== undefined) {
    const ch = String.fromCodePoint(first);
    if (ch !== ch.toLowerCase()) return true;
  }
  // Contains digits or special chars.
  for (const ch of tok) {
    if (/\d/.test(ch) || ch === '@' || ch === '.' || ch === '/') return true;
  }
  return false;
}

export function extractEntityMentions(
  facts: Array<{ content: string }>,
  maxMentions: number = 15
): string[] {
  if (facts.length === 0 || maxMentions <= 0) return [];

  const seen = new Set<string>();
  const mentions: string[] = [];

  for (let i = facts.length - 1; i >= 0; i--) {
    const tokens = tokenize(facts[i].content);
    for (const tok of tokens) {
      if (tok.length < 3) continue;
      const lower = tok.toLowerCase();
      if (STOP_WORDS.has(lower)) continue;
      if (!isCandidate(tok)) continue;
      if (seen.has(lower)) continue;
      seen.add(lower);
      mentions.push(tok);
      if (mentions.length >= maxMentions) return mentions;
    }
  }

  return mentions;
}
