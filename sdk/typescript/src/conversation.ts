/**
 * Conversation message utilities: normalize, diff, and merge.
 * Matches the Go memory/conversation.go logic.
 */

interface Message {
  role: string;
  content: string;
}

/**
 * Drop non-conversation roles (e.g., system prompts), trim whitespace,
 * and return only user/assistant turns with non-empty content.
 */
export function normalizeConversationMessages(
  messages: Message[]
): Message[] {
  const out: Message[] = [];
  for (const msg of messages) {
    if (!msg) continue;
    const role = msg.role.trim();
    if (role !== 'user' && role !== 'assistant') continue;
    const content = msg.content.trim();
    if (content === '') continue;
    out.push({ role, content });
  }
  return out;
}

/**
 * Return only messages from `incoming` that are not already present in
 * `existing`. Uses sequence-based tail comparison (the tail of existing
 * matching the prefix of incoming) so that legitimately repeated messages
 * are not filtered out.
 */
export function diffIncomingMessages(
  existing: Message[],
  incoming: Message[]
): Message[] {
  const normalizedExisting = normalizeConversationMessages(existing);
  const normalizedIncoming = normalizeConversationMessages(incoming);

  if (normalizedExisting.length === 0) return normalizedIncoming;

  const overlap = overlapLength(normalizedExisting, normalizedIncoming);

  if (overlap >= normalizedIncoming.length) return [];
  return normalizedIncoming.slice(overlap);
}

/**
 * Merge stored history with the latest incoming request without duplicating
 * turns the client already replayed. Returns history + incoming[overlap:].
 */
export function mergeHistory(
  history: Message[],
  incoming: Message[]
): Message[] {
  const normalizedHistory = normalizeConversationMessages(history);
  const normalizedIncoming = normalizeConversationMessages(incoming);

  if (normalizedHistory.length === 0) return normalizedIncoming;
  if (normalizedIncoming.length === 0) return normalizedHistory;

  const overlap = overlapLength(normalizedHistory, normalizedIncoming);
  const merged: Message[] = [];
  merged.push(...normalizedHistory);
  merged.push(...normalizedIncoming.slice(overlap));
  return merged;
}

/**
 * Find the longest overlap between the tail of `existing` and the prefix
 * of `incoming`.
 */
function overlapLength(existing: Message[], incoming: Message[]): number {
  const existingLen = existing.length;
  let maxOverlap = incoming.length;
  if (maxOverlap > existingLen) maxOverlap = existingLen;

  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    let match = true;
    for (let i = 0; i < overlap; i++) {
      const e = existing[existingLen - overlap + i];
      const m = incoming[i];
      if (e.role !== m.role || e.content !== m.content) {
        match = false;
        break;
      }
    }
    if (match) return overlap;
  }
  return 0;
}
