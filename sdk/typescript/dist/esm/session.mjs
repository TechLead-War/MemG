/**
 * Session management for the native MemG engine.
 * Handles sliding window sessions and conversation tracking.
 */
/**
 * Ensure an active session exists for the entity/process pair.
 * If an active session exists, its expiry is slid forward.
 * If no active session exists, a new one is created.
 */
export async function ensureSession(store, entityUuid, processUuid, timeoutMs) {
    return await store.ensureSession(entityUuid, processUuid, timeoutMs);
}
/**
 * Ensure an active conversation exists within the session.
 * Returns the conversation UUID.
 */
export async function ensureConversation(store, sessionUuid, entityUuid) {
    const existing = await store.activeConversation(sessionUuid);
    if (existing)
        return existing.uuid;
    return await store.startConversation(sessionUuid, entityUuid);
}
/**
 * Save a user message to the active conversation.
 */
export async function saveUserMessage(store, conversationUuid, content) {
    await store.appendMessage(conversationUuid, {
        uuid: '',
        conversationId: conversationUuid,
        role: 'user',
        content,
        kind: 'text',
    });
}
/**
 * Save an assistant message to the active conversation.
 */
export async function saveAssistantMessage(store, conversationUuid, content) {
    if (!content.trim())
        return;
    await store.appendMessage(conversationUuid, {
        uuid: '',
        conversationId: conversationUuid,
        role: 'assistant',
        content: content.trim(),
        kind: 'text',
    });
}
/**
 * Load recent conversation history.
 */
export async function loadRecentHistory(store, sessionUuid, maxTurns) {
    const conv = await store.activeConversation(sessionUuid);
    if (!conv)
        return [];
    if (maxTurns > 0) {
        return await store.readRecentMessages(conv.uuid, maxTurns);
    }
    return await store.readMessages(conv.uuid);
}
