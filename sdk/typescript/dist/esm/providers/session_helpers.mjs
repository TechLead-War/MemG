export async function buildRecallQuery(memg, entityId, userMessage) {
    return userMessage;
}
const sessionCache = new Map();
export async function saveExchangeToSession(memg, entityId, messages) {
    const store = memg.getStore();
    const config = memg.getConfig();
    if (!store || !config.sessionTimeout)
        return;
    const entityUuid = await store.upsertEntity(entityId);
    // Use cached session if still valid.
    let sessionUuid;
    const cached = sessionCache.get(entityId);
    const now = Date.now();
    if (cached && cached.expiresAt > now) {
        sessionUuid = cached.sessionUuid;
    }
    else {
        const { session } = await store.ensureSession(entityUuid, 'default', config.sessionTimeout);
        sessionUuid = session.uuid;
        sessionCache.set(entityId, { sessionUuid, expiresAt: now + config.sessionTimeout / 2 });
    }
    const { ensureConversation, saveUserMessage, saveAssistantMessage, } = await import('../session.mjs');
    const convUuid = await ensureConversation(store, sessionUuid, entityUuid);
    const lastUser = messages.filter((m) => m.role === 'user').pop();
    const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
    if (lastUser?.content)
        await saveUserMessage(store, convUuid, lastUser.content);
    if (lastAssistant?.content)
        await saveAssistantMessage(store, convUuid, lastAssistant.content);
}
