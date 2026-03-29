export async function buildRecallQuery(
  memg: any,
  entityId: string,
  userMessage: string
): Promise<string> {
  return userMessage;
}

const sessionCache = new Map<string, { sessionUuid: string; expiresAt: number }>();

export async function saveExchangeToSession(
  memg: any,
  entityId: string,
  messages: Array<{ role: string; content: string }>
): Promise<void> {
  const store = memg.getStore();
  const config = memg.getConfig();
  if (!store || !config.sessionTimeout) return;

  const entityUuid = await store.upsertEntity(entityId);

  // Use cached session if still valid.
  let sessionUuid: string;
  const cached = sessionCache.get(entityId);
  const now = Date.now();
  if (cached && cached.expiresAt > now) {
    sessionUuid = cached.sessionUuid;
  } else {
    const { session } = await store.ensureSession(entityUuid, 'default', config.sessionTimeout);
    sessionUuid = session.uuid;
    sessionCache.set(entityId, { sessionUuid, expiresAt: now + config.sessionTimeout / 2 });
  }

  const {
    ensureConversation,
    saveUserMessage,
    saveAssistantMessage,
  } = await import('../session.js');

  const convUuid = await ensureConversation(store, sessionUuid, entityUuid);

  const lastUser = messages.filter((m) => m.role === 'user').pop();
  const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
  if (lastUser?.content) await saveUserMessage(store, convUuid, lastUser.content);
  if (lastAssistant?.content) await saveAssistantMessage(store, convUuid, lastAssistant.content);
}
