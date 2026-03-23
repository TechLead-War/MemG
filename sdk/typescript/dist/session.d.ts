/**
 * Session management for the native MemG engine.
 * Handles sliding window sessions and conversation tracking.
 */
import type { Store } from './store';
import type { FactMessage, FactSession } from './types';
export interface SessionManager {
    store: Store;
    timeoutMs: number;
}
/**
 * Ensure an active session exists for the entity/process pair.
 * If an active session exists, its expiry is slid forward.
 * If no active session exists, a new one is created.
 */
export declare function ensureSession(store: Store, entityUuid: string, processUuid: string, timeoutMs: number): Promise<{
    session: FactSession;
    isNew: boolean;
}>;
/**
 * Ensure an active conversation exists within the session.
 * Returns the conversation UUID.
 */
export declare function ensureConversation(store: Store, sessionUuid: string, entityUuid: string): Promise<string>;
/**
 * Save a user message to the active conversation.
 */
export declare function saveUserMessage(store: Store, conversationUuid: string, content: string): Promise<void>;
/**
 * Save an assistant message to the active conversation.
 */
export declare function saveAssistantMessage(store: Store, conversationUuid: string, content: string): Promise<void>;
/**
 * Load recent conversation history.
 */
export declare function loadRecentHistory(store: Store, sessionUuid: string, maxTurns: number): Promise<FactMessage[]>;
