/**
 * Client-mode interception for LLM clients.
 *
 * Patches the LLM client's create method to:
 * 1. Query MCP for relevant memories before each call
 * 2. Inject memory context into the request
 * 3. Optionally extract and store knowledge from responses
 *
 * This mode does not require the MemG proxy to be running.
 */
import { MemGClient } from './client';
/**
 * Wrap an OpenAI client in client mode.
 *
 * Patches `client.chat.completions.create` to inject memory context and
 * optionally extract knowledge from responses.
 *
 * @param client - An OpenAI SDK client instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier.
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export declare function wrapOpenAIClient(client: any, mcp: MemGClient, entity?: string, extract?: boolean): any;
/**
 * Wrap an Anthropic client in client mode.
 *
 * Patches `client.messages.create` to inject memory context into the
 * `system` parameter and optionally extract knowledge from responses.
 *
 * Note: Anthropic uses `system` as a top-level parameter, not a message role.
 *
 * @param client - An Anthropic SDK client instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier.
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export declare function wrapAnthropicClient(client: any, mcp: MemGClient, entity?: string, extract?: boolean): any;
/**
 * Wrap a Gemini client (GenerativeModel) in client mode.
 *
 * Patches `model.generateContent` to inject memory context into the
 * systemInstruction and optionally extract knowledge from responses.
 *
 * @param client - A Gemini GenerativeModel instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier.
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export declare function wrapGeminiClient(client: any, mcp: MemGClient, entity?: string, extract?: boolean): any;
