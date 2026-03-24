/**
 * Client-mode interception for LLM clients.
 *
 * Patches the LLM client's create method to:
 * 1. Query MCP for relevant memories before each call
 * 2. Inject memory context into the request
 * 3. Extract and store structured knowledge from responses via the server pipeline
 *
 * This mode does not require the MemG proxy to be running.
 */
import { MemGClient } from './client.js';
/**
 * Wrap an OpenAI client in client mode.
 *
 * Patches `client.chat.completions.create` to inject memory context and
 * extract knowledge from responses via the server's extraction pipeline.
 *
 * @param client - An OpenAI SDK client instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier (defaults to "default").
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export declare function wrapOpenAIClient(client: any, mcp: MemGClient, entity?: string, extract?: boolean): any;
/**
 * Wrap an Anthropic client in client mode.
 *
 * Patches `client.messages.create` to inject memory context into the
 * `system` parameter and extract knowledge from responses via the server pipeline.
 *
 * @param client - An Anthropic SDK client instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier (defaults to "default").
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export declare function wrapAnthropicClient(client: any, mcp: MemGClient, entity?: string, extract?: boolean): any;
/**
 * Wrap a Gemini client (GenerativeModel) in client mode.
 *
 * Patches `model.generateContent` to inject memory context into the
 * systemInstruction and extract knowledge from responses via the server pipeline.
 *
 * @param client - A Gemini GenerativeModel instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier (defaults to "default").
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export declare function wrapGeminiClient(client: any, mcp: MemGClient, entity?: string, extract?: boolean): any;
