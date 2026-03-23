/**
 * Gemini provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 * - **proxy**: Not supported for Gemini (Gemini SDK does not use OpenAI-compatible endpoints).
 */
import type { WrapOptions } from '../types';
/**
 * Wrap a Gemini GenerativeModel with MemG memory capabilities.
 */
export declare function wrap(client: any, opts: WrapOptions): any;
