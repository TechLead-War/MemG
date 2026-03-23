/**
 * Anthropic provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **proxy**: Redirects traffic through the MemG reverse proxy.
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 */
import type { WrapOptions } from '../types';
/**
 * Wrap an Anthropic client with MemG memory capabilities.
 */
export declare function wrap(client: any, opts: WrapOptions): any;
