"use strict";
/**
 * Proxy-mode wrapping for LLM clients.
 *
 * Redirects traffic through the MemG reverse proxy using the client SDK's
 * built-in `withOptions()` method. The proxy handles memory recall and
 * extraction transparently.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.wrapOpenAIProxy = wrapOpenAIProxy;
exports.wrapAnthropicProxy = wrapAnthropicProxy;
/**
 * Wrap an OpenAI client to route through the MemG proxy.
 *
 * Uses `client.withOptions()` to create a new client instance pointing at
 * the proxy URL with the entity header injected as a default header.
 *
 * @param client - An OpenAI SDK client instance.
 * @param entity - Optional entity identifier for memory scoping.
 * @param proxyUrl - MemG proxy URL (default: http://localhost:8787/v1).
 * @returns A new OpenAI client routed through the proxy.
 */
function wrapOpenAIProxy(client, entity, proxyUrl = 'http://localhost:8787/v1') {
    const headers = {};
    if (entity)
        headers['X-MemG-Entity'] = entity;
    return client.withOptions({ baseURL: proxyUrl, defaultHeaders: headers });
}
/**
 * Wrap an Anthropic client to route through the MemG proxy.
 *
 * Uses `client.withOptions()` to create a new client instance pointing at
 * the proxy URL with the entity header injected as a default header.
 *
 * @param client - An Anthropic SDK client instance.
 * @param entity - Optional entity identifier for memory scoping.
 * @param proxyUrl - MemG proxy URL (default: http://localhost:8787/v1).
 * @returns A new Anthropic client routed through the proxy.
 */
function wrapAnthropicProxy(client, entity, proxyUrl = 'http://localhost:8787/v1') {
    const headers = {};
    if (entity)
        headers['X-MemG-Entity'] = entity;
    return client.withOptions({ baseURL: proxyUrl, defaultHeaders: headers });
}
//# sourceMappingURL=proxy.js.map