/**
 * Proxy-mode wrapping for LLM clients.
 *
 * Redirects traffic through the MemG reverse proxy using the client SDK's
 * built-in `withOptions()` method. The proxy handles memory recall and
 * extraction transparently.
 */

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
export function wrapOpenAIProxy(
  client: any,
  entity?: string,
  proxyUrl: string = 'http://localhost:8787/v1'
): any {
  const headers: Record<string, string> = {};
  if (entity) headers['X-MemG-Entity'] = entity;
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
export function wrapAnthropicProxy(
  client: any,
  entity?: string,
  proxyUrl: string = 'http://localhost:8787/v1'
): any {
  const headers: Record<string, string> = {};
  if (entity) headers['X-MemG-Entity'] = entity;
  return client.withOptions({ baseURL: proxyUrl, defaultHeaders: headers });
}
