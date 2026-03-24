/**
 * Anthropic provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **proxy**: Redirects traffic through the MemG reverse proxy.
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 */
import { MemGClient } from '../client.mjs';
import { wrapAnthropicClient } from '../intercept.mjs';
import { wrapAnthropicProxy } from '../proxy.mjs';
import { buildRecallQuery, saveExchangeToSession } from './session_helpers.mjs';
/**
 * Wrap an Anthropic client with MemG memory capabilities.
 */
export function wrap(client, opts) {
    if (opts.mode === 'native') {
        return wrapAnthropicNative(client, opts);
    }
    if (opts.mode === 'client') {
        const mcp = new MemGClient(opts.mcpUrl);
        return wrapAnthropicClient(client, mcp, opts.entity, opts.extract ?? true);
    }
    return wrapAnthropicProxy(client, opts.entity, opts.proxyUrl);
}
function wrapAnthropicNative(client, opts) {
    let memg = null;
    let initPromise = null;
    async function ensureInit() {
        if (!initPromise) {
            initPromise = (async () => {
                const { MemG } = await import('../index.mjs');
                memg = new MemG(opts.nativeConfig);
                client._memg = memg;
                await memg.init();
            })();
        }
        await initPromise;
        return memg;
    }
    const originalCreate = client.messages.create.bind(client.messages);
    client.messages.create = async function patchedCreate(params, requestOptions) {
        const m = await ensureInit();
        const entityId = opts.entity || 'default';
        let augmentedParams = params;
        const userMessage = extractLastUserMessage(params.messages || []);
        if (userMessage) {
            const queryText = await buildRecallQuery(m, entityId, userMessage);
            const context = await m.buildMemoryContext(entityId, queryText);
            if (context) {
                augmentedParams = injectAnthropicContext(params, context);
            }
        }
        // Step 2: Call the original create.
        const response = await originalCreate(augmentedParams, requestOptions);
        // Step 3: Extract knowledge from the exchange (fire-and-forget).
        if (opts.extract !== false) {
            try {
                const messages = (params.messages || [])
                    .filter((msg) => msg.role === 'user' || msg.role === 'assistant')
                    .map((msg) => ({
                    role: msg.role,
                    content: typeof msg.content === 'string'
                        ? msg.content
                        : Array.isArray(msg.content)
                            ? msg.content.filter((p) => p.type === 'text').map((p) => p.text).join(' ')
                            : '',
                }));
                if (params.stream) {
                    return wrapAnthropicStreamNative(response, m, entityId, messages);
                }
                // Non-streaming: extract assistant content.
                const contentBlocks = response?.content;
                if (Array.isArray(contentBlocks)) {
                    const textContent = contentBlocks
                        .filter((b) => b.type === 'text')
                        .map((b) => b.text)
                        .join('');
                    if (textContent) {
                        messages.push({ role: 'assistant', content: textContent });
                    }
                }
                saveExchangeToSession(m, entityId, messages).catch(() => { });
                m.extractFromMessages(entityId, messages).catch(() => { });
            }
            catch {
                // Never let extraction errors affect the response.
            }
        }
        return response;
    };
    // Attach close method so users can clean up.
    client._memg = null;
    client._memgClose = () => { if (memg)
        memg.close(); };
    return client;
}
function wrapAnthropicStreamNative(stream, memg, entityId, messages) {
    if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
        const originalIterator = stream[Symbol.asyncIterator].bind(stream);
        let accumulated = '';
        stream[Symbol.asyncIterator] = function () {
            const iterator = originalIterator();
            return {
                async next() {
                    const result = await iterator.next();
                    if (!result.done) {
                        const event = result.value;
                        if (event?.type === 'content_block_delta' && event?.delta?.type === 'text_delta') {
                            accumulated += event.delta.text || '';
                        }
                    }
                    else {
                        if (accumulated) {
                            messages.push({ role: 'assistant', content: accumulated });
                            memg.extractFromMessages(entityId, messages).catch(() => { });
                        }
                    }
                    return result;
                },
                async return(value) {
                    if (accumulated) {
                        messages.push({ role: 'assistant', content: accumulated });
                        memg.extractFromMessages(entityId, messages).catch(() => { });
                    }
                    if (iterator.return)
                        return iterator.return(value);
                    return { done: true, value };
                },
            };
        };
    }
    return stream;
}
function extractLastUserMessage(messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role === 'user') {
            if (typeof msg.content === 'string')
                return msg.content;
            if (Array.isArray(msg.content)) {
                const textParts = msg.content
                    .filter((p) => p.type === 'text')
                    .map((p) => p.text);
                if (textParts.length > 0)
                    return textParts.join(' ');
            }
        }
    }
    return null;
}
function injectAnthropicContext(params, contextText) {
    const existing = params.system;
    if (typeof existing === 'string' && existing) {
        return { ...params, system: existing + '\n\n' + contextText };
    }
    if (Array.isArray(existing)) {
        return {
            ...params,
            system: [...existing, { type: 'text', text: '\n\n' + contextText }],
        };
    }
    return { ...params, system: contextText };
}
