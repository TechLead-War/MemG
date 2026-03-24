"use strict";
/**
 * OpenAI provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **proxy**: Redirects traffic through the MemG reverse proxy.
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.wrap = wrap;
const client_js_1 = require("../client.js");
const intercept_js_1 = require("../intercept.js");
const proxy_js_1 = require("../proxy.js");
const session_helpers_js_1 = require("./session_helpers.js");
/**
 * Wrap an OpenAI client with MemG memory capabilities.
 */
function wrap(client, opts) {
    if (opts.mode === 'native') {
        return wrapOpenAINative(client, opts);
    }
    if (opts.mode === 'client') {
        const mcp = new client_js_1.MemGClient(opts.mcpUrl);
        return (0, intercept_js_1.wrapOpenAIClient)(client, mcp, opts.entity, opts.extract ?? true);
    }
    return (0, proxy_js_1.wrapOpenAIProxy)(client, opts.entity, opts.proxyUrl);
}
function wrapOpenAINative(client, opts) {
    let memg = null;
    let initPromise = null;
    async function ensureInit() {
        if (!initPromise) {
            initPromise = (async () => {
                const { MemG } = await Promise.resolve().then(() => __importStar(require('../index.js')));
                memg = new MemG(opts.nativeConfig);
                client._memg = memg;
                await memg.init();
            })();
        }
        await initPromise;
        return memg;
    }
    const originalCreate = client.chat.completions.create.bind(client.chat.completions);
    client.chat.completions.create = async function patchedCreate(params, requestOptions) {
        const m = await ensureInit();
        const entityId = opts.entity || 'default';
        let augmentedParams = params;
        const userMessage = extractLastUserMessage(params.messages || []);
        if (userMessage) {
            const queryText = await (0, session_helpers_js_1.buildRecallQuery)(m, entityId, userMessage);
            const context = await m.buildMemoryContext(entityId, queryText);
            if (context) {
                augmentedParams = injectOpenAIContext(params, context);
            }
        }
        // Step 2: Call the original create.
        const response = await originalCreate(augmentedParams, requestOptions);
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
                    return wrapOpenAIStreamNative(response, m, entityId, messages);
                }
                const assistantContent = response?.choices?.[0]?.message?.content;
                if (assistantContent) {
                    messages.push({ role: 'assistant', content: assistantContent });
                }
                (0, session_helpers_js_1.saveExchangeToSession)(m, entityId, messages).catch(() => { });
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
function wrapOpenAIStreamNative(stream, memg, entityId, messages) {
    if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
        const originalIterator = stream[Symbol.asyncIterator].bind(stream);
        let accumulated = '';
        stream[Symbol.asyncIterator] = function () {
            const iterator = originalIterator();
            return {
                async next() {
                    const result = await iterator.next();
                    if (!result.done) {
                        const delta = result.value?.choices?.[0]?.delta?.content;
                        if (typeof delta === 'string') {
                            accumulated += delta;
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
function injectOpenAIContext(params, contextText) {
    const messages = [...(params.messages || [])];
    for (let i = 0; i < messages.length; i++) {
        if (messages[i].role === 'system') {
            messages[i] = {
                ...messages[i],
                content: (messages[i].content || '') + '\n\n' + contextText,
            };
            return { ...params, messages };
        }
    }
    messages.unshift({ role: 'system', content: contextText });
    return { ...params, messages };
}
//# sourceMappingURL=openai.js.map