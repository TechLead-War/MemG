"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.wrapOpenAIClient = wrapOpenAIClient;
exports.wrapAnthropicClient = wrapAnthropicClient;
exports.wrapGeminiClient = wrapGeminiClient;
/**
 * Format retrieved memories into a context block for injection.
 */
function formatMemoryContext(memories) {
    if (memories.length === 0)
        return '';
    const lines = memories.map((m) => `- ${m.content}`).join('\n');
    return `[Memory Context]\nThe following are relevant memories about this user:\n${lines}\n[End Memory Context]`;
}
/**
 * Extract the last user message content from an OpenAI-style messages array.
 */
function extractLastUserMessage(messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role === 'user') {
            if (typeof msg.content === 'string')
                return msg.content;
            // Handle content arrays (multi-part messages).
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
function wrapOpenAIClient(client, mcp, entity, extract = true) {
    const originalCreate = client.chat.completions.create.bind(client.chat.completions);
    client.chat.completions.create = async function patchedCreate(params, requestOptions) {
        const entityId = entity || 'default';
        let augmentedParams = params;
        // Step 1: Search for relevant memories.
        try {
            const userMessage = extractLastUserMessage(params.messages || []);
            if (userMessage) {
                const searchResult = await mcp.search(entityId, userMessage, 10);
                if (searchResult.memories.length > 0) {
                    const context = formatMemoryContext(searchResult.memories);
                    augmentedParams = injectOpenAIContext(params, context);
                }
            }
        }
        catch (err) {
            console.warn('[memg] Failed to retrieve memories, proceeding without context:', err);
        }
        // Step 2: Call the original create.
        const response = await originalCreate(augmentedParams, requestOptions);
        // Step 3: Extract knowledge from response if enabled.
        if (extract && entity) {
            try {
                if (params.stream) {
                    // For streaming responses, return a wrapper that accumulates and extracts on completion.
                    return wrapOpenAIStream(response, mcp, entityId);
                }
                // Non-streaming: extract from the completed response.
                const assistantContent = response?.choices?.[0]?.message?.content;
                if (assistantContent && typeof assistantContent === 'string') {
                    const userMessage = extractLastUserMessage(params.messages || []);
                    if (userMessage) {
                        mcp
                            .add(entityId, [{ content: userMessage }])
                            .catch(() => { });
                    }
                }
            }
            catch {
                // Fire-and-forget: never let extraction errors affect the response.
            }
        }
        return response;
    };
    return client;
}
/**
 * Wrap an OpenAI streaming response to accumulate content and extract on completion.
 */
function wrapOpenAIStream(stream, mcp, entityId) {
    // If the stream has a Symbol.asyncIterator, wrap it.
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
                        // Stream completed — extract in background.
                        if (accumulated) {
                            mcp.add(entityId, [{ content: accumulated }]).catch(() => { });
                        }
                    }
                    return result;
                },
                async return(value) {
                    if (accumulated) {
                        mcp.add(entityId, [{ content: accumulated }]).catch(() => { });
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
function wrapAnthropicClient(client, mcp, entity, extract = true) {
    const originalCreate = client.messages.create.bind(client.messages);
    client.messages.create = async function patchedCreate(params, requestOptions) {
        const entityId = entity || 'default';
        let augmentedParams = params;
        // Step 1: Search for relevant memories.
        try {
            const userMessage = extractLastUserMessage(params.messages || []);
            if (userMessage) {
                const searchResult = await mcp.search(entityId, userMessage, 10);
                if (searchResult.memories.length > 0) {
                    const context = formatMemoryContext(searchResult.memories);
                    augmentedParams = injectAnthropicContext(params, context);
                }
            }
        }
        catch (err) {
            console.warn('[memg] Failed to retrieve memories, proceeding without context:', err);
        }
        // Step 2: Call the original create.
        const response = await originalCreate(augmentedParams, requestOptions);
        // Step 3: Extract knowledge from response if enabled.
        if (extract && entity) {
            try {
                if (params.stream) {
                    return wrapAnthropicStream(response, mcp, entityId);
                }
                // Non-streaming: extract user input as memory.
                const userMessage = extractLastUserMessage(params.messages || []);
                if (userMessage) {
                    mcp.add(entityId, [{ content: userMessage }]).catch(() => { });
                }
            }
            catch {
                // Fire-and-forget.
            }
        }
        return response;
    };
    return client;
}
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
function wrapGeminiClient(client, mcp, entity, extract = true) {
    const originalGenerate = client.generateContent.bind(client);
    client.generateContent = async function patchedGenerate(params) {
        const entityId = entity || 'default';
        let augmentedParams = typeof params === 'string' ? { contents: [{ role: 'user', parts: [{ text: params }] }] } : params;
        // Step 1: Search for relevant memories.
        try {
            const userText = extractLastGeminiUserMessage(augmentedParams);
            if (userText) {
                const searchResult = await mcp.search(entityId, userText, 10);
                if (searchResult.memories.length > 0) {
                    const context = formatMemoryContext(searchResult.memories);
                    augmentedParams = injectGeminiContext(augmentedParams, context);
                }
            }
        }
        catch (err) {
            console.warn('[memg] Failed to retrieve memories, proceeding without context:', err);
        }
        // Step 2: Call the original generateContent.
        const response = await originalGenerate(augmentedParams);
        // Step 3: Extract knowledge from response if enabled.
        if (extract && entity) {
            try {
                const text = response?.response?.candidates?.[0]?.content?.parts?.[0]?.text
                    ?? response?.response?.text?.();
                if (text) {
                    mcp.add(entityId, [{ content: text }]).catch(() => { });
                }
            }
            catch {
                // Fire-and-forget.
            }
        }
        return response;
    };
    return client;
}
function extractLastGeminiUserMessage(params) {
    const contents = params?.contents;
    if (!Array.isArray(contents))
        return null;
    for (let i = contents.length - 1; i >= 0; i--) {
        if (contents[i].role === 'user' && contents[i].parts?.length > 0) {
            return contents[i].parts[0].text ?? null;
        }
    }
    return null;
}
function injectGeminiContext(params, contextText) {
    const result = { ...params };
    const existing = result.systemInstruction;
    if (existing && existing.parts?.length > 0) {
        const existingText = existing.parts.map((p) => p.text).join('');
        result.systemInstruction = { parts: [{ text: existingText + '\n\n' + contextText }] };
    }
    else {
        result.systemInstruction = { parts: [{ text: contextText }] };
    }
    return result;
}
/**
 * Inject memory context into OpenAI-style messages.
 * Appends to existing system message if present, otherwise prepends a new one.
 */
function injectOpenAIContext(params, contextText) {
    const messages = [...(params.messages || [])];
    // Look for an existing system message and append to it.
    for (let i = 0; i < messages.length; i++) {
        if (messages[i].role === 'system') {
            messages[i] = {
                ...messages[i],
                content: (messages[i].content || '') + '\n\n' + contextText,
            };
            return { ...params, messages };
        }
    }
    // No system message found — prepend one.
    messages.unshift({ role: 'system', content: contextText });
    return { ...params, messages };
}
/**
 * Inject memory context into Anthropic's system parameter.
 * Handles both string and content-block-array formats.
 */
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
/**
 * Wrap an Anthropic streaming response to accumulate content and extract on completion.
 */
function wrapAnthropicStream(stream, mcp, entityId) {
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
                            mcp.add(entityId, [{ content: accumulated }]).catch(() => { });
                        }
                    }
                    return result;
                },
                async return(value) {
                    if (accumulated) {
                        mcp.add(entityId, [{ content: accumulated }]).catch(() => { });
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
//# sourceMappingURL=intercept.js.map