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
 * Format retrieved memories into a context block for injection.
 */
function formatMemoryContext(memories: Array<{ content: string }>): string {
  if (memories.length === 0) return '';
  const lines = memories.map((m) => `- ${m.content}`).join('\n');
  return `[Memory Context]\nThe following are relevant memories about this user:\n${lines}\n[End Memory Context]`;
}

/**
 * Extract the last user message content from an OpenAI-style messages array.
 */
function extractLastUserMessage(messages: any[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'user') {
      if (typeof msg.content === 'string') return msg.content;
      // Handle content arrays (multi-part messages).
      if (Array.isArray(msg.content)) {
        const textParts = msg.content
          .filter((p: any) => p.type === 'text')
          .map((p: any) => p.text);
        if (textParts.length > 0) return textParts.join(' ');
      }
    }
  }
  return null;
}

/**
 * Build a role/content message array from the request messages and assistant response.
 * This is the full exchange that gets sent to the server's extraction pipeline.
 */
function buildExchangeMessages(
  requestMessages: any[],
  assistantContent: string
): Array<{ role: string; content: string }> {
  const exchange: Array<{ role: string; content: string }> = [];

  for (const msg of requestMessages) {
    if (msg.role === 'user' || msg.role === 'assistant') {
      const content =
        typeof msg.content === 'string'
          ? msg.content
          : Array.isArray(msg.content)
            ? msg.content
                .filter((p: any) => p.type === 'text')
                .map((p: any) => p.text)
                .join(' ')
            : '';
      if (content) {
        exchange.push({ role: msg.role, content });
      }
    }
  }

  if (assistantContent) {
    exchange.push({ role: 'assistant', content: assistantContent });
  }

  return exchange;
}

/**
 * Fire extraction in the background, falling back to raw add if the server
 * doesn't support extract_from_messages.
 */
function fireExtraction(
  mcp: MemGClient,
  entityId: string,
  messages: any[],
  assistantContent: string
): void {
  const exchange = buildExchangeMessages(messages, assistantContent);
  if (exchange.length === 0) return;

  mcp.extractFromMessages(entityId, exchange).catch(() => {
    // Fallback: server may not have extraction pipeline configured.
    // Send the user messages as raw snippets (better than nothing).
    const userMessage = extractLastUserMessage(messages);
    if (userMessage) {
      mcp.add(entityId, [{ content: userMessage }]).catch(() => {});
    }
  });
}

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
export function wrapOpenAIClient(
  client: any,
  mcp: MemGClient,
  entity?: string,
  extract: boolean = true
): any {
  const originalCreate = client.chat.completions.create.bind(client.chat.completions);
  const entityId = entity || 'default';

  client.chat.completions.create = async function patchedCreate(
    params: any,
    requestOptions?: any
  ): Promise<any> {
    let augmentedParams = params;

    // Step 1: Search for relevant memories.
    const userMessage = extractLastUserMessage(params.messages || []);
    if (userMessage) {
      const searchResult = await mcp.search(entityId, userMessage, 10);
      if (searchResult.memories.length > 0) {
        const context = formatMemoryContext(searchResult.memories);
        augmentedParams = injectOpenAIContext(params, context);
      }
    }

    // Step 2: Call the original create.
    const response = await originalCreate(augmentedParams, requestOptions);

    // Step 3: Extract knowledge from the full exchange.
    if (extract) {
      try {
        if (params.stream) {
          return wrapOpenAIStream(response, mcp, entityId, params.messages || []);
        }
        const assistantContent = response?.choices?.[0]?.message?.content;
        if (typeof assistantContent === 'string' && assistantContent) {
          fireExtraction(mcp, entityId, params.messages || [], assistantContent);
        }
      } catch {
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
function wrapOpenAIStream(stream: any, mcp: MemGClient, entityId: string, requestMessages: any[]): any {
  if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
    const originalIterator = stream[Symbol.asyncIterator].bind(stream);
    let accumulated = '';

    stream[Symbol.asyncIterator] = function () {
      const iterator = originalIterator();
      return {
        async next(): Promise<IteratorResult<any>> {
          const result = await iterator.next();
          if (!result.done) {
            const delta = result.value?.choices?.[0]?.delta?.content;
            if (typeof delta === 'string') {
              accumulated += delta;
            }
          } else {
            if (accumulated) {
              fireExtraction(mcp, entityId, requestMessages, accumulated);
            }
          }
          return result;
        },
        async return(value?: any): Promise<IteratorResult<any>> {
          if (accumulated) {
            fireExtraction(mcp, entityId, requestMessages, accumulated);
          }
          if (iterator.return) return iterator.return(value);
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
 * `system` parameter and extract knowledge from responses via the server pipeline.
 *
 * @param client - An Anthropic SDK client instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier (defaults to "default").
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export function wrapAnthropicClient(
  client: any,
  mcp: MemGClient,
  entity?: string,
  extract: boolean = true
): any {
  const originalCreate = client.messages.create.bind(client.messages);
  const entityId = entity || 'default';

  client.messages.create = async function patchedCreate(
    params: any,
    requestOptions?: any
  ): Promise<any> {
    let augmentedParams = params;

    // Step 1: Search for relevant memories.
    const userMessage = extractLastUserMessage(params.messages || []);
    if (userMessage) {
      const searchResult = await mcp.search(entityId, userMessage, 10);
      if (searchResult.memories.length > 0) {
        const context = formatMemoryContext(searchResult.memories);
        augmentedParams = injectAnthropicContext(params, context);
      }
    }

    // Step 2: Call the original create.
    const response = await originalCreate(augmentedParams, requestOptions);

    // Step 3: Extract knowledge from the full exchange.
    if (extract) {
      try {
        if (params.stream) {
          return wrapAnthropicStream(response, mcp, entityId, params.messages || []);
        }
        // Non-streaming: extract the assistant response text.
        const blocks = response?.content;
        if (Array.isArray(blocks)) {
          const textBlock = blocks.find((b: any) => b.type === 'text');
          if (textBlock?.text) {
            fireExtraction(mcp, entityId, params.messages || [], textBlock.text);
          }
        }
      } catch {
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
 * systemInstruction and extract knowledge from responses via the server pipeline.
 *
 * @param client - A Gemini GenerativeModel instance (mutated in place).
 * @param mcp - MemGClient for memory operations.
 * @param entity - Optional entity identifier (defaults to "default").
 * @param extract - Whether to extract knowledge from responses (default: true).
 * @returns The patched client.
 */
export function wrapGeminiClient(
  client: any,
  mcp: MemGClient,
  entity?: string,
  extract: boolean = true
): any {
  const originalGenerate = client.generateContent.bind(client);
  const entityId = entity || 'default';

  client.generateContent = async function patchedGenerate(
    params: any
  ): Promise<any> {
    let augmentedParams = typeof params === 'string' ? { contents: [{ role: 'user', parts: [{ text: params }] }] } : params;

    // Step 1: Search for relevant memories.
    const userText = extractLastGeminiUserMessage(augmentedParams);
    if (userText) {
      const searchResult = await mcp.search(entityId, userText, 10);
      if (searchResult.memories.length > 0) {
        const context = formatMemoryContext(searchResult.memories);
        augmentedParams = injectGeminiContext(augmentedParams, context);
      }
    }

    // Step 2: Call the original generateContent.
    const response = await originalGenerate(augmentedParams);

    // Step 3: Extract knowledge from the full exchange.
    if (extract) {
      try {
        const text = response?.response?.candidates?.[0]?.content?.parts?.[0]?.text
          ?? response?.response?.text?.();
        if (text) {
          // Build exchange from Gemini's contents format.
          const requestMessages = geminiContentsToMessages(augmentedParams);
          fireExtraction(mcp, entityId, requestMessages, text);
        }
      } catch {
        // Fire-and-forget.
      }
    }

    return response;
  };

  return client;
}

/**
 * Convert Gemini contents array to a simple role/content messages array
 * for the extraction pipeline.
 */
function geminiContentsToMessages(params: any): Array<{ role: string; content: string }> {
  const contents = params?.contents;
  if (!Array.isArray(contents)) return [];
  const messages: Array<{ role: string; content: string }> = [];
  for (const c of contents) {
    const role = c.role === 'model' ? 'assistant' : c.role;
    const text = c.parts?.map((p: any) => p.text).filter(Boolean).join(' ');
    if (role && text) {
      messages.push({ role, content: text });
    }
  }
  return messages;
}

function extractLastGeminiUserMessage(params: any): string | null {
  const contents = params?.contents;
  if (!Array.isArray(contents)) return null;
  for (let i = contents.length - 1; i >= 0; i--) {
    if (contents[i].role === 'user' && contents[i].parts?.length > 0) {
      return contents[i].parts[0].text ?? null;
    }
  }
  return null;
}

function injectGeminiContext(params: any, contextText: string): any {
  const result = { ...params };
  const existing = result.systemInstruction;
  if (existing && existing.parts?.length > 0) {
    const existingText = existing.parts.map((p: any) => p.text).join('');
    result.systemInstruction = { parts: [{ text: existingText + '\n\n' + contextText }] };
  } else {
    result.systemInstruction = { parts: [{ text: contextText }] };
  }
  return result;
}

/**
 * Inject memory context into OpenAI-style messages.
 * Appends to existing system message if present, otherwise prepends a new one.
 */
function injectOpenAIContext(params: any, contextText: string): any {
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
function injectAnthropicContext(params: any, contextText: string): any {
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
function wrapAnthropicStream(stream: any, mcp: MemGClient, entityId: string, requestMessages: any[]): any {
  if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
    const originalIterator = stream[Symbol.asyncIterator].bind(stream);
    let accumulated = '';

    stream[Symbol.asyncIterator] = function () {
      const iterator = originalIterator();
      return {
        async next(): Promise<IteratorResult<any>> {
          const result = await iterator.next();
          if (!result.done) {
            const event = result.value;
            if (event?.type === 'content_block_delta' && event?.delta?.type === 'text_delta') {
              accumulated += event.delta.text || '';
            }
          } else {
            if (accumulated) {
              fireExtraction(mcp, entityId, requestMessages, accumulated);
            }
          }
          return result;
        },
        async return(value?: any): Promise<IteratorResult<any>> {
          if (accumulated) {
            fireExtraction(mcp, entityId, requestMessages, accumulated);
          }
          if (iterator.return) return iterator.return(value);
          return { done: true, value };
        },
      };
    };
  }

  return stream;
}
