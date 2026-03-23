/**
 * OpenAI provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **proxy**: Redirects traffic through the MemG reverse proxy.
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 */

import { MemGClient } from '../client';
import { wrapOpenAIClient } from '../intercept';
import { wrapOpenAIProxy } from '../proxy';
import type { WrapOptions } from '../types';

/**
 * Wrap an OpenAI client with MemG memory capabilities.
 */
export function wrap(client: any, opts: WrapOptions): any {
  if (opts.mode === 'native') {
    return wrapOpenAINative(client, opts);
  }
  if (opts.mode === 'client') {
    const mcp = new MemGClient(opts.mcpUrl);
    return wrapOpenAIClient(client, mcp, opts.entity, opts.extract ?? true);
  }
  return wrapOpenAIProxy(client, opts.entity, opts.proxyUrl);
}

function wrapOpenAINative(client: any, opts: WrapOptions): any {
  const { MemG } = require('../index') as typeof import('../index');

  const memg = new MemG(opts.nativeConfig);
  let initPromise: Promise<void> | null = null;

  async function ensureInit(): Promise<typeof memg> {
    if (!initPromise) {
      initPromise = memg.init();
    }
    await initPromise;
    return memg;
  }

  const originalCreate = client.chat.completions.create.bind(client.chat.completions);

  client.chat.completions.create = async function patchedCreate(
    params: any,
    requestOptions?: any
  ): Promise<any> {
    const m = await ensureInit();
    const entityId = opts.entity || 'default';
    let augmentedParams = params;

    // Step 1: Build memory context and inject into messages.
    try {
      const userMessage = extractLastUserMessage(params.messages || []);
      if (userMessage) {
        const context = await m.buildMemoryContext(entityId, userMessage);
        if (context) {
          augmentedParams = injectOpenAIContext(params, context);
        }
      }
    } catch (err) {
      console.warn('[memg] Failed to build memory context, proceeding without:', err);
    }

    // Step 2: Call the original create.
    const response = await originalCreate(augmentedParams, requestOptions);

    // Step 3: Extract knowledge from the exchange (fire-and-forget).
    if (opts.extract !== false) {
      try {
        const messages = (params.messages || [])
          .filter((msg: any) => msg.role === 'user' || msg.role === 'assistant')
          .map((msg: any) => ({
            role: msg.role,
            content: typeof msg.content === 'string'
              ? msg.content
              : Array.isArray(msg.content)
                ? msg.content.filter((p: any) => p.type === 'text').map((p: any) => p.text).join(' ')
                : '',
          }));

        if (params.stream) {
          return wrapOpenAIStreamNative(response, m, entityId, messages);
        }

        const assistantContent = response?.choices?.[0]?.message?.content;
        if (assistantContent) {
          messages.push({ role: 'assistant', content: assistantContent });
        }

        m.extractFromMessages(entityId, messages).catch(() => {});
      } catch {
        // Never let extraction errors affect the response.
      }
    }

    return response;
  };

  // Attach close method so users can clean up.
  client._memg = memg;
  client._memgClose = () => memg.close();

  return client;
}

function wrapOpenAIStreamNative(
  stream: any,
  memg: any,
  entityId: string,
  messages: Array<{ role: string; content: string }>
): any {
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
              messages.push({ role: 'assistant', content: accumulated });
              memg.extractFromMessages(entityId, messages).catch(() => {});
            }
          }
          return result;
        },
        async return(value?: any): Promise<IteratorResult<any>> {
          if (accumulated) {
            messages.push({ role: 'assistant', content: accumulated });
            memg.extractFromMessages(entityId, messages).catch(() => {});
          }
          if (iterator.return) return iterator.return(value);
          return { done: true, value };
        },
      };
    };
  }
  return stream;
}

function extractLastUserMessage(messages: any[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'user') {
      if (typeof msg.content === 'string') return msg.content;
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

function injectOpenAIContext(params: any, contextText: string): any {
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
