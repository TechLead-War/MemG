/**
 * Gemini provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 * - **proxy**: Not supported for Gemini (Gemini SDK does not use OpenAI-compatible endpoints).
 */

import { MemGClient } from '../client';
import { wrapGeminiClient } from '../intercept';
import type { WrapOptions } from '../types';

/**
 * Wrap a Gemini GenerativeModel with MemG memory capabilities.
 */
export function wrap(client: any, opts: WrapOptions): any {
  if (opts.mode === 'native') {
    return wrapGeminiNative(client, opts);
  }
  if (opts.mode === 'client') {
    const mcp = new MemGClient(opts.mcpUrl);
    return wrapGeminiClient(client, mcp, opts.entity, opts.extract ?? true);
  }
  // Proxy mode: Gemini SDK doesn't route through OpenAI-compatible proxy.
  // Fall through to native mode as the best alternative.
  console.warn('[memg] Proxy mode is not supported for Gemini clients. Using native mode instead.');
  return wrapGeminiNative(client, opts);
}

function wrapGeminiNative(client: any, opts: WrapOptions): any {
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

  const originalGenerate = client.generateContent.bind(client);

  client.generateContent = async function patchedGenerate(
    params: any
  ): Promise<any> {
    const m = await ensureInit();
    const entityId = opts.entity || 'default';

    // Normalize string input to Gemini format.
    let augmentedParams =
      typeof params === 'string'
        ? { contents: [{ role: 'user', parts: [{ text: params }] }] }
        : { ...params };

    // Step 1: Build memory context and inject into systemInstruction.
    try {
      const userText = extractLastGeminiUserMessage(augmentedParams);
      if (userText) {
        const context = await m.buildMemoryContext(entityId, userText);
        if (context) {
          augmentedParams = injectGeminiContext(augmentedParams, context);
        }
      }
    } catch (err) {
      console.warn('[memg] Failed to build memory context, proceeding without:', err);
    }

    // Step 2: Call the original generateContent.
    const response = await originalGenerate(augmentedParams);

    // Step 3: Extract knowledge from the exchange (fire-and-forget).
    if (opts.extract !== false) {
      try {
        const contents = augmentedParams.contents || [];
        const messages = contents
          .filter((c: any) => c.role === 'user' || c.role === 'model')
          .map((c: any) => ({
            role: c.role === 'model' ? 'assistant' : c.role,
            content: c.parts?.map((p: any) => p.text).join('') ?? '',
          }));

        const assistantText =
          response?.response?.candidates?.[0]?.content?.parts?.[0]?.text ??
          response?.response?.text?.();
        if (assistantText) {
          messages.push({ role: 'assistant', content: assistantText });
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
    result.systemInstruction = {
      parts: [{ text: existingText + '\n\n' + contextText }],
    };
  } else {
    result.systemInstruction = { parts: [{ text: contextText }] };
  }
  return result;
}
