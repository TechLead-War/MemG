/**
 * Gemini provider wrapping for MemG.
 *
 * Supports three modes:
 * - **native**: Full in-process engine (no Go server needed).
 * - **client**: Intercepts calls locally, querying MCP for memory context.
 * - **proxy**: Not supported for Gemini (Gemini SDK does not use OpenAI-compatible endpoints).
 */
import { MemGClient } from '../client.mjs';
import { wrapGeminiClient } from '../intercept.mjs';
import { buildRecallQuery, saveExchangeToSession } from './session_helpers.mjs';
/**
 * Wrap a Gemini GenerativeModel with MemG memory capabilities.
 */
export function wrap(client, opts) {
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
function wrapGeminiNative(client, opts) {
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
    const originalGenerate = client.generateContent.bind(client);
    client.generateContent = async function patchedGenerate(params) {
        const m = await ensureInit();
        const entityId = opts.entity || 'default';
        // Normalize string input to Gemini format.
        let augmentedParams = typeof params === 'string'
            ? { contents: [{ role: 'user', parts: [{ text: params }] }] }
            : { ...params };
        const userText = extractLastGeminiUserMessage(augmentedParams);
        if (userText) {
            const queryText = await buildRecallQuery(m, entityId, userText);
            const context = await m.buildMemoryContext(entityId, queryText);
            if (context) {
                augmentedParams = injectGeminiContext(augmentedParams, context);
            }
        }
        // Step 2: Call the original generateContent.
        const response = await originalGenerate(augmentedParams);
        // Step 3: Extract knowledge from the exchange (fire-and-forget).
        if (opts.extract !== false) {
            try {
                const contents = augmentedParams.contents || [];
                const messages = contents
                    .filter((c) => c.role === 'user' || c.role === 'model')
                    .map((c) => ({
                    role: c.role === 'model' ? 'assistant' : c.role,
                    content: c.parts?.map((p) => p.text).join('') ?? '',
                }));
                const assistantText = response?.response?.candidates?.[0]?.content?.parts?.[0]?.text ??
                    response?.response?.text?.();
                if (assistantText) {
                    messages.push({ role: 'assistant', content: assistantText });
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
        result.systemInstruction = {
            parts: [{ text: existingText + '\n\n' + contextText }],
        };
    }
    else {
        result.systemInstruction = { parts: [{ text: contextText }] };
    }
    return result;
}
