"use strict";
/**
 * Provider detection for LLM clients.
 *
 * Identifies whether a client instance is from the OpenAI or Anthropic SDK
 * based on constructor name, with a fallback to constructor source inspection.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.detectProvider = detectProvider;
/**
 * Detect the LLM provider of a client instance.
 *
 * @param client - An LLM SDK client instance.
 * @returns The detected provider name, or null if unrecognized.
 */
function detectProvider(client) {
    const name = client?.constructor?.name;
    if (name === 'OpenAI')
        return 'openai';
    if (name === 'Anthropic')
        return 'anthropic';
    if (name === 'GoogleGenerativeAI' || name === 'GenerativeModel')
        return 'gemini';
    // Fallback: inspect constructor source for package identifiers.
    const mod = client?.constructor?.toString?.() ?? '';
    if (mod.includes('openai'))
        return 'openai';
    if (mod.includes('anthropic'))
        return 'anthropic';
    if (mod.includes('generative-ai') || mod.includes('google'))
        return 'gemini';
    // Fallback: check for Gemini SDK methods.
    if (typeof client?.generateContent === 'function')
        return 'gemini';
    return null;
}
//# sourceMappingURL=index.js.map