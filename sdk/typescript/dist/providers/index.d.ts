/**
 * Provider detection for LLM clients.
 *
 * Identifies whether a client instance is from the OpenAI or Anthropic SDK
 * based on constructor name, with a fallback to constructor source inspection.
 */
/**
 * Detect the LLM provider of a client instance.
 *
 * @param client - An LLM SDK client instance.
 * @returns The detected provider name, or null if unrecognized.
 */
export declare function detectProvider(client: any): 'openai' | 'anthropic' | 'gemini' | null;
