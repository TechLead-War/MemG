/**
 * Default configuration for the native MemG engine.
 * All values match the Go DefaultConfig().
 */
import type { NativeConfig } from './types.js';
export declare const DEFAULT_CONFIG: Required<NativeConfig>;
/**
 * Merge user config with defaults.
 */
export declare function resolveConfig(user?: NativeConfig): Required<NativeConfig>;
