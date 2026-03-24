export declare function buildRecallQuery(memg: any, entityId: string, userMessage: string): Promise<string>;
export declare function saveExchangeToSession(memg: any, entityId: string, messages: Array<{
    role: string;
    content: string;
}>): Promise<void>;
