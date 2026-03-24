"use strict";
/**
 * Recall: retrieve relevant facts and summaries using hybrid search.
 * Matches the Go memory.Recall and memory.RecallSummaries logic.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.recallFacts = recallFacts;
exports.recallFactsWithVector = recallFactsWithVector;
exports.recallSummaries = recallSummaries;
exports.recallSummariesWithVector = recallSummariesWithVector;
const search_js_1 = require("./search.js");
/**
 * Recall facts for an entity using hybrid search.
 */
async function recallFacts(store, engine, embedder, queryText, entityUuid, limit, threshold, maxCandidates, filter) {
    let vectors;
    try {
        vectors = await embedder.embed([queryText]);
    }
    catch {
        return [];
    }
    if (vectors.length === 0 || vectors[0].length === 0)
        return [];
    const queryVec = vectors[0];
    const queryModel = embedder.modelName();
    return recallFactsWithVector(store, engine, queryVec, queryModel, queryText, entityUuid, limit, threshold, maxCandidates, filter);
}
/**
 * Recall facts using a pre-computed query vector.
 */
async function recallFactsWithVector(store, engine, queryVec, queryModel, queryText, entityUuid, limit, threshold, maxCandidates, filter) {
    if (queryVec.length === 0)
        return [];
    const effectiveFilter = {
        ...filter,
        excludeExpired: true,
    };
    if (maxCandidates <= 0)
        maxCandidates = 10000;
    const facts = await store.listFactsForRecall(entityUuid, effectiveFilter, maxCandidates);
    if (facts.length === 0)
        return [];
    const candidates = buildRecallCandidates(queryVec, queryModel, facts);
    const results = engine.rank(queryVec, queryText, candidates, limit, threshold);
    return results.map((r) => ({
        id: r.id,
        content: r.content,
        score: r.score,
        temporalStatus: r.temporalStatus,
        significance: r.significance,
        createdAt: r.createdAt,
    }));
}
/**
 * Recall conversation summaries using hybrid search.
 */
async function recallSummaries(store, engine, embedder, queryText, entityUuid, limit, threshold) {
    let vectors;
    try {
        vectors = await embedder.embed([queryText]);
    }
    catch {
        return [];
    }
    if (vectors.length === 0 || vectors[0].length === 0)
        return [];
    return recallSummariesWithVector(store, engine, vectors[0], queryText, entityUuid, limit, threshold);
}
/**
 * Recall summaries using a pre-computed query vector.
 */
async function recallSummariesWithVector(store, engine, queryVec, queryText, entityUuid, limit, threshold) {
    if (queryVec.length === 0)
        return [];
    const convs = await store.listConversationSummaries(entityUuid, 0);
    if (convs.length === 0)
        return [];
    const candidates = convs.map((c) => {
        let embedding = c.summaryEmbedding;
        if (queryVec.length > 0 && !(0, search_js_1.dimensionMatch)(queryVec, embedding)) {
            embedding = undefined;
        }
        return {
            id: c.uuid,
            content: c.summary,
            embedding,
            createdAt: c.createdAt,
            temporalStatus: 'current',
            significance: 0,
            confidence: 1.0,
        };
    });
    const results = engine.rank(queryVec, queryText, candidates, limit, threshold);
    return results.map((r) => ({
        conversationId: r.id,
        summary: r.content,
        score: r.score,
        createdAt: r.createdAt,
    }));
}
function buildRecallCandidates(queryVec, queryModel, facts) {
    return facts.map((f) => {
        let confidence = f.confidence;
        if (confidence === 0)
            confidence = 1.0;
        let embedding = f.embedding;
        if (queryVec.length > 0 && !(0, search_js_1.dimensionMatch)(queryVec, embedding)) {
            embedding = undefined;
        }
        else if (queryModel &&
            f.embeddingModel &&
            f.embeddingModel !== queryModel) {
            embedding = undefined;
        }
        return {
            id: f.uuid,
            content: f.content,
            embedding,
            createdAt: f.createdAt,
            temporalStatus: f.temporalStatus,
            significance: f.significance,
            confidence,
        };
    });
}
//# sourceMappingURL=recall.js.map