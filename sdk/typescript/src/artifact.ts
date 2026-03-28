/**
 * Artifact storage and recall logic.
 * Port of Go memory/artifact.go.
 */

import type { Artifact } from './types.js';
import type { Store } from './store.js';
import type { Embedder } from './embedder.js';
import { HybridEngine, cosineSimilarity, dimensionMatch, type SearchCandidate } from './search.js';
import type { DetectedArtifact } from './artifact_detect.js';

/**
 * Persist detected artifacts, generating descriptions via LLM and checking
 * for superseding of existing artifacts by cosine similarity.
 */
export async function storeArtifacts(
  store: Store,
  embedder: Embedder,
  llmChat: (prompt: string) => Promise<string>,
  detected: DetectedArtifact[],
  existing: Artifact[],
  conversationId: string,
  entityId: string,
  turnNumber: number
): Promise<void> {
  if (detected.length === 0) return;

  const descriptions: string[] = [];
  for (const d of detected) {
    let content = d.content;
    if (content.length > 500) content = content.slice(0, 500);
    const desc = await llmChat('Describe this code/data in one sentence: ' + content);
    descriptions.push(desc.trim());
  }

  const vectors = await embedder.embed(descriptions);
  if (vectors.length !== descriptions.length) {
    throw new Error(`artifact embed: expected ${descriptions.length} vectors, got ${vectors.length}`);
  }

  for (let i = 0; i < detected.length; i++) {
    const d = detected[i];
    const newVec = vectors[i];

    const a: Artifact = {
      uuid: '',
      conversationId,
      entityId,
      content: d.content,
      artifactType: d.artifactType,
      language: d.language,
      description: descriptions[i],
      descriptionEmbedding: newVec,
      turnNumber,
      createdAt: new Date().toISOString(),
    };
    await store.insertArtifact(a);

    for (const ex of existing) {
      if (ex.supersededBy) continue;
      if (!ex.descriptionEmbedding || !dimensionMatch(newVec, ex.descriptionEmbedding)) continue;
      if (cosineSimilarity(newVec, ex.descriptionEmbedding) > 0.8) {
        try {
          await store.supersedeArtifact(ex.uuid, a.uuid);
        } catch (err) {
          console.warn(`[memg] artifact: supersede ${ex.uuid}:`, err);
        }
      }
    }
  }
}

/**
 * Retrieve relevant artifacts using hybrid search.
 */
export async function recallArtifacts(
  store: Store,
  engine: HybridEngine,
  queryVec: number[],
  queryText: string,
  entityId: string,
  conversationId: string,
  limit: number,
  threshold: number
): Promise<Artifact[]> {
  let artifacts: Artifact[];
  if (conversationId) {
    artifacts = await store.listActiveArtifacts(entityId, conversationId);
  } else {
    artifacts = await store.listActiveArtifactsByEntity(entityId);
  }

  if (artifacts.length === 0) return [];

  const candidates: SearchCandidate[] = artifacts.map((a) => ({
    id: a.uuid,
    content: a.description,
    embedding: a.descriptionEmbedding,
    createdAt: a.createdAt,
    temporalStatus: 'current',
    significance: 5,
    confidence: 1.0,
  }));

  const results = engine.rank(queryVec, queryText, candidates, limit, threshold);

  const byUUID = new Map<string, Artifact>();
  for (const a of artifacts) {
    byUUID.set(a.uuid, a);
  }

  const out: Artifact[] = [];
  for (const r of results) {
    const a = byUUID.get(r.id);
    if (a) out.push(a);
  }
  return out;
}
