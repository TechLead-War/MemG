/**
 * SQLite-backed repository for the native MemG engine.
 * Uses better-sqlite3 (synchronous API) for fast, in-process persistence.
 */

import { createHash, randomUUID } from 'crypto';
import type {
  Artifact,
  Fact,
  FactConversation,
  FactEntity,
  FactFilter,
  FactMessage,
  FactSession,
  TurnSummary,
} from './types.js';
import { SCHEMA_DDL } from './schema.js';
import BetterSqlite3 from 'better-sqlite3';

type Database = InstanceType<typeof BetterSqlite3>;

/** A value that may be synchronous or a Promise. */
type MaybeAsync<T> = T | Promise<T>;

/**
 * Store interface for pluggable persistence backends.
 *
 * Methods return `T | Promise<T>` so both synchronous stores (SQLite)
 * and asynchronous stores (Postgres, MySQL) can implement this interface.
 * All callsites use `await` — this is a no-op for sync values.
 *
 * Built-in implementations: `MemGStore` (SQLite), `PostgresStore`, `MySQLStore`.
 */
export interface Store {
  // Entity
  upsertEntity(externalId: string): MaybeAsync<string>;
  lookupEntity(externalId: string): MaybeAsync<FactEntity | null>;
  listEntityUuids(limit: number): MaybeAsync<string[]>;

  // Fact metadata (without embedding decoding)
  listFactsMetadata(entityUuid: string, filter: FactFilter, limit: number): MaybeAsync<Fact[]>;

  // Process
  upsertProcess(externalId: string): MaybeAsync<string>;
  insertProcessAttribute(processUuid: string, key: string, value: string): MaybeAsync<void>;

  // Canonical slots
  listCanonicalSlots(): MaybeAsync<Array<{name: string; embedding?: number[]; createdAt: string}>>;
  insertCanonicalSlot(name: string, embedding: number[]): MaybeAsync<void>;
  findCanonicalSlotByName(name: string): MaybeAsync<{name: string; embedding?: number[]; createdAt: string} | null>;

  // Fact CRUD
  insertFact(entityUuid: string, fact: Fact): MaybeAsync<void>;
  insertFacts(entityUuid: string, facts: Fact[]): MaybeAsync<void>;
  listFacts(entityUuid: string, limit: number): MaybeAsync<Fact[]>;
  listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): MaybeAsync<Fact[]>;
  listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): MaybeAsync<Fact[]>;
  findFactByKey(entityUuid: string, contentKey: string): MaybeAsync<Fact | null>;

  // Fact lifecycle
  reinforceFact(factUuid: string, newExpiresAt: string | null): MaybeAsync<void>;
  updateTemporalStatus(factUuid: string, status: string): MaybeAsync<void>;
  updateSignificance(factUuid: string, sig: number): MaybeAsync<void>;
  deleteFact(entityUuid: string, factUuid: string): MaybeAsync<void>;
  deleteEntityFacts(entityUuid: string): MaybeAsync<number>;
  pruneExpiredFacts(entityUuid: string, now: string): MaybeAsync<number>;
  pruneStaleSummaries(days?: number): MaybeAsync<number>;
  updateRecallUsage(factUuids: string[]): MaybeAsync<void>;
  listUnembeddedFacts(entityUuid: string, limit: number): MaybeAsync<Fact[]>;
  updateFactEmbedding(factUuid: string, embedding: number[], model: string): MaybeAsync<void>;

  // Session
  ensureSession(
    entityUuid: string,
    processUuid: string,
    timeoutMs: number
  ): MaybeAsync<{ session: FactSession; isNew: boolean }>;

  // Conversation
  startConversation(sessionUuid: string, entityUuid: string): MaybeAsync<string>;
  activeConversation(sessionUuid: string): MaybeAsync<FactConversation | null>;
  appendMessage(conversationUuid: string, msg: FactMessage): MaybeAsync<void>;
  readMessages(conversationUuid: string): MaybeAsync<FactMessage[]>;
  readRecentMessages(conversationUuid: string, limit: number): MaybeAsync<FactMessage[]>;
  listConversationSummaries(entityUuid: string, limit: number): MaybeAsync<FactConversation[]>;
  updateConversationSummary(conversationUuid: string, summary: string, embedding: number[], embeddingModel: string): MaybeAsync<void>;
  findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): MaybeAsync<FactConversation | null>;

  // Turn Summary
  insertTurnSummary(ts: TurnSummary): MaybeAsync<void>;
  listTurnSummaries(conversationId: string): MaybeAsync<TurnSummary[]>;
  countTurnSummaries(conversationId: string): MaybeAsync<number>;
  deleteTurnSummaries(conversationId: string, uuids: string[]): MaybeAsync<void>;

  // Artifact
  insertArtifact(a: Artifact): MaybeAsync<void>;
  supersedeArtifact(oldUuid: string, newUuid: string): MaybeAsync<void>;
  listActiveArtifacts(entityId: string, conversationId: string): MaybeAsync<Artifact[]>;
  listActiveArtifactsByEntity(entityId: string): MaybeAsync<Artifact[]>;

  // Session metadata
  updateSessionMentions(sessionUuid: string, mentions: string[]): MaybeAsync<void>;
  incrementSessionMessageCount(sessionUuid: string): MaybeAsync<void>;
  getSessionMentions(sessionUuid: string): MaybeAsync<string[]>;

  // Lifecycle
  close(): MaybeAsync<void>;
}

function nowISO(): string {
  return new Date().toISOString();
}

function encodeEmbedding(vec: number[]): Buffer {
  const buf = Buffer.alloc(vec.length * 4);
  for (let i = 0; i < vec.length; i++) {
    buf.writeFloatLE(vec[i], i * 4);
  }
  return buf;
}

function decodeEmbedding(raw: Buffer | null | undefined): number[] | undefined {
  if (!raw || raw.length === 0) return undefined;
  if (raw.length % 4 !== 0 || raw[0] === 0x5b) {
    // Might be JSON
    try {
      return JSON.parse(raw.toString()) as number[];
    } catch {
      return undefined;
    }
  }
  const count = raw.length / 4;
  const out: number[] = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = raw.readFloatLE(i * 4);
  }
  return out;
}

function rowToFact(row: any): Fact {
  return {
    uuid: row.uuid,
    content: row.content,
    embedding: decodeEmbedding(row.embedding),
    createdAt: row.created_at ?? undefined,
    updatedAt: row.updated_at ?? undefined,
    factType: row.fact_type ?? 'identity',
    temporalStatus: row.temporal_status ?? 'current',
    significance: row.significance ?? 5,
    contentKey: row.content_key ?? '',
    referenceTime: row.reference_time ?? undefined,
    expiresAt: row.expires_at ?? undefined,
    reinforcedAt: row.reinforced_at ?? undefined,
    reinforcedCount: row.reinforced_count ?? 0,
    tag: row.tag ?? '',
    slot: row.slot ?? '',
    confidence: row.confidence ?? 1.0,
    embeddingModel: row.embedding_model ?? '',
    sourceRole: row.source_role ?? '',
    recallCount: row.recall_count ?? 0,
    lastRecalledAt: row.last_recalled_at ?? undefined,
  };
}

function rowToConversation(row: any): FactConversation {
  return {
    uuid: row.uuid,
    sessionId: row.session_id,
    entityId: row.entity_id ?? '',
    summary: row.summary ?? '',
    summaryEmbedding: decodeEmbedding(row.summary_embedding),
    summaryEmbeddingModel: row.summary_embedding_model ?? '',
    createdAt: row.created_at ?? undefined,
    updatedAt: row.updated_at ?? undefined,
  };
}

function rowToMessage(row: any): FactMessage {
  return {
    uuid: row.uuid,
    conversationId: row.conversation_id,
    role: row.role,
    content: row.content,
    kind: row.kind ?? 'text',
    createdAt: row.created_at ?? undefined,
  };
}

function rowToTurnSummary(row: any): TurnSummary {
  return {
    uuid: row.uuid,
    conversationId: row.conversation_id,
    entityId: row.entity_id,
    startTurn: row.start_turn,
    endTurn: row.end_turn,
    summary: row.summary,
    summaryEmbedding: decodeEmbedding(row.summary_embedding),
    isOverview: row.is_overview === 1,
    createdAt: row.created_at,
  };
}

function rowToArtifact(row: any): Artifact {
  return {
    uuid: row.uuid,
    conversationId: row.conversation_id,
    entityId: row.entity_id,
    content: row.content,
    artifactType: row.artifact_type ?? 'code',
    language: row.language ?? '',
    description: row.description ?? '',
    descriptionEmbedding: decodeEmbedding(row.description_embedding),
    supersededBy: row.superseded_by ?? undefined,
    turnNumber: row.turn_number ?? 0,
    createdAt: row.created_at,
  };
}

const FACT_COLUMNS = `uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at`;

function buildFactFilterClauses(filter: FactFilter): { clauses: string; args: any[] } {
  let clauses = '';
  const args: any[] = [];

  if (filter.types && filter.types.length > 0) {
    clauses += ` AND fact_type IN (${filter.types.map(() => '?').join(',')})`;
    args.push(...filter.types);
  }
  if (filter.statuses && filter.statuses.length > 0) {
    clauses += ` AND temporal_status IN (${filter.statuses.map(() => '?').join(',')})`;
    args.push(...filter.statuses);
  }
  if (filter.tags && filter.tags.length > 0) {
    clauses += ` AND tag IN (${filter.tags.map(() => '?').join(',')})`;
    args.push(...filter.tags);
  }
  if (filter.minSignificance && filter.minSignificance > 0) {
    clauses += ` AND significance >= ?`;
    args.push(filter.minSignificance);
  }
  if (filter.excludeExpired) {
    clauses += ` AND (expires_at IS NULL OR expires_at > ?)`;
    args.push(nowISO());
  }
  if (filter.slots && filter.slots.length > 0) {
    clauses += ` AND slot IN (${filter.slots.map(() => '?').join(',')})`;
    args.push(...filter.slots);
  }
  if (filter.minConfidence && filter.minConfidence > 0) {
    clauses += ` AND confidence >= ?`;
    args.push(filter.minConfidence);
  }
  if (filter.sourceRoles && filter.sourceRoles.length > 0) {
    clauses += ` AND source_role IN (${filter.sourceRoles.map(() => '?').join(',')})`;
    args.push(...filter.sourceRoles);
  }
  if (filter.unembeddedOnly) {
    clauses += ` AND embedding IS NULL`;
  }
  if (filter.maxSignificance && filter.maxSignificance > 0) {
    clauses += ` AND significance <= ?`;
    args.push(filter.maxSignificance);
  }
  if (filter.referenceTimeAfter) {
    clauses += ` AND reference_time IS NOT NULL AND reference_time >= ?`;
    args.push(filter.referenceTimeAfter);
  }
  if (filter.referenceTimeBefore) {
    clauses += ` AND reference_time IS NOT NULL AND reference_time <= ?`;
    args.push(filter.referenceTimeBefore);
  }

  return { clauses, args };
}

export class MemGStore implements Store {
  private db: Database;

  constructor(dbPath: string) {
    this.db = new BetterSqlite3(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.createSchema();
  }

  private createSchema(): void {
    const tx = this.db.transaction(() => {
      for (const ddl of SCHEMA_DDL) {
        this.db.exec(ddl);
      }
    });
    tx();
  }

  // ---- Entity ----

  upsertEntity(externalId: string): string {
    const id = randomUUID();
    this.db
      .prepare(`INSERT OR IGNORE INTO mg_entity (uuid, external_id) VALUES (?, ?)`)
      .run(id, externalId);
    const row = this.db
      .prepare(`SELECT uuid FROM mg_entity WHERE external_id = ?`)
      .get(externalId) as { uuid: string } | undefined;
    return row!.uuid;
  }

  lookupEntity(externalId: string): FactEntity | null {
    const row = this.db
      .prepare(`SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = ?`)
      .get(externalId) as any;
    if (!row) return null;
    return { uuid: row.uuid, externalId: row.external_id, createdAt: row.created_at };
  }

  listEntityUuids(limit: number): string[] {
    const rows = this.db
      .prepare(`SELECT DISTINCT uuid FROM mg_entity LIMIT ?`)
      .all(limit) as any[];
    return rows.map((r: any) => r.uuid);
  }

  // ---- Fact metadata ----

  listFactsMetadata(entityUuid: string, filter: FactFilter, limit: number): Fact[] {
    const metaCols = `uuid, content, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at`;
    const { clauses, args: filterArgs } = buildFactFilterClauses(filter);
    let query = `SELECT ${metaCols} FROM mg_entity_fact WHERE entity_id = ?${clauses}`;
    const args: any[] = [entityUuid, ...filterArgs];

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const rows = this.db.prepare(query).all(...args) as any[];
    return rows.map((row: any) => ({
      uuid: row.uuid,
      content: row.content,
      createdAt: row.created_at ?? undefined,
      updatedAt: row.updated_at ?? undefined,
      factType: row.fact_type ?? 'identity',
      temporalStatus: row.temporal_status ?? 'current',
      significance: row.significance ?? 5,
      contentKey: row.content_key ?? '',
      referenceTime: row.reference_time ?? undefined,
      expiresAt: row.expires_at ?? undefined,
      reinforcedAt: row.reinforced_at ?? undefined,
      reinforcedCount: row.reinforced_count ?? 0,
      tag: row.tag ?? '',
      slot: row.slot ?? '',
      confidence: row.confidence ?? 1.0,
      embeddingModel: row.embedding_model ?? '',
      sourceRole: row.source_role ?? '',
      recallCount: row.recall_count ?? 0,
      lastRecalledAt: row.last_recalled_at ?? undefined,
    }));
  }

  // ---- Process ----

  upsertProcess(externalId: string): string {
    const id = randomUUID();
    this.db
      .prepare(`INSERT OR IGNORE INTO mg_process (uuid, external_id) VALUES (?, ?)`)
      .run(id, externalId);
    const row = this.db
      .prepare(`SELECT uuid FROM mg_process WHERE external_id = ?`)
      .get(externalId) as { uuid: string } | undefined;
    return row!.uuid;
  }

  insertProcessAttribute(processUuid: string, key: string, value: string): void {
    const id = randomUUID();
    const now = nowISO();
    this.db
      .prepare(`INSERT INTO mg_process_attribute (uuid, process_id, key, value, created_at) VALUES (?, ?, ?, ?, ?)`)
      .run(id, processUuid, key, value, now);
  }

  // ---- Canonical slots ----

  listCanonicalSlots(): Array<{name: string; embedding?: number[]; createdAt: string}> {
    const rows = this.db
      .prepare(`SELECT name, embedding, created_at FROM mg_slot_canonical ORDER BY name ASC`)
      .all() as any[];
    return rows.map((row: any) => ({
      name: row.name,
      embedding: decodeEmbedding(row.embedding),
      createdAt: row.created_at,
    }));
  }

  insertCanonicalSlot(name: string, embedding: number[]): void {
    const buf = encodeEmbedding(embedding);
    this.db
      .prepare(`INSERT OR IGNORE INTO mg_slot_canonical (name, embedding) VALUES (?, ?)`)
      .run(name, buf);
  }

  findCanonicalSlotByName(name: string): {name: string; embedding?: number[]; createdAt: string} | null {
    const row = this.db
      .prepare(`SELECT name, embedding, created_at FROM mg_slot_canonical WHERE name = ?`)
      .get(name) as any;
    if (!row) return null;
    return {
      name: row.name,
      embedding: decodeEmbedding(row.embedding),
      createdAt: row.created_at,
    };
  }

  // ---- Fact CRUD ----

  insertFact(entityUuid: string, fact: Fact): void {
    if (!fact.uuid) fact.uuid = randomUUID();
    const now = nowISO();
    const embedding = fact.embedding ? encodeEmbedding(fact.embedding) : null;
    this.db
      .prepare(
        `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        fact.uuid,
        entityUuid,
        fact.content,
        embedding,
        fact.createdAt ?? now,
        fact.updatedAt ?? now,
        fact.factType ?? 'identity',
        fact.temporalStatus ?? 'current',
        fact.significance ?? 5,
        fact.contentKey ?? '',
        fact.referenceTime ?? null,
        fact.expiresAt ?? null,
        fact.reinforcedAt ?? null,
        fact.reinforcedCount ?? 0,
        fact.tag ?? '',
        fact.slot ?? '',
        fact.confidence ?? 1.0,
        fact.embeddingModel ?? '',
        fact.sourceRole ?? '',
        fact.recallCount ?? 0,
        fact.lastRecalledAt ?? null
      );
  }

  insertFacts(entityUuid: string, facts: Fact[]): void {
    const tx = this.db.transaction(() => {
      for (const f of facts) {
        this.insertFact(entityUuid, f);
      }
    });
    tx();
  }

  listFacts(entityUuid: string, limit: number): Fact[] {
    const rows = this.db
      .prepare(
        `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?`
      )
      .all(entityUuid, limit) as any[];
    return rows.map(rowToFact);
  }

  listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): Fact[] {
    const { clauses, args: filterArgs } = buildFactFilterClauses(filter);
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?${clauses}`;
    const args: any[] = [entityUuid, ...filterArgs];

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const rows = this.db.prepare(query).all(...args) as any[];
    return rows.map(rowToFact);
  }

  listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): Fact[] {
    const { clauses, args: filterArgs } = buildFactFilterClauses(filter);
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?${clauses}`;
    const args: any[] = [entityUuid, ...filterArgs];

    query += ` ORDER BY significance DESC, created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const rows = this.db.prepare(query).all(...args) as any[];
    return rows.map(rowToFact);
  }

  findFactByKey(entityUuid: string, contentKey: string): Fact | null {
    const row = this.db
      .prepare(
        `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND content_key = ? LIMIT 1`
      )
      .get(entityUuid, contentKey) as any;
    if (!row) return null;
    return rowToFact(row);
  }

  reinforceFact(factUuid: string, newExpiresAt: string | null): void {
    const now = nowISO();
    this.db
      .prepare(
        `UPDATE mg_entity_fact SET reinforced_at = ?, reinforced_count = reinforced_count + 1, expires_at = ?, updated_at = ? WHERE uuid = ?`
      )
      .run(now, newExpiresAt, now, factUuid);
  }

  updateTemporalStatus(factUuid: string, status: string): void {
    const now = nowISO();
    this.db
      .prepare(`UPDATE mg_entity_fact SET temporal_status = ?, updated_at = ? WHERE uuid = ?`)
      .run(status, now, factUuid);
  }

  updateSignificance(factUuid: string, sig: number): void {
    const now = nowISO();
    this.db
      .prepare(`UPDATE mg_entity_fact SET significance = ?, updated_at = ? WHERE uuid = ?`)
      .run(sig, now, factUuid);
  }

  deleteFact(entityUuid: string, factUuid: string): void {
    this.db
      .prepare(`DELETE FROM mg_entity_fact WHERE uuid = ? AND entity_id = ?`)
      .run(factUuid, entityUuid);
  }

  deleteEntityFacts(entityUuid: string): number {
    const result = this.db
      .prepare(`DELETE FROM mg_entity_fact WHERE entity_id = ?`)
      .run(entityUuid);
    return result.changes;
  }

  pruneExpiredFacts(entityUuid: string, now: string): number {
    let total = 0;
    for (;;) {
      let selectQuery = `SELECT uuid FROM mg_entity_fact WHERE expires_at IS NOT NULL AND expires_at < ?`;
      const selectArgs: any[] = [now];
      if (entityUuid) {
        selectQuery = `SELECT uuid FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?`;
        selectArgs.unshift(entityUuid);
      }
      selectQuery += ` LIMIT 1000`;
      const rows = this.db
        .prepare(selectQuery)
        .all(...selectArgs) as any[];
      if (rows.length === 0) break;
      const uuids = rows.map((r: any) => r.uuid);
      const placeholders = uuids.map(() => '?').join(',');
      this.db
        .prepare(`DELETE FROM mg_entity_fact WHERE uuid IN (${placeholders})`)
        .run(...uuids);
      total += uuids.length;
      if (uuids.length < 1000) break;
    }
    return total;
  }

  pruneStaleSummaries(days: number = 90): number {
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    const result = this.db
      .prepare(
        `UPDATE mg_conversation SET summary = '', summary_embedding = NULL WHERE created_at < ? AND summary != ''`
      )
      .run(cutoff);
    return result.changes;
  }

  updateRecallUsage(factUuids: string[]): void {
    const now = nowISO();
    const stmt = this.db.prepare(
      `UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?`
    );
    const tx = this.db.transaction(() => {
      for (const uuid of factUuids) {
        stmt.run(now, uuid);
      }
    });
    tx();
  }

  // ---- Session ----

  ensureSession(
    entityUuid: string,
    processUuid: string,
    timeoutMs: number
  ): { session: FactSession; isNew: boolean } {
    const now = new Date();
    const nowStr = now.toISOString();

    const row = this.db
      .prepare(
        `SELECT uuid, entity_id, process_id, created_at, expires_at, entity_mentions, message_count FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1`
      )
      .get(entityUuid, processUuid, nowStr) as any;

    if (row) {
      const newExpiry = new Date(now.getTime() + timeoutMs).toISOString();
      this.db.prepare(`UPDATE mg_session SET expires_at = ? WHERE uuid = ?`).run(newExpiry, row.uuid);
      let mentions: string[] = [];
      try { mentions = JSON.parse(row.entity_mentions ?? '[]'); } catch { mentions = []; }
      return {
        session: {
          uuid: row.uuid,
          entityId: row.entity_id,
          processId: row.process_id,
          createdAt: row.created_at,
          expiresAt: newExpiry,
          entityMentions: mentions,
          messageCount: row.message_count ?? 0,
        },
        isNew: false,
      };
    }

    const id = randomUUID();
    const expiresAt = new Date(now.getTime() + timeoutMs).toISOString();
    this.db
      .prepare(
        `INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?)`
      )
      .run(id, entityUuid, processUuid, nowStr, expiresAt);

    return {
      session: {
        uuid: id,
        entityId: entityUuid,
        processId: processUuid,
        createdAt: nowStr,
        expiresAt,
        entityMentions: [],
        messageCount: 0,
      },
      isNew: true,
    };
  }

  // ---- Conversation ----

  startConversation(sessionUuid: string, entityUuid: string): string {
    const id = randomUUID();
    const now = nowISO();
    this.db
      .prepare(
        `INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`
      )
      .run(id, sessionUuid, entityUuid, now, now);
    return id;
  }

  activeConversation(sessionUuid: string): FactConversation | null {
    const row = this.db
      .prepare(
        `SELECT uuid, session_id, entity_id, summary, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1`
      )
      .get(sessionUuid) as any;
    if (!row) return null;
    return rowToConversation(row);
  }

  appendMessage(conversationUuid: string, msg: FactMessage): void {
    if (!msg.uuid) msg.uuid = randomUUID();
    const now = nowISO();
    this.db
      .prepare(
        `INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)`
      )
      .run(msg.uuid, conversationUuid, msg.role, msg.content, msg.kind ?? 'text', msg.createdAt ?? now);
    this.db
      .prepare(`UPDATE mg_conversation SET updated_at = ? WHERE uuid = ?`)
      .run(now, conversationUuid);
  }

  readMessages(conversationUuid: string): FactMessage[] {
    const rows = this.db
      .prepare(
        `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at ASC`
      )
      .all(conversationUuid) as any[];
    return rows.map(rowToMessage);
  }

  readRecentMessages(conversationUuid: string, limit: number): FactMessage[] {
    const rows = this.db
      .prepare(
        `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?`
      )
      .all(conversationUuid, limit) as any[];
    return rows.map(rowToMessage).reverse();
  }

  listConversationSummaries(entityUuid: string, limit: number): FactConversation[] {
    let query = `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC`;
    const args: any[] = [entityUuid];
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }
    const rows = this.db.prepare(query).all(...args) as any[];
    return rows.map(rowToConversation);
  }

  updateConversationSummary(
    conversationUuid: string,
    summary: string,
    embedding: number[],
    embeddingModel: string = ''
  ): void {
    const now = nowISO();
    const embBuf = encodeEmbedding(embedding);
    this.db
      .prepare(
        `UPDATE mg_conversation SET summary = ?, summary_embedding = ?, summary_embedding_model = ?, updated_at = ? WHERE uuid = ?`
      )
      .run(summary, embBuf, embeddingModel, now, conversationUuid);
  }

  findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): FactConversation | null {
    const row = this.db
      .prepare(
        `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at
         FROM mg_conversation
         WHERE entity_id = ? AND session_id != ? AND (summary IS NULL OR summary = '')
         ORDER BY created_at DESC LIMIT 1`
      )
      .get(entityUuid, excludeSessionUuid) as any;
    if (!row) return null;
    return rowToConversation(row);
  }

  listUnembeddedFacts(entityUuid: string, limit: number): Fact[] {
    const effectiveLimit = limit > 0 ? limit : 50;
    const rows = this.db
      .prepare(
        `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND embedding IS NULL ORDER BY created_at DESC LIMIT ?`
      )
      .all(entityUuid, effectiveLimit) as any[];
    return rows.map(rowToFact);
  }

  updateFactEmbedding(factUuid: string, embedding: number[], model: string): void {
    const buf = encodeEmbedding(embedding);
    this.db
      .prepare(
        `UPDATE mg_entity_fact SET embedding = ?, embedding_model = ?, updated_at = ? WHERE uuid = ?`
      )
      .run(buf, model, nowISO(), factUuid);
  }

  // ---- Turn Summary ----

  insertTurnSummary(ts: TurnSummary): void {
    if (!ts.uuid) ts.uuid = randomUUID();
    const embedding = ts.summaryEmbedding ? encodeEmbedding(ts.summaryEmbedding) : null;
    this.db
      .prepare(
        `INSERT INTO mg_turn_summary (uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        ts.uuid,
        ts.conversationId,
        ts.entityId,
        ts.startTurn,
        ts.endTurn,
        ts.summary,
        embedding,
        ts.isOverview ? 1 : 0,
        ts.createdAt ?? nowISO()
      );
  }

  listTurnSummaries(conversationId: string): TurnSummary[] {
    const rows = this.db
      .prepare(
        `SELECT uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at FROM mg_turn_summary WHERE conversation_id = ? ORDER BY is_overview ASC, start_turn ASC`
      )
      .all(conversationId) as any[];
    return rows.map(rowToTurnSummary);
  }

  countTurnSummaries(conversationId: string): number {
    const row = this.db
      .prepare(
        `SELECT COUNT(*) as cnt FROM mg_turn_summary WHERE conversation_id = ? AND is_overview = 0`
      )
      .get(conversationId) as any;
    return row?.cnt ?? 0;
  }

  deleteTurnSummaries(conversationId: string, uuids: string[]): void {
    if (uuids.length === 0) return;
    const placeholders = uuids.map(() => '?').join(',');
    this.db
      .prepare(`DELETE FROM mg_turn_summary WHERE conversation_id = ? AND uuid IN (${placeholders})`)
      .run(conversationId, ...uuids);
  }

  // ---- Artifact ----

  insertArtifact(a: Artifact): void {
    if (!a.uuid) a.uuid = randomUUID();
    const embedding = a.descriptionEmbedding ? encodeEmbedding(a.descriptionEmbedding) : null;
    this.db
      .prepare(
        `INSERT INTO mg_artifact (uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        a.uuid,
        a.conversationId,
        a.entityId,
        a.content,
        a.artifactType ?? 'code',
        a.language ?? '',
        a.description ?? '',
        embedding,
        a.supersededBy ?? null,
        a.turnNumber ?? 0,
        a.createdAt ?? nowISO()
      );
  }

  supersedeArtifact(oldUuid: string, newUuid: string): void {
    this.db
      .prepare(`UPDATE mg_artifact SET superseded_by = ? WHERE uuid = ?`)
      .run(newUuid, oldUuid);
  }

  listActiveArtifacts(entityId: string, conversationId: string): Artifact[] {
    const rows = this.db
      .prepare(
        `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = ? AND conversation_id = ? AND superseded_by IS NULL ORDER BY created_at DESC`
      )
      .all(entityId, conversationId) as any[];
    return rows.map(rowToArtifact);
  }

  listActiveArtifactsByEntity(entityId: string): Artifact[] {
    const rows = this.db
      .prepare(
        `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = ? AND superseded_by IS NULL ORDER BY created_at DESC`
      )
      .all(entityId) as any[];
    return rows.map(rowToArtifact);
  }

  // ---- Session metadata ----

  updateSessionMentions(sessionUuid: string, mentions: string[]): void {
    this.db
      .prepare(`UPDATE mg_session SET entity_mentions = ? WHERE uuid = ?`)
      .run(JSON.stringify(mentions), sessionUuid);
  }

  incrementSessionMessageCount(sessionUuid: string): void {
    this.db
      .prepare(`UPDATE mg_session SET message_count = message_count + 1 WHERE uuid = ?`)
      .run(sessionUuid);
  }

  getSessionMentions(sessionUuid: string): string[] {
    const row = this.db
      .prepare(`SELECT entity_mentions FROM mg_session WHERE uuid = ?`)
      .get(sessionUuid) as any;
    if (!row) return [];
    try { return JSON.parse(row.entity_mentions ?? '[]'); } catch { return []; }
  }

  // ---- Lifecycle ----

  close(): void {
    this.db.close();
  }
}

/**
 * Compute a content key from fact content. Matches Go's DefaultContentKey:
 * lowercase, strip punctuation, collapse whitespace, SHA-256, first 16 hex chars.
 */
export function defaultContentKey(content: string): string {
  const lower = content.toLowerCase();
  const cleaned = lower.replace(/[^\p{L}\p{N}\s]/gu, '');
  const normalized = cleaned.split(/\s+/).filter(Boolean).join(' ');
  const hash = createHash('sha256').update(normalized).digest('hex');
  return hash.slice(0, 16);
}
