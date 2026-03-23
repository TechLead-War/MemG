/**
 * SQLite-backed repository for the native MemG engine.
 * Uses better-sqlite3 (synchronous API) for fast, in-process persistence.
 */

import { createHash, randomUUID } from 'crypto';
import type {
  Fact,
  FactConversation,
  FactEntity,
  FactFilter,
  FactMessage,
  FactSession,
} from './types';
import { SCHEMA_DDL } from './schema';

type Database = import('better-sqlite3').Database;

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
  updateRecallUsage(factUuids: string[]): MaybeAsync<void>;

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
  updateConversationSummary(conversationUuid: string, summary: string, embedding: number[]): MaybeAsync<void>;

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

const FACT_COLUMNS = `uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at`;

export class MemGStore implements Store {
  private db: Database;

  constructor(dbPath: string) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const BetterSqlite3 = require('better-sqlite3') as typeof import('better-sqlite3');
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
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?`;
    const args: any[] = [entityUuid];

    if (filter.types && filter.types.length > 0) {
      query += ` AND fact_type IN (${filter.types.map(() => '?').join(',')})`;
      args.push(...filter.types);
    }
    if (filter.statuses && filter.statuses.length > 0) {
      query += ` AND temporal_status IN (${filter.statuses.map(() => '?').join(',')})`;
      args.push(...filter.statuses);
    }
    if (filter.tags && filter.tags.length > 0) {
      query += ` AND tag IN (${filter.tags.map(() => '?').join(',')})`;
      args.push(...filter.tags);
    }
    if (filter.minSignificance && filter.minSignificance > 0) {
      query += ` AND significance >= ?`;
      args.push(filter.minSignificance);
    }
    if (filter.excludeExpired) {
      query += ` AND (expires_at IS NULL OR expires_at > ?)`;
      args.push(nowISO());
    }
    if (filter.slots && filter.slots.length > 0) {
      query += ` AND slot IN (${filter.slots.map(() => '?').join(',')})`;
      args.push(...filter.slots);
    }
    if (filter.minConfidence && filter.minConfidence > 0) {
      query += ` AND confidence >= ?`;
      args.push(filter.minConfidence);
    }
    if (filter.sourceRoles && filter.sourceRoles.length > 0) {
      query += ` AND source_role IN (${filter.sourceRoles.map(() => '?').join(',')})`;
      args.push(...filter.sourceRoles);
    }

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const rows = this.db.prepare(query).all(...args) as any[];
    return rows.map(rowToFact);
  }

  listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): Fact[] {
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?`;
    const args: any[] = [entityUuid];

    if (filter.types && filter.types.length > 0) {
      query += ` AND fact_type IN (${filter.types.map(() => '?').join(',')})`;
      args.push(...filter.types);
    }
    if (filter.statuses && filter.statuses.length > 0) {
      query += ` AND temporal_status IN (${filter.statuses.map(() => '?').join(',')})`;
      args.push(...filter.statuses);
    }
    if (filter.tags && filter.tags.length > 0) {
      query += ` AND tag IN (${filter.tags.map(() => '?').join(',')})`;
      args.push(...filter.tags);
    }
    if (filter.minSignificance && filter.minSignificance > 0) {
      query += ` AND significance >= ?`;
      args.push(filter.minSignificance);
    }
    if (filter.excludeExpired) {
      query += ` AND (expires_at IS NULL OR expires_at > ?)`;
      args.push(nowISO());
    }
    if (filter.slots && filter.slots.length > 0) {
      query += ` AND slot IN (${filter.slots.map(() => '?').join(',')})`;
      args.push(...filter.slots);
    }
    if (filter.minConfidence && filter.minConfidence > 0) {
      query += ` AND confidence >= ?`;
      args.push(filter.minConfidence);
    }
    if (filter.sourceRoles && filter.sourceRoles.length > 0) {
      query += ` AND source_role IN (${filter.sourceRoles.map(() => '?').join(',')})`;
      args.push(...filter.sourceRoles);
    }

    // No ORDER BY created_at — prevents newest-first bias in recall.
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
    const result = this.db
      .prepare(
        `DELETE FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?`
      )
      .run(entityUuid, now);
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
        `SELECT uuid, entity_id, process_id, created_at, expires_at FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1`
      )
      .get(entityUuid, processUuid, nowStr) as any;

    if (row) {
      const newExpiry = new Date(now.getTime() + timeoutMs).toISOString();
      this.db.prepare(`UPDATE mg_session SET expires_at = ? WHERE uuid = ?`).run(newExpiry, row.uuid);
      return {
        session: {
          uuid: row.uuid,
          entityId: row.entity_id,
          processId: row.process_id,
          createdAt: row.created_at,
          expiresAt: newExpiry,
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
        `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1`
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
    let query = `SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC`;
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
    embedding: number[]
  ): void {
    const now = nowISO();
    const embBuf = encodeEmbedding(embedding);
    this.db
      .prepare(
        `UPDATE mg_conversation SET summary = ?, summary_embedding = ?, updated_at = ? WHERE uuid = ?`
      )
      .run(summary, embBuf, now, conversationUuid);
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
