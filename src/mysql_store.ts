/**
 * MySQL-backed repository for the native MemG engine.
 * Uses the `mysql2/promise` npm package (async API) for networked persistence.
 */

import { randomUUID } from 'crypto';
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
import type { Store } from './store.js';

// mysql2 is an optional peer dependency — use `any` to avoid compile-time requirement.
type Pool = any;

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
    emotionalWeight: row.emotional_weight ?? undefined,
    emotionalValence: row.emotional_valence ?? undefined,
    verbatim: row.verbatim ?? undefined,
    startedAt: row.started_at ?? undefined,
    threadStatus: row.thread_status ?? undefined,
    engagementScore: row.engagement_score ?? 0,
    pinned: row.pinned === 1,
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

const FACT_COLUMNS = `uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at, emotional_weight, emotional_valence, verbatim, started_at, thread_status, engagement_score, pinned`;

const MYSQL_SCHEMA_DDL: string[] = [
  `CREATE TABLE IF NOT EXISTS mg_entity (
    uuid        VARCHAR(36) PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL UNIQUE,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
  )`,
  `CREATE TABLE IF NOT EXISTS mg_entity_fact (
    uuid             VARCHAR(36) PRIMARY KEY,
    entity_id        VARCHAR(36) NOT NULL,
    content          TEXT NOT NULL,
    embedding        LONGBLOB,
    created_at       DATETIME NOT NULL,
    updated_at       DATETIME NOT NULL,
    fact_type        VARCHAR(50) NOT NULL DEFAULT 'identity',
    temporal_status  VARCHAR(50) NOT NULL DEFAULT 'current',
    significance     INTEGER NOT NULL DEFAULT 5,
    content_key      VARCHAR(255) NOT NULL DEFAULT '',
    reference_time   DATETIME,
    expires_at       DATETIME,
    reinforced_at    DATETIME,
    reinforced_count INTEGER NOT NULL DEFAULT 0,
    tag              VARCHAR(100) NOT NULL DEFAULT '',
    slot             VARCHAR(100) NOT NULL DEFAULT '',
    confidence       DOUBLE NOT NULL DEFAULT 1.0,
    embedding_model  VARCHAR(255) NOT NULL DEFAULT '',
    source_role      VARCHAR(50) NOT NULL DEFAULT '',
    recall_count     INTEGER NOT NULL DEFAULT 0,
    last_recalled_at DATETIME,
    emotional_weight DOUBLE,
    emotional_valence VARCHAR(50),
    verbatim TEXT,
    started_at VARCHAR(50),
    thread_status VARCHAR(20),
    engagement_score DOUBLE NOT NULL DEFAULT 0,
    pinned TINYINT NOT NULL DEFAULT 0,
    FOREIGN KEY (entity_id) REFERENCES mg_entity(uuid),
    INDEX idx_mg_fact_entity (entity_id),
    INDEX idx_mg_fact_content_key (entity_id, content_key),
    INDEX idx_mg_fact_expires (entity_id, expires_at),
    INDEX idx_mg_fact_slot (entity_id, slot),
    INDEX idx_mg_fact_thread (entity_id, thread_status),
    INDEX idx_mg_fact_pinned (entity_id, pinned),
    INDEX idx_mg_fact_emotional (entity_id, emotional_valence)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_process (
    uuid        VARCHAR(36) PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL UNIQUE,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
  )`,
  `CREATE TABLE IF NOT EXISTS mg_process_attribute (
    uuid       VARCHAR(36) PRIMARY KEY,
    process_id VARCHAR(36) NOT NULL,
    \`key\`      VARCHAR(255) NOT NULL,
    value      TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (process_id) REFERENCES mg_process(uuid)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_session (
    uuid            VARCHAR(36) PRIMARY KEY,
    entity_id       VARCHAR(36) NOT NULL,
    process_id      VARCHAR(36) NOT NULL DEFAULT '',
    created_at      DATETIME NOT NULL,
    expires_at      DATETIME NOT NULL,
    entity_mentions TEXT NOT NULL,
    message_count   INTEGER NOT NULL DEFAULT 0,
    INDEX idx_mg_session_lookup (entity_id, process_id, expires_at)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_conversation (
    uuid              VARCHAR(36) PRIMARY KEY,
    session_id        VARCHAR(36) NOT NULL,
    entity_id         VARCHAR(36) NOT NULL DEFAULT '',
    summary           TEXT NOT NULL,
    summary_embedding LONGBLOB,
    summary_embedding_model VARCHAR(255) NOT NULL DEFAULT '',
    created_at        DATETIME NOT NULL,
    updated_at        DATETIME NOT NULL,
    INDEX idx_mg_conv_session (session_id, created_at)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_message (
    uuid            VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    role            VARCHAR(50) NOT NULL,
    content         TEXT NOT NULL,
    kind            VARCHAR(50) NOT NULL DEFAULT 'text',
    created_at      DATETIME NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES mg_conversation(uuid),
    INDEX idx_mg_msg_conv (conversation_id, created_at)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_slot_canonical (
    name       VARCHAR(255) PRIMARY KEY,
    embedding  LONGBLOB NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
  )`,
  `CREATE TABLE IF NOT EXISTS mg_turn_summary (
    uuid              VARCHAR(36) PRIMARY KEY,
    conversation_id   VARCHAR(36) NOT NULL,
    entity_id         VARCHAR(36) NOT NULL,
    start_turn        INTEGER NOT NULL,
    end_turn          INTEGER NOT NULL,
    summary           TEXT NOT NULL,
    summary_embedding LONGBLOB,
    is_overview       TINYINT NOT NULL DEFAULT 0,
    created_at        DATETIME NOT NULL,
    INDEX idx_mg_turnsummary_conv (conversation_id, is_overview, created_at)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_artifact (
    uuid                  VARCHAR(36) PRIMARY KEY,
    conversation_id       VARCHAR(36) NOT NULL,
    entity_id             VARCHAR(36) NOT NULL,
    content               TEXT NOT NULL,
    artifact_type         VARCHAR(50) NOT NULL DEFAULT 'code',
    language              VARCHAR(100) NOT NULL DEFAULT '',
    description           TEXT NOT NULL,
    description_embedding LONGBLOB,
    superseded_by         VARCHAR(36),
    turn_number           INTEGER NOT NULL DEFAULT 0,
    created_at            DATETIME NOT NULL,
    INDEX idx_mg_artifact_entity (entity_id, created_at),
    INDEX idx_mg_artifact_conv (conversation_id, created_at)
  )`,
  `CREATE TABLE IF NOT EXISTS mg_schema_version (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1
  )`,
];

const MYSQL_ALTER_MIGRATIONS: string[] = [
  `ALTER TABLE mg_entity_fact ADD COLUMN emotional_weight DOUBLE`,
  `ALTER TABLE mg_entity_fact ADD COLUMN emotional_valence VARCHAR(50)`,
  `ALTER TABLE mg_entity_fact ADD COLUMN verbatim TEXT`,
  `ALTER TABLE mg_entity_fact ADD COLUMN started_at VARCHAR(50)`,
  `ALTER TABLE mg_entity_fact ADD COLUMN thread_status VARCHAR(20)`,
  `ALTER TABLE mg_entity_fact ADD COLUMN engagement_score DOUBLE NOT NULL DEFAULT 0`,
  `ALTER TABLE mg_entity_fact ADD COLUMN pinned TINYINT NOT NULL DEFAULT 0`,
  `ALTER TABLE mg_entity_fact ADD INDEX idx_mg_fact_thread (entity_id, thread_status)`,
  `ALTER TABLE mg_entity_fact ADD INDEX idx_mg_fact_pinned (entity_id, pinned)`,
  `ALTER TABLE mg_entity_fact ADD INDEX idx_mg_fact_emotional (entity_id, emotional_valence)`,
];

export class MySQLStore implements Store {
  private pool: Pool;
  private initialized: Promise<void>;

  private constructor(pool: Pool) {
    this.pool = pool;
    this.initialized = this.createSchema();
  }

  static async create(connectionString: string): Promise<MySQLStore> {
    const mod = 'mysql2/promise';
    const mysqlModule = await import(mod);
    const mysql = mysqlModule.default ?? mysqlModule;
    const pool = mysql.createPool(connectionString);
    const store = new MySQLStore(pool);
    await store.ready();
    return store;
  }

  private async createSchema(): Promise<void> {
    const conn = await this.pool.getConnection();
    try {
      for (const ddl of MYSQL_SCHEMA_DDL) {
        await conn.query(ddl);
      }
      for (const alter of MYSQL_ALTER_MIGRATIONS) {
        try {
          await conn.query(alter);
        } catch (_) {
          // Column or index may already exist — ignore.
        }
      }
    } finally {
      conn.release();
    }
  }

  private async ready(): Promise<void> {
    await this.initialized;
  }

  // ---- Entity ----

  async upsertEntity(externalId: string): Promise<string> {
    await this.ready();
    const id = randomUUID();
    await this.pool.query(
      `INSERT IGNORE INTO mg_entity (uuid, external_id) VALUES (?, ?)`,
      [id, externalId]
    );
    const [rows] = await this.pool.query(
      `SELECT uuid FROM mg_entity WHERE external_id = ?`,
      [externalId]
    ) as any;
    return rows[0].uuid;
  }

  async lookupEntity(externalId: string): Promise<FactEntity | null> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = ?`,
      [externalId]
    ) as any;
    if (rows.length === 0) return null;
    const row = rows[0];
    return { uuid: row.uuid, externalId: row.external_id, createdAt: row.created_at };
  }

  async listEntityUuids(limit: number): Promise<string[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT DISTINCT uuid FROM mg_entity LIMIT ?`,
      [limit]
    ) as any;
    return rows.map((r: any) => r.uuid);
  }

  async listFactsMetadata(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]> {
    await this.ready();
    const metaCols = `uuid, content, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at, emotional_weight, emotional_valence, verbatim, started_at, thread_status, engagement_score, pinned`;
    let query = `SELECT ${metaCols} FROM mg_entity_fact WHERE entity_id = ?`;
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
    if (filter.unembeddedOnly) {
      query += ` AND embedding IS NULL`;
    }
    if (filter.maxSignificance && filter.maxSignificance > 0) {
      query += ` AND significance <= ?`;
      args.push(filter.maxSignificance);
    }
    if (filter.referenceTimeAfter) {
      query += ` AND reference_time IS NOT NULL AND reference_time >= ?`;
      args.push(filter.referenceTimeAfter);
    }
    if (filter.referenceTimeBefore) {
      query += ` AND reference_time IS NOT NULL AND reference_time <= ?`;
      args.push(filter.referenceTimeBefore);
    }
    if (filter.threadStatuses && filter.threadStatuses.length > 0) {
      query += ` AND thread_status IN (${filter.threadStatuses.map(() => '?').join(',')})`;
      args.push(...filter.threadStatuses);
    }
    if (filter.pinned !== undefined) {
      query += ` AND pinned = ?`;
      args.push(filter.pinned ? 1 : 0);
    }
    if (filter.emotionalValences && filter.emotionalValences.length > 0) {
      query += ` AND emotional_valence IN (${filter.emotionalValences.map(() => '?').join(',')})`;
      args.push(...filter.emotionalValences);
    }

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const [rows] = await this.pool.query(query, args) as any;
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
      emotionalWeight: row.emotional_weight ?? undefined,
      emotionalValence: row.emotional_valence ?? undefined,
      verbatim: row.verbatim ?? undefined,
      startedAt: row.started_at ?? undefined,
      threadStatus: row.thread_status ?? undefined,
      engagementScore: row.engagement_score ?? 0,
      pinned: row.pinned === 1,
    }));
  }

  async upsertProcess(externalId: string): Promise<string> {
    await this.ready();
    const id = randomUUID();
    await this.pool.query(
      `INSERT IGNORE INTO mg_process (uuid, external_id) VALUES (?, ?)`,
      [id, externalId]
    );
    const [rows] = await this.pool.query(
      `SELECT uuid FROM mg_process WHERE external_id = ?`,
      [externalId]
    ) as any;
    return rows[0].uuid;
  }

  async insertProcessAttribute(processUuid: string, key: string, value: string): Promise<void> {
    await this.ready();
    const id = randomUUID();
    const now = nowISO();
    await this.pool.query(
      'INSERT INTO mg_process_attribute (uuid, process_id, `key`, value, created_at) VALUES (?, ?, ?, ?, ?)',
      [id, processUuid, key, value, now]
    );
  }

  async listCanonicalSlots(): Promise<Array<{name: string; embedding?: number[]; createdAt: string}>> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT name, embedding, created_at FROM mg_slot_canonical ORDER BY name ASC`
    ) as any;
    return rows.map((row: any) => ({
      name: row.name,
      embedding: decodeEmbedding(row.embedding),
      createdAt: row.created_at,
    }));
  }

  async insertCanonicalSlot(name: string, embedding: number[]): Promise<void> {
    await this.ready();
    const buf = encodeEmbedding(embedding);
    await this.pool.query(
      `INSERT IGNORE INTO mg_slot_canonical (name, embedding) VALUES (?, ?)`,
      [name, buf]
    );
  }

  async findCanonicalSlotByName(name: string): Promise<{name: string; embedding?: number[]; createdAt: string} | null> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT name, embedding, created_at FROM mg_slot_canonical WHERE name = ?`,
      [name]
    ) as any;
    if (rows.length === 0) return null;
    const row = rows[0];
    return {
      name: row.name,
      embedding: decodeEmbedding(row.embedding),
      createdAt: row.created_at,
    };
  }

  // ---- Fact CRUD ----

  async insertFact(entityUuid: string, fact: Fact): Promise<void> {
    await this.ready();
    if (!fact.uuid) fact.uuid = randomUUID();
    const now = nowISO();
    const embedding = fact.embedding ? encodeEmbedding(fact.embedding) : null;
    await this.pool.query(
      `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at, emotional_weight, emotional_valence, verbatim, started_at, thread_status, engagement_score, pinned) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
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
        fact.lastRecalledAt ?? null,
        fact.emotionalWeight ?? null,
        fact.emotionalValence ?? null,
        fact.verbatim ?? null,
        fact.startedAt ?? null,
        fact.threadStatus ?? null,
        fact.engagementScore ?? 0,
        fact.pinned ? 1 : 0,
      ]
    );
  }

  async insertFacts(entityUuid: string, facts: Fact[]): Promise<void> {
    await this.ready();
    const conn = await this.pool.getConnection();
    try {
      await conn.beginTransaction();
      for (const fact of facts) {
        if (!fact.uuid) fact.uuid = randomUUID();
        const now = nowISO();
        const embedding = fact.embedding ? encodeEmbedding(fact.embedding) : null;
        await conn.query(
          `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at, emotional_weight, emotional_valence, verbatim, started_at, thread_status, engagement_score, pinned) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
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
            fact.lastRecalledAt ?? null,
            fact.emotionalWeight ?? null,
            fact.emotionalValence ?? null,
            fact.verbatim ?? null,
            fact.startedAt ?? null,
            fact.threadStatus ?? null,
            fact.engagementScore ?? 0,
            fact.pinned ? 1 : 0,
          ]
        );
      }
      await conn.commit();
    } catch (err) {
      await conn.rollback();
      throw err;
    } finally {
      conn.release();
    }
  }

  async listFacts(entityUuid: string, limit: number): Promise<Fact[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?`,
      [entityUuid, limit]
    ) as any;
    return rows.map(rowToFact);
  }

  async listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]> {
    await this.ready();
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
    if (filter.unembeddedOnly) {
      query += ` AND embedding IS NULL`;
    }
    if (filter.maxSignificance && filter.maxSignificance > 0) {
      query += ` AND significance <= ?`;
      args.push(filter.maxSignificance);
    }
    if (filter.referenceTimeAfter) {
      query += ` AND reference_time IS NOT NULL AND reference_time >= ?`;
      args.push(filter.referenceTimeAfter);
    }
    if (filter.referenceTimeBefore) {
      query += ` AND reference_time IS NOT NULL AND reference_time <= ?`;
      args.push(filter.referenceTimeBefore);
    }
    if (filter.threadStatuses && filter.threadStatuses.length > 0) {
      query += ` AND thread_status IN (${filter.threadStatuses.map(() => '?').join(',')})`;
      args.push(...filter.threadStatuses);
    }
    if (filter.pinned !== undefined) {
      query += ` AND pinned = ?`;
      args.push(filter.pinned ? 1 : 0);
    }
    if (filter.emotionalValences && filter.emotionalValences.length > 0) {
      query += ` AND emotional_valence IN (${filter.emotionalValences.map(() => '?').join(',')})`;
      args.push(...filter.emotionalValences);
    }

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const [rows] = await this.pool.query(query, args) as any;
    return rows.map(rowToFact);
  }

  async listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]> {
    await this.ready();
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
    if (filter.unembeddedOnly) {
      query += ` AND embedding IS NULL`;
    }
    if (filter.maxSignificance && filter.maxSignificance > 0) {
      query += ` AND significance <= ?`;
      args.push(filter.maxSignificance);
    }
    if (filter.referenceTimeAfter) {
      query += ` AND reference_time IS NOT NULL AND reference_time >= ?`;
      args.push(filter.referenceTimeAfter);
    }
    if (filter.referenceTimeBefore) {
      query += ` AND reference_time IS NOT NULL AND reference_time <= ?`;
      args.push(filter.referenceTimeBefore);
    }
    if (filter.threadStatuses && filter.threadStatuses.length > 0) {
      query += ` AND thread_status IN (${filter.threadStatuses.map(() => '?').join(',')})`;
      args.push(...filter.threadStatuses);
    }
    if (filter.pinned !== undefined) {
      query += ` AND pinned = ?`;
      args.push(filter.pinned ? 1 : 0);
    }
    if (filter.emotionalValences && filter.emotionalValences.length > 0) {
      query += ` AND emotional_valence IN (${filter.emotionalValences.map(() => '?').join(',')})`;
      args.push(...filter.emotionalValences);
    }

    query += ` ORDER BY significance DESC, created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }

    const [rows] = await this.pool.query(query, args) as any;
    return rows.map(rowToFact);
  }

  async findFactByKey(entityUuid: string, contentKey: string): Promise<Fact | null> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND content_key = ? LIMIT 1`,
      [entityUuid, contentKey]
    ) as any;
    if (rows.length === 0) return null;
    return rowToFact(rows[0]);
  }

  // ---- Fact lifecycle ----

  async reinforceFact(factUuid: string, newExpiresAt: string | null): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET reinforced_at = ?, reinforced_count = reinforced_count + 1, expires_at = ?, updated_at = ? WHERE uuid = ?`,
      [now, newExpiresAt, now, factUuid]
    );
  }

  async updateTemporalStatus(factUuid: string, status: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET temporal_status = ?, updated_at = ? WHERE uuid = ?`,
      [status, now, factUuid]
    );
  }

  async updateSignificance(factUuid: string, sig: number): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET significance = ?, updated_at = ? WHERE uuid = ?`,
      [sig, now, factUuid]
    );
  }

  async deleteFact(entityUuid: string, factUuid: string): Promise<void> {
    await this.ready();
    await this.pool.query(
      `DELETE FROM mg_entity_fact WHERE uuid = ? AND entity_id = ?`,
      [factUuid, entityUuid]
    );
  }

  async deleteEntityFacts(entityUuid: string): Promise<number> {
    await this.ready();
    const [result] = await this.pool.query(
      `DELETE FROM mg_entity_fact WHERE entity_id = ?`,
      [entityUuid]
    ) as any;
    return result.affectedRows ?? 0;
  }

  async pruneExpiredFacts(entityUuid: string, now: string): Promise<number> {
    await this.ready();
    let total = 0;
    for (;;) {
      let selectQuery: string;
      let selectArgs: any[];
      if (entityUuid) {
        selectQuery = `SELECT uuid FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ? LIMIT 1000`;
        selectArgs = [entityUuid, now];
      } else {
        selectQuery = `SELECT uuid FROM mg_entity_fact WHERE expires_at IS NOT NULL AND expires_at < ? LIMIT 1000`;
        selectArgs = [now];
      }
      const [rows] = await this.pool.query(selectQuery, selectArgs) as any;
      if (rows.length === 0) break;
      const uuids = rows.map((r: any) => r.uuid);
      const placeholders = uuids.map(() => '?').join(',');
      await this.pool.query(`DELETE FROM mg_entity_fact WHERE uuid IN (${placeholders})`, uuids);
      total += uuids.length;
      if (uuids.length < 1000) break;
    }
    return total;
  }

  async pruneStaleSummaries(days: number = 90): Promise<number> {
    await this.ready();
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    const [result] = await this.pool.query(
      `UPDATE mg_conversation SET summary = '', summary_embedding = NULL WHERE created_at < ? AND summary != ''`,
      [cutoff]
    ) as any;
    return result.affectedRows ?? 0;
  }

  async updateRecallUsage(factUuids: string[]): Promise<void> {
    await this.ready();
    const now = nowISO();
    const conn = await this.pool.getConnection();
    try {
      await conn.beginTransaction();
      for (const uuid of factUuids) {
        await conn.query(
          `UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?`,
          [now, uuid]
        );
      }
      await conn.commit();
    } catch (err) {
      await conn.rollback();
      throw err;
    } finally {
      conn.release();
    }
  }

  // ---- Session ----

  async ensureSession(
    entityUuid: string,
    processUuid: string,
    timeoutMs: number
  ): Promise<{ session: FactSession; isNew: boolean }> {
    await this.ready();
    const now = new Date();
    const nowStr = now.toISOString();

    const [rows] = await this.pool.query(
      `SELECT uuid, entity_id, process_id, created_at, expires_at, entity_mentions, message_count FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1`,
      [entityUuid, processUuid, nowStr]
    ) as any;

    if (rows.length > 0) {
      const row = rows[0];
      const newExpiry = new Date(now.getTime() + timeoutMs).toISOString();
      await this.pool.query(
        `UPDATE mg_session SET expires_at = ? WHERE uuid = ?`,
        [newExpiry, row.uuid]
      );
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
    await this.pool.query(
      `INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at, entity_mentions) VALUES (?, ?, ?, ?, ?, '[]')`,
      [id, entityUuid, processUuid, nowStr, expiresAt]
    );

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

  async startConversation(sessionUuid: string, entityUuid: string): Promise<string> {
    await this.ready();
    const id = randomUUID();
    const now = nowISO();
    await this.pool.query(
      `INSERT INTO mg_conversation (uuid, session_id, entity_id, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)`,
      [id, sessionUuid, entityUuid, '', now, now]
    );
    return id;
  }

  async activeConversation(sessionUuid: string): Promise<FactConversation | null> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, session_id, entity_id, summary, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1`,
      [sessionUuid]
    ) as any;
    if (rows.length === 0) return null;
    return rowToConversation(rows[0]);
  }

  async appendMessage(conversationUuid: string, msg: FactMessage): Promise<void> {
    await this.ready();
    if (!msg.uuid) msg.uuid = randomUUID();
    const now = nowISO();
    await this.pool.query(
      `INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
      [msg.uuid, conversationUuid, msg.role, msg.content, msg.kind ?? 'text', msg.createdAt ?? now]
    );
    await this.pool.query(
      `UPDATE mg_conversation SET updated_at = ? WHERE uuid = ?`,
      [now, conversationUuid]
    );
  }

  async readMessages(conversationUuid: string): Promise<FactMessage[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at ASC`,
      [conversationUuid]
    ) as any;
    return rows.map(rowToMessage);
  }

  async readRecentMessages(conversationUuid: string, limit: number): Promise<FactMessage[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?`,
      [conversationUuid, limit]
    ) as any;
    return rows.map(rowToMessage).reverse();
  }

  async listConversationSummaries(entityUuid: string, limit: number): Promise<FactConversation[]> {
    await this.ready();
    let query = `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC`;
    const args: any[] = [entityUuid];
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }
    const [rows] = await this.pool.query(query, args) as any;
    return rows.map(rowToConversation);
  }

  async updateConversationSummary(
    conversationUuid: string,
    summary: string,
    embedding: number[],
    embeddingModel: string = ''
  ): Promise<void> {
    await this.ready();
    const now = nowISO();
    const embBuf = encodeEmbedding(embedding);
    await this.pool.query(
      `UPDATE mg_conversation SET summary = ?, summary_embedding = ?, summary_embedding_model = ?, updated_at = ? WHERE uuid = ?`,
      [summary, embBuf, embeddingModel, now, conversationUuid]
    );
  }

  // ---- Turn Summary ----

  async insertTurnSummary(ts: TurnSummary): Promise<void> {
    await this.ready();
    if (!ts.uuid) ts.uuid = randomUUID();
    const embedding = ts.summaryEmbedding ? encodeEmbedding(ts.summaryEmbedding) : null;
    await this.pool.query(
      `INSERT INTO mg_turn_summary (uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        ts.uuid,
        ts.conversationId,
        ts.entityId,
        ts.startTurn,
        ts.endTurn,
        ts.summary,
        embedding,
        ts.isOverview ? 1 : 0,
        ts.createdAt ?? nowISO(),
      ]
    );
  }

  async listTurnSummaries(conversationId: string): Promise<TurnSummary[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at FROM mg_turn_summary WHERE conversation_id = ? ORDER BY is_overview ASC, start_turn ASC`,
      [conversationId]
    ) as any;
    return rows.map(rowToTurnSummary);
  }

  async countTurnSummaries(conversationId: string): Promise<number> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT COUNT(*) as cnt FROM mg_turn_summary WHERE conversation_id = ? AND is_overview = 0`,
      [conversationId]
    ) as any;
    return rows[0]?.cnt ?? 0;
  }

  async deleteTurnSummaries(conversationId: string, uuids: string[]): Promise<void> {
    await this.ready();
    if (uuids.length === 0) return;
    const placeholders = uuids.map(() => '?').join(',');
    await this.pool.query(
      `DELETE FROM mg_turn_summary WHERE conversation_id = ? AND uuid IN (${placeholders})`,
      [conversationId, ...uuids]
    );
  }

  // ---- Artifact ----

  async insertArtifact(a: Artifact): Promise<void> {
    await this.ready();
    if (!a.uuid) a.uuid = randomUUID();
    const embedding = a.descriptionEmbedding ? encodeEmbedding(a.descriptionEmbedding) : null;
    await this.pool.query(
      `INSERT INTO mg_artifact (uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
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
        a.createdAt ?? nowISO(),
      ]
    );
  }

  async supersedeArtifact(oldUuid: string, newUuid: string): Promise<void> {
    await this.ready();
    await this.pool.query(
      `UPDATE mg_artifact SET superseded_by = ? WHERE uuid = ?`,
      [newUuid, oldUuid]
    );
  }

  async listActiveArtifacts(entityId: string, conversationId: string): Promise<Artifact[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = ? AND conversation_id = ? AND superseded_by IS NULL ORDER BY created_at DESC`,
      [entityId, conversationId]
    ) as any;
    return rows.map(rowToArtifact);
  }

  async listActiveArtifactsByEntity(entityId: string): Promise<Artifact[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = ? AND superseded_by IS NULL ORDER BY created_at DESC`,
      [entityId]
    ) as any;
    return rows.map(rowToArtifact);
  }

  // ---- Session metadata ----

  async updateSessionMentions(sessionUuid: string, mentions: string[]): Promise<void> {
    await this.ready();
    await this.pool.query(
      `UPDATE mg_session SET entity_mentions = ? WHERE uuid = ?`,
      [JSON.stringify(mentions), sessionUuid]
    );
  }

  async incrementSessionMessageCount(sessionUuid: string): Promise<void> {
    await this.ready();
    await this.pool.query(
      `UPDATE mg_session SET message_count = message_count + 1 WHERE uuid = ?`,
      [sessionUuid]
    );
  }

  async getSessionMentions(sessionUuid: string): Promise<string[]> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT entity_mentions FROM mg_session WHERE uuid = ?`,
      [sessionUuid]
    ) as any;
    if (rows.length === 0) return [];
    try { return JSON.parse(rows[0].entity_mentions ?? '[]'); } catch { return []; }
  }

  // ---- Lifecycle ----

  async findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): Promise<FactConversation | null> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at
       FROM mg_conversation
       WHERE entity_id = ? AND session_id != ? AND (summary IS NULL OR summary = '')
       ORDER BY created_at DESC LIMIT 1`,
      [entityUuid, excludeSessionUuid]
    );
    const arr = rows as any[];
    if (arr.length === 0) return null;
    return rowToConversation(arr[0]);
  }

  async listUnembeddedFacts(entityUuid: string, limit: number): Promise<Fact[]> {
    await this.ready();
    const effectiveLimit = limit > 0 ? limit : 50;
    const [rows] = await this.pool.query(
      `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND embedding IS NULL ORDER BY created_at DESC LIMIT ?`,
      [entityUuid, effectiveLimit]
    );
    return (rows as any[]).map(rowToFact);
  }

  async updateFactEmbedding(factUuid: string, embedding: number[], model: string): Promise<void> {
    await this.ready();
    const buf = encodeEmbedding(embedding);
    await this.pool.query(
      `UPDATE mg_entity_fact SET embedding = ?, embedding_model = ?, updated_at = ? WHERE uuid = ?`,
      [buf, model, nowISO(), factUuid]
    );
  }

  // ---- Extended fact operations ----

  async pinFact(factUuid: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET pinned = 1, significance = 10, expires_at = NULL, updated_at = ? WHERE uuid = ?`,
      [now, factUuid]
    );
  }

  async unpinFact(factUuid: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET pinned = 0, updated_at = ? WHERE uuid = ?`,
      [now, factUuid]
    );
  }

  async confirmFact(factUuid: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET confidence = 1.0, reinforced_at = ?, reinforced_count = reinforced_count + 1, updated_at = ? WHERE uuid = ?`,
      [now, now, factUuid]
    );
  }

  async updateThreadStatus(factUuid: string, status: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET thread_status = ?, updated_at = ? WHERE uuid = ?`,
      [status, now, factUuid]
    );
  }

  async listOpenThreads(entityUuid: string, limit: number): Promise<Fact[]> {
    await this.ready();
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND thread_status = 'open' ORDER BY created_at DESC`;
    const args: any[] = [entityUuid];
    if (limit > 0) {
      query += ` LIMIT ?`;
      args.push(limit);
    }
    const [rows] = await this.pool.query(query, args) as any;
    return rows.map(rowToFact);
  }

  async updateEmotionalWeight(factUuid: string, weight: number, valence: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET emotional_weight = ?, emotional_valence = ?, updated_at = ? WHERE uuid = ?`,
      [weight, valence, now, factUuid]
    );
  }

  async updateEngagementScore(factUuid: string, score: number): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET engagement_score = ?, updated_at = ? WHERE uuid = ?`,
      [score, now, factUuid]
    );
  }

  async countEntityFacts(entityUuid: string): Promise<number> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT COUNT(*) as cnt FROM mg_entity_fact WHERE entity_id = ?`,
      [entityUuid]
    ) as any;
    return rows[0]?.cnt ?? 0;
  }

  async getEntityFirstFactDate(entityUuid: string): Promise<string | null> {
    await this.ready();
    const [rows] = await this.pool.query(
      `SELECT MIN(created_at) as first_date FROM mg_entity_fact WHERE entity_id = ?`,
      [entityUuid]
    ) as any;
    return rows[0]?.first_date ?? null;
  }

  async close(): Promise<void> {
    await this.pool.end();
  }
}
