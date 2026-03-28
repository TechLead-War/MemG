/**
 * PostgreSQL-backed repository for the native MemG engine.
 * Uses the `pg` npm package (async API) for networked persistence.
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

// pg is an optional peer dependency — use `any` to avoid compile-time requirement.
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
    isOverview: row.is_overview === true || row.is_overview === 1,
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

const POSTGRES_SCHEMA_DDL: string[] = [
  `CREATE TABLE IF NOT EXISTS mg_entity (
    uuid        TEXT PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`,
  `CREATE TABLE IF NOT EXISTS mg_entity_fact (
    uuid             TEXT PRIMARY KEY,
    entity_id        TEXT NOT NULL REFERENCES mg_entity(uuid),
    content          TEXT NOT NULL,
    embedding        BYTEA,
    created_at       TIMESTAMPTZ NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL,
    fact_type        TEXT NOT NULL DEFAULT 'identity',
    temporal_status  TEXT NOT NULL DEFAULT 'current',
    significance     INTEGER NOT NULL DEFAULT 5,
    content_key      TEXT NOT NULL DEFAULT '',
    reference_time   TIMESTAMPTZ,
    expires_at       TIMESTAMPTZ,
    reinforced_at    TIMESTAMPTZ,
    reinforced_count INTEGER NOT NULL DEFAULT 0,
    tag              TEXT NOT NULL DEFAULT '',
    slot             TEXT NOT NULL DEFAULT '',
    confidence       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    embedding_model  TEXT NOT NULL DEFAULT '',
    source_role      TEXT NOT NULL DEFAULT '',
    recall_count     INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TIMESTAMPTZ
  )`,
  `CREATE INDEX IF NOT EXISTS idx_mg_fact_entity ON mg_entity_fact(entity_id)`,
  `CREATE INDEX IF NOT EXISTS idx_mg_fact_content_key ON mg_entity_fact(entity_id, content_key)`,
  `CREATE INDEX IF NOT EXISTS idx_mg_fact_expires ON mg_entity_fact(entity_id, expires_at)`,
  `CREATE INDEX IF NOT EXISTS idx_mg_fact_slot ON mg_entity_fact(entity_id, slot)`,
  `CREATE TABLE IF NOT EXISTS mg_process (
    uuid        TEXT PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`,
  `CREATE TABLE IF NOT EXISTS mg_process_attribute (
    uuid       TEXT PRIMARY KEY,
    process_id TEXT NOT NULL REFERENCES mg_process(uuid),
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
  )`,
  `CREATE TABLE IF NOT EXISTS mg_session (
    uuid            TEXT PRIMARY KEY,
    entity_id       TEXT NOT NULL,
    process_id      TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    entity_mentions TEXT NOT NULL DEFAULT '[]',
    message_count   INTEGER NOT NULL DEFAULT 0
  )`,
  `CREATE INDEX IF NOT EXISTS idx_mg_session_lookup ON mg_session(entity_id, process_id, expires_at)`,
  `CREATE TABLE IF NOT EXISTS mg_conversation (
    uuid              TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    entity_id         TEXT NOT NULL DEFAULT '',
    summary           TEXT NOT NULL DEFAULT '',
    summary_embedding BYTEA,
    summary_embedding_model TEXT NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_mg_conv_session ON mg_conversation(session_id, created_at)`,
  `CREATE TABLE IF NOT EXISTS mg_message (
    uuid            TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES mg_conversation(uuid),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'text',
    created_at      TIMESTAMPTZ NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_mg_msg_conv ON mg_message(conversation_id, created_at)`,
  `CREATE TABLE IF NOT EXISTS mg_slot_canonical (
    name       TEXT PRIMARY KEY,
    embedding  BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`,
  `CREATE TABLE IF NOT EXISTS mg_turn_summary (
    uuid              TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL,
    entity_id         TEXT NOT NULL,
    start_turn        INTEGER NOT NULL,
    end_turn          INTEGER NOT NULL,
    summary           TEXT NOT NULL,
    summary_embedding BYTEA,
    is_overview       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_mg_turnsummary_conv ON mg_turn_summary(conversation_id, is_overview, created_at)`,
  `CREATE TABLE IF NOT EXISTS mg_artifact (
    uuid                  TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL,
    entity_id             TEXT NOT NULL,
    content               TEXT NOT NULL,
    artifact_type         TEXT NOT NULL DEFAULT 'code',
    language              TEXT NOT NULL DEFAULT '',
    description           TEXT NOT NULL DEFAULT '',
    description_embedding BYTEA,
    superseded_by         TEXT,
    turn_number           INTEGER NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_mg_artifact_entity ON mg_artifact(entity_id, created_at)`,
  `CREATE INDEX IF NOT EXISTS idx_mg_artifact_conv ON mg_artifact(conversation_id, created_at)`,
  `CREATE TABLE IF NOT EXISTS mg_schema_version (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1
  )`,
];

export class PostgresStore implements Store {
  private pool: Pool;
  private initialized: Promise<void>;

  private constructor(pool: Pool) {
    this.pool = pool;
    this.initialized = this.createSchema();
  }

  static async create(connectionString: string): Promise<PostgresStore> {
    const mod = 'pg';
    const pgModule = await import(mod);
    const pg = pgModule.default ?? pgModule;
    const pool = new pg.Pool({ connectionString });
    const store = new PostgresStore(pool);
    await store.ready();
    return store;
  }

  private async createSchema(): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      for (const ddl of POSTGRES_SCHEMA_DDL) {
        await client.query(ddl);
      }
      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
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
      `INSERT INTO mg_entity (uuid, external_id) VALUES ($1, $2) ON CONFLICT (external_id) DO NOTHING`,
      [id, externalId]
    );
    const res = await this.pool.query(
      `SELECT uuid FROM mg_entity WHERE external_id = $1`,
      [externalId]
    );
    return res.rows[0].uuid;
  }

  async lookupEntity(externalId: string): Promise<FactEntity | null> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = $1`,
      [externalId]
    );
    if (res.rows.length === 0) return null;
    const row = res.rows[0];
    return { uuid: row.uuid, externalId: row.external_id, createdAt: row.created_at };
  }

  async listEntityUuids(limit: number): Promise<string[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT DISTINCT uuid FROM mg_entity LIMIT $1`,
      [limit]
    );
    return res.rows.map((r: any) => r.uuid);
  }

  async listFactsMetadata(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]> {
    await this.ready();
    const metaCols = `uuid, content, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at`;
    let query = `SELECT ${metaCols} FROM mg_entity_fact WHERE entity_id = $1`;
    const args: any[] = [entityUuid];
    let paramIdx = 2;

    if (filter.types && filter.types.length > 0) {
      query += ` AND fact_type IN (${filter.types.map(() => `$${paramIdx++}`).join(',')})`;
      args.push(...filter.types);
    }
    if (filter.statuses && filter.statuses.length > 0) {
      query += ` AND temporal_status IN (${filter.statuses.map(() => `$${paramIdx++}`).join(',')})`;
      args.push(...filter.statuses);
    }
    if (filter.tags && filter.tags.length > 0) {
      query += ` AND tag IN (${filter.tags.map(() => `$${paramIdx++}`).join(',')})`;
      args.push(...filter.tags);
    }
    if (filter.minSignificance && filter.minSignificance > 0) {
      query += ` AND significance >= $${paramIdx++}`;
      args.push(filter.minSignificance);
    }
    if (filter.excludeExpired) {
      query += ` AND (expires_at IS NULL OR expires_at > $${paramIdx++})`;
      args.push(nowISO());
    }
    if (filter.slots && filter.slots.length > 0) {
      query += ` AND slot IN (${filter.slots.map(() => `$${paramIdx++}`).join(',')})`;
      args.push(...filter.slots);
    }
    if (filter.minConfidence && filter.minConfidence > 0) {
      query += ` AND confidence >= $${paramIdx++}`;
      args.push(filter.minConfidence);
    }
    if (filter.sourceRoles && filter.sourceRoles.length > 0) {
      query += ` AND source_role IN (${filter.sourceRoles.map(() => `$${paramIdx++}`).join(',')})`;
      args.push(...filter.sourceRoles);
    }
    if (filter.unembeddedOnly) {
      query += ` AND embedding IS NULL`;
    }
    if (filter.maxSignificance && filter.maxSignificance > 0) {
      query += ` AND significance <= $${paramIdx++}`;
      args.push(filter.maxSignificance);
    }
    if (filter.referenceTimeAfter) {
      query += ` AND reference_time IS NOT NULL AND reference_time >= $${paramIdx++}`;
      args.push(filter.referenceTimeAfter);
    }
    if (filter.referenceTimeBefore) {
      query += ` AND reference_time IS NOT NULL AND reference_time <= $${paramIdx++}`;
      args.push(filter.referenceTimeBefore);
    }

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT $${paramIdx++}`;
      args.push(limit);
    }

    const res = await this.pool.query(query, args);
    return res.rows.map((row: any) => ({
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

  async upsertProcess(externalId: string): Promise<string> {
    await this.ready();
    const id = randomUUID();
    await this.pool.query(
      `INSERT INTO mg_process (uuid, external_id) VALUES ($1, $2) ON CONFLICT (external_id) DO NOTHING`,
      [id, externalId]
    );
    const res = await this.pool.query(
      `SELECT uuid FROM mg_process WHERE external_id = $1`,
      [externalId]
    );
    return res.rows[0].uuid;
  }

  async insertProcessAttribute(processUuid: string, key: string, value: string): Promise<void> {
    await this.ready();
    const id = randomUUID();
    const now = nowISO();
    await this.pool.query(
      `INSERT INTO mg_process_attribute (uuid, process_id, key, value, created_at) VALUES ($1, $2, $3, $4, $5)`,
      [id, processUuid, key, value, now]
    );
  }

  async listCanonicalSlots(): Promise<Array<{name: string; embedding?: number[]; createdAt: string}>> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT name, embedding, created_at FROM mg_slot_canonical ORDER BY name ASC`
    );
    return res.rows.map((row: any) => ({
      name: row.name,
      embedding: decodeEmbedding(row.embedding),
      createdAt: row.created_at,
    }));
  }

  async insertCanonicalSlot(name: string, embedding: number[]): Promise<void> {
    await this.ready();
    const buf = encodeEmbedding(embedding);
    await this.pool.query(
      `INSERT INTO mg_slot_canonical (name, embedding) VALUES ($1, $2) ON CONFLICT (name) DO NOTHING`,
      [name, buf]
    );
  }

  async findCanonicalSlotByName(name: string): Promise<{name: string; embedding?: number[]; createdAt: string} | null> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT name, embedding, created_at FROM mg_slot_canonical WHERE name = $1`,
      [name]
    );
    if (res.rows.length === 0) return null;
    const row = res.rows[0];
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
      `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)`,
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
      ]
    );
  }

  async insertFacts(entityUuid: string, facts: Fact[]): Promise<void> {
    await this.ready();
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      for (const fact of facts) {
        if (!fact.uuid) fact.uuid = randomUUID();
        const now = nowISO();
        const embedding = fact.embedding ? encodeEmbedding(fact.embedding) : null;
        await client.query(
          `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)`,
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
          ]
        );
      }
      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }
  }

  async listFacts(entityUuid: string, limit: number): Promise<Fact[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = $1 ORDER BY created_at DESC LIMIT $2`,
      [entityUuid, limit]
    );
    return res.rows.map(rowToFact);
  }

  async listFactsFiltered(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]> {
    await this.ready();
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = $1`;
    const args: any[] = [entityUuid];
    let paramIdx = 2;

    if (filter.types && filter.types.length > 0) {
      const placeholders = filter.types.map(() => `$${paramIdx++}`).join(',');
      query += ` AND fact_type IN (${placeholders})`;
      args.push(...filter.types);
    }
    if (filter.statuses && filter.statuses.length > 0) {
      const placeholders = filter.statuses.map(() => `$${paramIdx++}`).join(',');
      query += ` AND temporal_status IN (${placeholders})`;
      args.push(...filter.statuses);
    }
    if (filter.tags && filter.tags.length > 0) {
      const placeholders = filter.tags.map(() => `$${paramIdx++}`).join(',');
      query += ` AND tag IN (${placeholders})`;
      args.push(...filter.tags);
    }
    if (filter.minSignificance && filter.minSignificance > 0) {
      query += ` AND significance >= $${paramIdx++}`;
      args.push(filter.minSignificance);
    }
    if (filter.excludeExpired) {
      query += ` AND (expires_at IS NULL OR expires_at > $${paramIdx++})`;
      args.push(nowISO());
    }
    if (filter.slots && filter.slots.length > 0) {
      const placeholders = filter.slots.map(() => `$${paramIdx++}`).join(',');
      query += ` AND slot IN (${placeholders})`;
      args.push(...filter.slots);
    }
    if (filter.minConfidence && filter.minConfidence > 0) {
      query += ` AND confidence >= $${paramIdx++}`;
      args.push(filter.minConfidence);
    }
    if (filter.sourceRoles && filter.sourceRoles.length > 0) {
      const placeholders = filter.sourceRoles.map(() => `$${paramIdx++}`).join(',');
      query += ` AND source_role IN (${placeholders})`;
      args.push(...filter.sourceRoles);
    }
    if (filter.unembeddedOnly) {
      query += ` AND embedding IS NULL`;
    }
    if (filter.maxSignificance && filter.maxSignificance > 0) {
      query += ` AND significance <= $${paramIdx++}`;
      args.push(filter.maxSignificance);
    }
    if (filter.referenceTimeAfter) {
      query += ` AND reference_time IS NOT NULL AND reference_time >= $${paramIdx++}`;
      args.push(filter.referenceTimeAfter);
    }
    if (filter.referenceTimeBefore) {
      query += ` AND reference_time IS NOT NULL AND reference_time <= $${paramIdx++}`;
      args.push(filter.referenceTimeBefore);
    }

    query += ` ORDER BY created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT $${paramIdx++}`;
      args.push(limit);
    }

    const res = await this.pool.query(query, args);
    return res.rows.map(rowToFact);
  }

  async listFactsForRecall(entityUuid: string, filter: FactFilter, limit: number): Promise<Fact[]> {
    await this.ready();
    let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = $1`;
    const args: any[] = [entityUuid];
    let paramIdx = 2;

    if (filter.types && filter.types.length > 0) {
      const placeholders = filter.types.map(() => `$${paramIdx++}`).join(',');
      query += ` AND fact_type IN (${placeholders})`;
      args.push(...filter.types);
    }
    if (filter.statuses && filter.statuses.length > 0) {
      const placeholders = filter.statuses.map(() => `$${paramIdx++}`).join(',');
      query += ` AND temporal_status IN (${placeholders})`;
      args.push(...filter.statuses);
    }
    if (filter.tags && filter.tags.length > 0) {
      const placeholders = filter.tags.map(() => `$${paramIdx++}`).join(',');
      query += ` AND tag IN (${placeholders})`;
      args.push(...filter.tags);
    }
    if (filter.minSignificance && filter.minSignificance > 0) {
      query += ` AND significance >= $${paramIdx++}`;
      args.push(filter.minSignificance);
    }
    if (filter.excludeExpired) {
      query += ` AND (expires_at IS NULL OR expires_at > $${paramIdx++})`;
      args.push(nowISO());
    }
    if (filter.slots && filter.slots.length > 0) {
      const placeholders = filter.slots.map(() => `$${paramIdx++}`).join(',');
      query += ` AND slot IN (${placeholders})`;
      args.push(...filter.slots);
    }
    if (filter.minConfidence && filter.minConfidence > 0) {
      query += ` AND confidence >= $${paramIdx++}`;
      args.push(filter.minConfidence);
    }
    if (filter.sourceRoles && filter.sourceRoles.length > 0) {
      const placeholders = filter.sourceRoles.map(() => `$${paramIdx++}`).join(',');
      query += ` AND source_role IN (${placeholders})`;
      args.push(...filter.sourceRoles);
    }
    if (filter.unembeddedOnly) {
      query += ` AND embedding IS NULL`;
    }
    if (filter.maxSignificance && filter.maxSignificance > 0) {
      query += ` AND significance <= $${paramIdx++}`;
      args.push(filter.maxSignificance);
    }
    if (filter.referenceTimeAfter) {
      query += ` AND reference_time IS NOT NULL AND reference_time >= $${paramIdx++}`;
      args.push(filter.referenceTimeAfter);
    }
    if (filter.referenceTimeBefore) {
      query += ` AND reference_time IS NOT NULL AND reference_time <= $${paramIdx++}`;
      args.push(filter.referenceTimeBefore);
    }

    query += ` ORDER BY significance DESC, created_at DESC`;
    if (limit > 0) {
      query += ` LIMIT $${paramIdx++}`;
      args.push(limit);
    }

    const res = await this.pool.query(query, args);
    return res.rows.map(rowToFact);
  }

  async findFactByKey(entityUuid: string, contentKey: string): Promise<Fact | null> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = $1 AND content_key = $2 LIMIT 1`,
      [entityUuid, contentKey]
    );
    if (res.rows.length === 0) return null;
    return rowToFact(res.rows[0]);
  }

  // ---- Fact lifecycle ----

  async reinforceFact(factUuid: string, newExpiresAt: string | null): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET reinforced_at = $1, reinforced_count = reinforced_count + 1, expires_at = $2, updated_at = $3 WHERE uuid = $4`,
      [now, newExpiresAt, now, factUuid]
    );
  }

  async updateTemporalStatus(factUuid: string, status: string): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET temporal_status = $1, updated_at = $2 WHERE uuid = $3`,
      [status, now, factUuid]
    );
  }

  async updateSignificance(factUuid: string, sig: number): Promise<void> {
    await this.ready();
    const now = nowISO();
    await this.pool.query(
      `UPDATE mg_entity_fact SET significance = $1, updated_at = $2 WHERE uuid = $3`,
      [sig, now, factUuid]
    );
  }

  async deleteFact(entityUuid: string, factUuid: string): Promise<void> {
    await this.ready();
    await this.pool.query(
      `DELETE FROM mg_entity_fact WHERE uuid = $1 AND entity_id = $2`,
      [factUuid, entityUuid]
    );
  }

  async deleteEntityFacts(entityUuid: string): Promise<number> {
    await this.ready();
    const res = await this.pool.query(
      `DELETE FROM mg_entity_fact WHERE entity_id = $1`,
      [entityUuid]
    );
    return res.rowCount ?? 0;
  }

  async pruneExpiredFacts(entityUuid: string, now: string): Promise<number> {
    await this.ready();
    let total = 0;
    for (;;) {
      let selectQuery: string;
      let selectArgs: any[];
      if (entityUuid) {
        selectQuery = `SELECT uuid FROM mg_entity_fact WHERE entity_id = $1 AND expires_at IS NOT NULL AND expires_at < $2 LIMIT 1000`;
        selectArgs = [entityUuid, now];
      } else {
        selectQuery = `SELECT uuid FROM mg_entity_fact WHERE expires_at IS NOT NULL AND expires_at < $1 LIMIT 1000`;
        selectArgs = [now];
      }
      const sel = await this.pool.query(selectQuery, selectArgs);
      if (sel.rows.length === 0) break;
      const uuids = sel.rows.map((r: any) => r.uuid);
      const placeholders = uuids.map((_: any, i: number) => `$${i + 1}`).join(',');
      await this.pool.query(`DELETE FROM mg_entity_fact WHERE uuid IN (${placeholders})`, uuids);
      total += uuids.length;
      if (uuids.length < 1000) break;
    }
    return total;
  }

  async pruneStaleSummaries(days: number = 90): Promise<number> {
    await this.ready();
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    const res = await this.pool.query(
      `UPDATE mg_conversation SET summary = '', summary_embedding = NULL WHERE created_at < $1 AND summary != ''`,
      [cutoff]
    );
    return res.rowCount ?? 0;
  }

  async updateRecallUsage(factUuids: string[]): Promise<void> {
    await this.ready();
    const now = nowISO();
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      for (const uuid of factUuids) {
        await client.query(
          `UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = $1 WHERE uuid = $2`,
          [now, uuid]
        );
      }
      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
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

    const res = await this.pool.query(
      `SELECT uuid, entity_id, process_id, created_at, expires_at, entity_mentions, message_count FROM mg_session WHERE entity_id = $1 AND process_id = $2 AND expires_at > $3 ORDER BY created_at DESC LIMIT 1`,
      [entityUuid, processUuid, nowStr]
    );

    if (res.rows.length > 0) {
      const row = res.rows[0];
      const newExpiry = new Date(now.getTime() + timeoutMs).toISOString();
      await this.pool.query(
        `UPDATE mg_session SET expires_at = $1 WHERE uuid = $2`,
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
      `INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) VALUES ($1, $2, $3, $4, $5)`,
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
      `INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)`,
      [id, sessionUuid, entityUuid, now, now]
    );
    return id;
  }

  async activeConversation(sessionUuid: string): Promise<FactConversation | null> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, session_id, entity_id, summary, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1`,
      [sessionUuid]
    );
    if (res.rows.length === 0) return null;
    return rowToConversation(res.rows[0]);
  }

  async appendMessage(conversationUuid: string, msg: FactMessage): Promise<void> {
    await this.ready();
    if (!msg.uuid) msg.uuid = randomUUID();
    const now = nowISO();
    await this.pool.query(
      `INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES ($1, $2, $3, $4, $5, $6)`,
      [msg.uuid, conversationUuid, msg.role, msg.content, msg.kind ?? 'text', msg.createdAt ?? now]
    );
    await this.pool.query(
      `UPDATE mg_conversation SET updated_at = $1 WHERE uuid = $2`,
      [now, conversationUuid]
    );
  }

  async readMessages(conversationUuid: string): Promise<FactMessage[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = $1 ORDER BY created_at ASC`,
      [conversationUuid]
    );
    return res.rows.map(rowToMessage);
  }

  async readRecentMessages(conversationUuid: string, limit: number): Promise<FactMessage[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = $1 ORDER BY created_at DESC LIMIT $2`,
      [conversationUuid, limit]
    );
    return res.rows.map(rowToMessage).reverse();
  }

  async listConversationSummaries(entityUuid: string, limit: number): Promise<FactConversation[]> {
    await this.ready();
    let query = `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE entity_id = $1 AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC`;
    const args: any[] = [entityUuid];
    if (limit > 0) {
      query += ` LIMIT $2`;
      args.push(limit);
    }
    const res = await this.pool.query(query, args);
    return res.rows.map(rowToConversation);
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
      `UPDATE mg_conversation SET summary = $1, summary_embedding = $2, summary_embedding_model = $3, updated_at = $4 WHERE uuid = $5`,
      [summary, embBuf, embeddingModel, now, conversationUuid]
    );
  }

  // ---- Turn Summary ----

  async insertTurnSummary(ts: TurnSummary): Promise<void> {
    await this.ready();
    if (!ts.uuid) ts.uuid = randomUUID();
    const embedding = ts.summaryEmbedding ? encodeEmbedding(ts.summaryEmbedding) : null;
    await this.pool.query(
      `INSERT INTO mg_turn_summary (uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
      [
        ts.uuid,
        ts.conversationId,
        ts.entityId,
        ts.startTurn,
        ts.endTurn,
        ts.summary,
        embedding,
        ts.isOverview,
        ts.createdAt ?? nowISO(),
      ]
    );
  }

  async listTurnSummaries(conversationId: string): Promise<TurnSummary[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at FROM mg_turn_summary WHERE conversation_id = $1 ORDER BY is_overview ASC, start_turn ASC`,
      [conversationId]
    );
    return res.rows.map(rowToTurnSummary);
  }

  async countTurnSummaries(conversationId: string): Promise<number> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT COUNT(*) as cnt FROM mg_turn_summary WHERE conversation_id = $1 AND is_overview = FALSE`,
      [conversationId]
    );
    return parseInt(res.rows[0]?.cnt ?? '0', 10);
  }

  async deleteTurnSummaries(conversationId: string, uuids: string[]): Promise<void> {
    await this.ready();
    if (uuids.length === 0) return;
    const placeholders = uuids.map((_, i) => `$${i + 2}`).join(',');
    await this.pool.query(
      `DELETE FROM mg_turn_summary WHERE conversation_id = $1 AND uuid IN (${placeholders})`,
      [conversationId, ...uuids]
    );
  }

  // ---- Artifact ----

  async insertArtifact(a: Artifact): Promise<void> {
    await this.ready();
    if (!a.uuid) a.uuid = randomUUID();
    const embedding = a.descriptionEmbedding ? encodeEmbedding(a.descriptionEmbedding) : null;
    await this.pool.query(
      `INSERT INTO mg_artifact (uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
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
      `UPDATE mg_artifact SET superseded_by = $1 WHERE uuid = $2`,
      [newUuid, oldUuid]
    );
  }

  async listActiveArtifacts(entityId: string, conversationId: string): Promise<Artifact[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = $1 AND conversation_id = $2 AND superseded_by IS NULL ORDER BY created_at DESC`,
      [entityId, conversationId]
    );
    return res.rows.map(rowToArtifact);
  }

  async listActiveArtifactsByEntity(entityId: string): Promise<Artifact[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = $1 AND superseded_by IS NULL ORDER BY created_at DESC`,
      [entityId]
    );
    return res.rows.map(rowToArtifact);
  }

  // ---- Session metadata ----

  async updateSessionMentions(sessionUuid: string, mentions: string[]): Promise<void> {
    await this.ready();
    await this.pool.query(
      `UPDATE mg_session SET entity_mentions = $1 WHERE uuid = $2`,
      [JSON.stringify(mentions), sessionUuid]
    );
  }

  async incrementSessionMessageCount(sessionUuid: string): Promise<void> {
    await this.ready();
    await this.pool.query(
      `UPDATE mg_session SET message_count = message_count + 1 WHERE uuid = $1`,
      [sessionUuid]
    );
  }

  async getSessionMentions(sessionUuid: string): Promise<string[]> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT entity_mentions FROM mg_session WHERE uuid = $1`,
      [sessionUuid]
    );
    if (res.rows.length === 0) return [];
    try { return JSON.parse(res.rows[0].entity_mentions ?? '[]'); } catch { return []; }
  }

  // ---- Lifecycle ----

  async findUnsummarizedConversation(entityUuid: string, excludeSessionUuid: string): Promise<FactConversation | null> {
    await this.ready();
    const res = await this.pool.query(
      `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at
       FROM mg_conversation
       WHERE entity_id = $1 AND session_id != $2 AND (summary IS NULL OR summary = '')
       ORDER BY created_at DESC LIMIT 1`,
      [entityUuid, excludeSessionUuid]
    );
    if (res.rows.length === 0) return null;
    return rowToConversation(res.rows[0]);
  }

  async listUnembeddedFacts(entityUuid: string, limit: number): Promise<Fact[]> {
    await this.ready();
    const effectiveLimit = limit > 0 ? limit : 50;
    const res = await this.pool.query(
      `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = $1 AND embedding IS NULL ORDER BY created_at DESC LIMIT $2`,
      [entityUuid, effectiveLimit]
    );
    return res.rows.map(rowToFact);
  }

  async updateFactEmbedding(factUuid: string, embedding: number[], model: string): Promise<void> {
    await this.ready();
    const buf = encodeEmbedding(embedding);
    await this.pool.query(
      `UPDATE mg_entity_fact SET embedding = $1, embedding_model = $2, updated_at = $3 WHERE uuid = $4`,
      [buf, model, nowISO(), factUuid]
    );
  }

  async close(): Promise<void> {
    await this.pool.end();
  }
}
