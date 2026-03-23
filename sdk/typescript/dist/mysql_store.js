"use strict";
/**
 * MySQL-backed repository for the native MemG engine.
 * Uses the `mysql2/promise` npm package (async API) for networked persistence.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MySQLStore = void 0;
const crypto_1 = require("crypto");
function nowISO() {
    return new Date().toISOString();
}
function encodeEmbedding(vec) {
    const buf = Buffer.alloc(vec.length * 4);
    for (let i = 0; i < vec.length; i++) {
        buf.writeFloatLE(vec[i], i * 4);
    }
    return buf;
}
function decodeEmbedding(raw) {
    if (!raw || raw.length === 0)
        return undefined;
    if (raw.length % 4 !== 0 || raw[0] === 0x5b) {
        try {
            return JSON.parse(raw.toString());
        }
        catch {
            return undefined;
        }
    }
    const count = raw.length / 4;
    const out = new Array(count);
    for (let i = 0; i < count; i++) {
        out[i] = raw.readFloatLE(i * 4);
    }
    return out;
}
function rowToFact(row) {
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
function rowToConversation(row) {
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
function rowToMessage(row) {
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
const MYSQL_SCHEMA_DDL = [
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
    FOREIGN KEY (entity_id) REFERENCES mg_entity(uuid),
    INDEX idx_mg_fact_entity (entity_id),
    INDEX idx_mg_fact_content_key (entity_id, content_key),
    INDEX idx_mg_fact_expires (entity_id, expires_at),
    INDEX idx_mg_fact_slot (entity_id, slot)
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
    uuid       VARCHAR(36) PRIMARY KEY,
    entity_id  VARCHAR(36) NOT NULL,
    process_id VARCHAR(36) NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL,
    INDEX idx_mg_session_lookup (entity_id, process_id, expires_at)
  )`,
    `CREATE TABLE IF NOT EXISTS mg_conversation (
    uuid              VARCHAR(36) PRIMARY KEY,
    session_id        VARCHAR(36) NOT NULL,
    entity_id         VARCHAR(36) NOT NULL DEFAULT '',
    summary           TEXT NOT NULL,
    summary_embedding LONGBLOB,
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
    `CREATE TABLE IF NOT EXISTS mg_schema_version (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1
  )`,
];
class MySQLStore {
    constructor(connectionString) {
        const mysql = require('mysql2/promise');
        this.pool = mysql.createPool(connectionString);
        this.initialized = this.createSchema();
    }
    static async create(connectionString) {
        const store = new MySQLStore(connectionString);
        await store.ready();
        return store;
    }
    async createSchema() {
        const conn = await this.pool.getConnection();
        try {
            for (const ddl of MYSQL_SCHEMA_DDL) {
                await conn.query(ddl);
            }
        }
        finally {
            conn.release();
        }
    }
    async ready() {
        await this.initialized;
    }
    // ---- Entity ----
    async upsertEntity(externalId) {
        await this.ready();
        const id = (0, crypto_1.randomUUID)();
        await this.pool.query(`INSERT IGNORE INTO mg_entity (uuid, external_id) VALUES (?, ?)`, [id, externalId]);
        const [rows] = await this.pool.query(`SELECT uuid FROM mg_entity WHERE external_id = ?`, [externalId]);
        return rows[0].uuid;
    }
    async lookupEntity(externalId) {
        await this.ready();
        const [rows] = await this.pool.query(`SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = ?`, [externalId]);
        if (rows.length === 0)
            return null;
        const row = rows[0];
        return { uuid: row.uuid, externalId: row.external_id, createdAt: row.created_at };
    }
    // ---- Fact CRUD ----
    async insertFact(entityUuid, fact) {
        await this.ready();
        if (!fact.uuid)
            fact.uuid = (0, crypto_1.randomUUID)();
        const now = nowISO();
        const embedding = fact.embedding ? encodeEmbedding(fact.embedding) : null;
        await this.pool.query(`INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
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
        ]);
    }
    async insertFacts(entityUuid, facts) {
        await this.ready();
        const conn = await this.pool.getConnection();
        try {
            await conn.beginTransaction();
            for (const fact of facts) {
                if (!fact.uuid)
                    fact.uuid = (0, crypto_1.randomUUID)();
                const now = nowISO();
                const embedding = fact.embedding ? encodeEmbedding(fact.embedding) : null;
                await conn.query(`INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
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
                ]);
            }
            await conn.commit();
        }
        catch (err) {
            await conn.rollback();
            throw err;
        }
        finally {
            conn.release();
        }
    }
    async listFacts(entityUuid, limit) {
        await this.ready();
        const [rows] = await this.pool.query(`SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?`, [entityUuid, limit]);
        return rows.map(rowToFact);
    }
    async listFactsFiltered(entityUuid, filter, limit) {
        await this.ready();
        let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?`;
        const args = [entityUuid];
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
        const [rows] = await this.pool.query(query, args);
        return rows.map(rowToFact);
    }
    async listFactsForRecall(entityUuid, filter, limit) {
        await this.ready();
        let query = `SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?`;
        const args = [entityUuid];
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
        if (limit > 0) {
            query += ` LIMIT ?`;
            args.push(limit);
        }
        const [rows] = await this.pool.query(query, args);
        return rows.map(rowToFact);
    }
    async findFactByKey(entityUuid, contentKey) {
        await this.ready();
        const [rows] = await this.pool.query(`SELECT ${FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND content_key = ? LIMIT 1`, [entityUuid, contentKey]);
        if (rows.length === 0)
            return null;
        return rowToFact(rows[0]);
    }
    // ---- Fact lifecycle ----
    async reinforceFact(factUuid, newExpiresAt) {
        await this.ready();
        const now = nowISO();
        await this.pool.query(`UPDATE mg_entity_fact SET reinforced_at = ?, reinforced_count = reinforced_count + 1, expires_at = ?, updated_at = ? WHERE uuid = ?`, [now, newExpiresAt, now, factUuid]);
    }
    async updateTemporalStatus(factUuid, status) {
        await this.ready();
        const now = nowISO();
        await this.pool.query(`UPDATE mg_entity_fact SET temporal_status = ?, updated_at = ? WHERE uuid = ?`, [status, now, factUuid]);
    }
    async updateSignificance(factUuid, sig) {
        await this.ready();
        const now = nowISO();
        await this.pool.query(`UPDATE mg_entity_fact SET significance = ?, updated_at = ? WHERE uuid = ?`, [sig, now, factUuid]);
    }
    async deleteFact(entityUuid, factUuid) {
        await this.ready();
        await this.pool.query(`DELETE FROM mg_entity_fact WHERE uuid = ? AND entity_id = ?`, [factUuid, entityUuid]);
    }
    async deleteEntityFacts(entityUuid) {
        await this.ready();
        const [result] = await this.pool.query(`DELETE FROM mg_entity_fact WHERE entity_id = ?`, [entityUuid]);
        return result.affectedRows ?? 0;
    }
    async pruneExpiredFacts(entityUuid, now) {
        await this.ready();
        const [result] = await this.pool.query(`DELETE FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?`, [entityUuid, now]);
        return result.affectedRows ?? 0;
    }
    async updateRecallUsage(factUuids) {
        await this.ready();
        const now = nowISO();
        const conn = await this.pool.getConnection();
        try {
            await conn.beginTransaction();
            for (const uuid of factUuids) {
                await conn.query(`UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?`, [now, uuid]);
            }
            await conn.commit();
        }
        catch (err) {
            await conn.rollback();
            throw err;
        }
        finally {
            conn.release();
        }
    }
    // ---- Session ----
    async ensureSession(entityUuid, processUuid, timeoutMs) {
        await this.ready();
        const now = new Date();
        const nowStr = now.toISOString();
        const [rows] = await this.pool.query(`SELECT uuid, entity_id, process_id, created_at, expires_at FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1`, [entityUuid, processUuid, nowStr]);
        if (rows.length > 0) {
            const row = rows[0];
            const newExpiry = new Date(now.getTime() + timeoutMs).toISOString();
            await this.pool.query(`UPDATE mg_session SET expires_at = ? WHERE uuid = ?`, [newExpiry, row.uuid]);
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
        const id = (0, crypto_1.randomUUID)();
        const expiresAt = new Date(now.getTime() + timeoutMs).toISOString();
        await this.pool.query(`INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?)`, [id, entityUuid, processUuid, nowStr, expiresAt]);
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
    async startConversation(sessionUuid, entityUuid) {
        await this.ready();
        const id = (0, crypto_1.randomUUID)();
        const now = nowISO();
        await this.pool.query(`INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [id, sessionUuid, entityUuid, now, now]);
        return id;
    }
    async activeConversation(sessionUuid) {
        await this.ready();
        const [rows] = await this.pool.query(`SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1`, [sessionUuid]);
        if (rows.length === 0)
            return null;
        return rowToConversation(rows[0]);
    }
    async appendMessage(conversationUuid, msg) {
        await this.ready();
        if (!msg.uuid)
            msg.uuid = (0, crypto_1.randomUUID)();
        const now = nowISO();
        await this.pool.query(`INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)`, [msg.uuid, conversationUuid, msg.role, msg.content, msg.kind ?? 'text', msg.createdAt ?? now]);
    }
    async readMessages(conversationUuid) {
        await this.ready();
        const [rows] = await this.pool.query(`SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at ASC`, [conversationUuid]);
        return rows.map(rowToMessage);
    }
    async readRecentMessages(conversationUuid, limit) {
        await this.ready();
        const [rows] = await this.pool.query(`SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?`, [conversationUuid, limit]);
        return rows.map(rowToMessage).reverse();
    }
    async listConversationSummaries(entityUuid, limit) {
        await this.ready();
        let query = `SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC`;
        const args = [entityUuid];
        if (limit > 0) {
            query += ` LIMIT ?`;
            args.push(limit);
        }
        const [rows] = await this.pool.query(query, args);
        return rows.map(rowToConversation);
    }
    async updateConversationSummary(conversationUuid, summary, embedding) {
        await this.ready();
        const now = nowISO();
        const embBuf = encodeEmbedding(embedding);
        await this.pool.query(`UPDATE mg_conversation SET summary = ?, summary_embedding = ?, updated_at = ? WHERE uuid = ?`, [summary, embBuf, now, conversationUuid]);
    }
    // ---- Lifecycle ----
    async close() {
        await this.pool.end();
    }
}
exports.MySQLStore = MySQLStore;
//# sourceMappingURL=mysql_store.js.map