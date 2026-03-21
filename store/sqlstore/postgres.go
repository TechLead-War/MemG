package sqlstore

import "database/sql"

// NewPostgres returns a Repository configured for PostgreSQL.
func NewPostgres(db *sql.DB) *Repository {
	return New(db, postgresQueries())
}

func postgresQueries() Queries {
	return Queries{
		CreateTables: []string{
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
				uuid       TEXT PRIMARY KEY,
				entity_id  TEXT NOT NULL,
				process_id TEXT NOT NULL DEFAULT '',
				created_at TIMESTAMPTZ NOT NULL,
				expires_at TIMESTAMPTZ NOT NULL
			)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_session_lookup ON mg_session(entity_id, process_id, expires_at)`,
			`CREATE TABLE IF NOT EXISTS mg_conversation (
				uuid              TEXT PRIMARY KEY,
				session_id        TEXT NOT NULL,
				entity_id         TEXT NOT NULL DEFAULT '',
				summary           TEXT NOT NULL DEFAULT '',
				summary_embedding BYTEA,
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
			`CREATE TABLE IF NOT EXISTS mg_schema_version (
				id      INTEGER PRIMARY KEY DEFAULT 1,
				version INTEGER NOT NULL DEFAULT 1
			)`,
		},

		SlotCanonicalList:       `SELECT name, embedding, created_at FROM mg_slot_canonical`,
		SlotCanonicalInsert:     `INSERT INTO mg_slot_canonical (name, embedding, created_at) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING`,
		SlotCanonicalFindByName: `SELECT name, embedding, created_at FROM mg_slot_canonical WHERE name = $1`,

		SchemaRead:  `SELECT version FROM mg_schema_version WHERE id = 1`,
		SchemaWrite: `INSERT INTO mg_schema_version (id, version) VALUES (1, $1) ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version`,

		EntityInsert:   `INSERT INTO mg_entity (uuid, external_id) VALUES ($1, $2) ON CONFLICT (external_id) DO NOTHING`,
		EntitySelectID: `SELECT uuid FROM mg_entity WHERE external_id = $1`,
		EntitySelect:   `SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = $1`,

		FactInsert: `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)`,
		FactSelect: `SELECT uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at FROM mg_entity_fact WHERE entity_id = $1 ORDER BY created_at DESC LIMIT $2`,

		FactFindByKey:    `SELECT uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at FROM mg_entity_fact WHERE entity_id = $1 AND content_key = $2 LIMIT 1`,
		FactUpdateStatus: `UPDATE mg_entity_fact SET temporal_status = $1, updated_at = $2 WHERE uuid = $3`,
		FactReinforce:    `UPDATE mg_entity_fact SET reinforced_at = $1, reinforced_count = reinforced_count + 1, expires_at = $2, updated_at = $3 WHERE uuid = $4`,
		FactPruneExpired: `DELETE FROM mg_entity_fact WHERE entity_id = $1 AND expires_at IS NOT NULL AND expires_at < $2`,
		FactDelete:             `DELETE FROM mg_entity_fact WHERE uuid = $1 AND entity_id = $2`,
		FactDeleteAll:          `DELETE FROM mg_entity_fact WHERE entity_id = $1`,
		FactUpdateSignificance: `UPDATE mg_entity_fact SET significance = $1, updated_at = $2 WHERE uuid = $3`,
		FactUpdateRecallUsage:  `UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = $1 WHERE uuid = $2`,
		FactUpdateEmbedding:    `UPDATE mg_entity_fact SET embedding = $1, embedding_model = $2, updated_at = $3 WHERE uuid = $4`,

		ConvInsert:          `INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)`,
		ConvSelect:          `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE uuid = $1`,
		ConvSelectActive:    `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1`,
		ConvUpdateSummary:   `UPDATE mg_conversation SET summary = $1, summary_embedding = $2, updated_at = $3 WHERE uuid = $4`,
		ConvSelectSummaries:    `SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at FROM mg_conversation WHERE entity_id = $1 AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC LIMIT $2`,
		ConvSelectUnsummarized: `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE entity_id = $1 AND summary = '' AND session_id != $2 ORDER BY created_at DESC LIMIT 1`,
		ConvPruneSummaries:     `UPDATE mg_conversation SET summary = '', summary_embedding = NULL, updated_at = $1 WHERE summary != '' AND created_at < $2`,

		MsgInsert:       `INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES ($1, $2, $3, $4, $5, $6)`,
		MsgSelect:       `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = $1 ORDER BY created_at ASC`,
		MsgSelectRecent: `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = $1 ORDER BY created_at DESC LIMIT $2`,

		SessionInsert: `INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) VALUES ($1, $2, $3, $4, $5)`,
		SessionSelect: `SELECT uuid, entity_id, process_id, created_at, expires_at FROM mg_session WHERE entity_id = $1 AND process_id = $2 AND expires_at > $3 ORDER BY created_at DESC LIMIT 1`,
		SessionSlide:  `UPDATE mg_session SET expires_at = $1 WHERE uuid = $2`,

		ProcessInsert:   `INSERT INTO mg_process (uuid, external_id) VALUES ($1, $2) ON CONFLICT (external_id) DO NOTHING`,
		ProcessSelectID: `SELECT uuid FROM mg_process WHERE external_id = $1`,

		AttrInsert: `INSERT INTO mg_process_attribute (uuid, process_id, key, value, created_at) VALUES ($1, $2, $3, $4, $5)`,
	}
}
