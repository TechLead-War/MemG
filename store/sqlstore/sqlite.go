package sqlstore

import "database/sql"

// NewSQLite returns a Repository configured for SQLite.
func NewSQLite(db *sql.DB) *Repository {
	return New(db, sqliteQueries())
}

func sqliteQueries() Queries {
	return Queries{
		CreateTables: []string{
			`CREATE TABLE IF NOT EXISTS mg_entity (
				uuid        TEXT PRIMARY KEY,
				external_id TEXT NOT NULL UNIQUE,
				created_at  TEXT NOT NULL DEFAULT (datetime('now'))
			)`,
			`CREATE TABLE IF NOT EXISTS mg_entity_fact (
				uuid             TEXT PRIMARY KEY,
				entity_id        TEXT NOT NULL REFERENCES mg_entity(uuid),
				content          TEXT NOT NULL,
				embedding        BLOB,
				created_at       TEXT NOT NULL,
				updated_at       TEXT NOT NULL,
				fact_type        TEXT NOT NULL DEFAULT 'identity',
				temporal_status  TEXT NOT NULL DEFAULT 'current',
				significance     INTEGER NOT NULL DEFAULT 5,
				content_key      TEXT NOT NULL DEFAULT '',
				reference_time   TEXT,
				expires_at       TEXT,
				reinforced_at    TEXT,
				reinforced_count INTEGER NOT NULL DEFAULT 0,
				tag              TEXT NOT NULL DEFAULT '',
				slot             TEXT NOT NULL DEFAULT '',
				confidence       REAL NOT NULL DEFAULT 1.0,
				embedding_model  TEXT NOT NULL DEFAULT '',
				source_role      TEXT NOT NULL DEFAULT '',
				recall_count     INTEGER NOT NULL DEFAULT 0,
				last_recalled_at TEXT
			)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_fact_entity ON mg_entity_fact(entity_id)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_fact_content_key ON mg_entity_fact(entity_id, content_key)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_fact_expires ON mg_entity_fact(entity_id, expires_at)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_fact_slot ON mg_entity_fact(entity_id, slot)`,
			`CREATE TABLE IF NOT EXISTS mg_process (
				uuid        TEXT PRIMARY KEY,
				external_id TEXT NOT NULL UNIQUE,
				created_at  TEXT NOT NULL DEFAULT (datetime('now'))
			)`,
			`CREATE TABLE IF NOT EXISTS mg_process_attribute (
				uuid       TEXT PRIMARY KEY,
				process_id TEXT NOT NULL REFERENCES mg_process(uuid),
				key        TEXT NOT NULL,
				value      TEXT NOT NULL,
				created_at TEXT NOT NULL
			)`,
			`CREATE TABLE IF NOT EXISTS mg_session (
				uuid            TEXT PRIMARY KEY,
				entity_id       TEXT NOT NULL,
				process_id      TEXT NOT NULL DEFAULT '',
				created_at      TEXT NOT NULL,
				expires_at      TEXT NOT NULL,
				entity_mentions TEXT NOT NULL DEFAULT '[]',
				message_count   INTEGER NOT NULL DEFAULT 0
			)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_session_lookup ON mg_session(entity_id, process_id, expires_at)`,
			`CREATE TABLE IF NOT EXISTS mg_conversation (
				uuid              TEXT PRIMARY KEY,
				session_id        TEXT NOT NULL,
				entity_id         TEXT NOT NULL DEFAULT '',
				summary           TEXT NOT NULL DEFAULT '',
				summary_embedding BLOB,
				summary_embedding_model TEXT NOT NULL DEFAULT '',
				created_at        TEXT NOT NULL,
				updated_at        TEXT NOT NULL
			)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_conv_session ON mg_conversation(session_id, created_at)`,
			`CREATE TABLE IF NOT EXISTS mg_message (
				uuid            TEXT PRIMARY KEY,
				conversation_id TEXT NOT NULL REFERENCES mg_conversation(uuid),
				role            TEXT NOT NULL,
				content         TEXT NOT NULL,
				kind            TEXT NOT NULL DEFAULT 'text',
				created_at      TEXT NOT NULL
			)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_msg_conv ON mg_message(conversation_id, created_at)`,
			`CREATE TABLE IF NOT EXISTS mg_slot_canonical (
				name       TEXT PRIMARY KEY,
				embedding  BLOB NOT NULL,
				created_at TEXT NOT NULL DEFAULT (datetime('now'))
			)`,
			`CREATE TABLE IF NOT EXISTS mg_turn_summary (
				uuid              TEXT PRIMARY KEY,
				conversation_id   TEXT NOT NULL,
				entity_id         TEXT NOT NULL,
				start_turn        INTEGER NOT NULL,
				end_turn          INTEGER NOT NULL,
				summary           TEXT NOT NULL,
				summary_embedding BLOB,
				is_overview       INTEGER NOT NULL DEFAULT 0,
				created_at        TEXT NOT NULL
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
				description_embedding BLOB,
				superseded_by         TEXT,
				turn_number           INTEGER NOT NULL DEFAULT 0,
				created_at            TEXT NOT NULL
			)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_artifact_entity ON mg_artifact(entity_id, created_at)`,
			`CREATE INDEX IF NOT EXISTS idx_mg_artifact_conv ON mg_artifact(conversation_id, created_at)`,
			`CREATE TABLE IF NOT EXISTS mg_schema_version (
				id      INTEGER PRIMARY KEY DEFAULT 1,
				version INTEGER NOT NULL DEFAULT 1
			)`,
		},

		SlotCanonicalList:       `SELECT name, embedding, created_at FROM mg_slot_canonical`,
		SlotCanonicalInsert:     `INSERT OR IGNORE INTO mg_slot_canonical (name, embedding, created_at) VALUES (?, ?, ?)`,
		SlotCanonicalFindByName: `SELECT name, embedding, created_at FROM mg_slot_canonical WHERE name = ?`,

		SchemaRead:  `SELECT version FROM mg_schema_version WHERE id = 1`,
		SchemaWrite: `INSERT OR REPLACE INTO mg_schema_version (id, version) VALUES (1, ?)`,

		EntityInsert:   `INSERT OR IGNORE INTO mg_entity (uuid, external_id) VALUES (?, ?)`,
		EntitySelectID: `SELECT uuid FROM mg_entity WHERE external_id = ?`,
		EntitySelect:   `SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = ?`,

		FactInsert: `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		FactSelect: `SELECT uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at FROM mg_entity_fact WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?`,

		FactFindByKey:    `SELECT uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at FROM mg_entity_fact WHERE entity_id = ? AND content_key = ? LIMIT 1`,
		FactUpdateStatus: `UPDATE mg_entity_fact SET temporal_status = ?, updated_at = ? WHERE uuid = ?`,
		FactReinforce:    `UPDATE mg_entity_fact SET reinforced_at = ?, reinforced_count = reinforced_count + 1, expires_at = ?, updated_at = ? WHERE uuid = ?`,
		FactPruneExpired: `DELETE FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?`,
		FactDelete:             `DELETE FROM mg_entity_fact WHERE uuid = ? AND entity_id = ?`,
		FactDeleteAll:          `DELETE FROM mg_entity_fact WHERE entity_id = ?`,
		FactUpdateSignificance: `UPDATE mg_entity_fact SET significance = ?, updated_at = ? WHERE uuid = ?`,
		FactUpdateRecallUsage:  `UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?`,
		FactUpdateEmbedding:    `UPDATE mg_entity_fact SET embedding = ?, embedding_model = ?, updated_at = ? WHERE uuid = ?`,

		ConvInsert:          `INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
		ConvSelect:          `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE uuid = ?`,
		ConvSelectActive:    `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1`,
		ConvUpdateSummary:   `UPDATE mg_conversation SET summary = ?, summary_embedding = ?, summary_embedding_model = ?, updated_at = ? WHERE uuid = ?`,
		ConvSelectSummaries:    `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC LIMIT ?`,
		ConvSelectUnsummarized: `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary = '' AND session_id != ? ORDER BY created_at DESC LIMIT 1`,
		ConvPruneSummaries:     `UPDATE mg_conversation SET summary = '', summary_embedding = NULL, updated_at = ? WHERE summary != '' AND created_at < ?`,

		MsgInsert:       `INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
		MsgSelect:       `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at ASC`,
		MsgSelectRecent: `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?`,

		SessionInsert: `INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at, entity_mentions, message_count) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		SessionSelect: `SELECT uuid, entity_id, process_id, created_at, expires_at, entity_mentions, message_count FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1`,
		SessionSlide:  `UPDATE mg_session SET expires_at = ? WHERE uuid = ?`,

		ProcessInsert:   `INSERT OR IGNORE INTO mg_process (uuid, external_id) VALUES (?, ?)`,
		ProcessSelectID: `SELECT uuid FROM mg_process WHERE external_id = ?`,

		AttrInsert: `INSERT INTO mg_process_attribute (uuid, process_id, key, value, created_at) VALUES (?, ?, ?, ?, ?)`,
	}
}
