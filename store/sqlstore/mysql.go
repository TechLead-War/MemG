package sqlstore

import "database/sql"

// NewMySQL returns a Repository configured for MySQL 8+.
func NewMySQL(db *sql.DB) *Repository {
	return New(db, mysqlQueries())
}

func mysqlQueries() Queries {
	return Queries{
		CreateTables: []string{
			"CREATE TABLE IF NOT EXISTS mg_entity (" +
				"uuid        VARCHAR(36) PRIMARY KEY," +
				"external_id VARCHAR(255) NOT NULL UNIQUE," +
				"created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_entity_fact (" +
				"uuid             VARCHAR(36) PRIMARY KEY," +
				"entity_id        VARCHAR(36) NOT NULL," +
				"content          TEXT NOT NULL," +
				"embedding        LONGBLOB," +
				"created_at       DATETIME NOT NULL," +
				"updated_at       DATETIME NOT NULL," +
				"fact_type        VARCHAR(50) NOT NULL DEFAULT 'identity'," +
				"temporal_status  VARCHAR(50) NOT NULL DEFAULT 'current'," +
				"significance     INTEGER NOT NULL DEFAULT 5," +
				"content_key      VARCHAR(255) NOT NULL DEFAULT ''," +
				"reference_time   DATETIME," +
				"expires_at       DATETIME," +
				"reinforced_at    DATETIME," +
				"reinforced_count INTEGER NOT NULL DEFAULT 0," +
				"tag              VARCHAR(100) NOT NULL DEFAULT ''," +
				"slot             VARCHAR(100) NOT NULL DEFAULT ''," +
				"confidence       DOUBLE NOT NULL DEFAULT 1.0," +
				"embedding_model  VARCHAR(255) NOT NULL DEFAULT ''," +
				"source_role      VARCHAR(50) NOT NULL DEFAULT ''," +
				"recall_count     INTEGER NOT NULL DEFAULT 0," +
				"last_recalled_at DATETIME," +
				"FOREIGN KEY (entity_id) REFERENCES mg_entity(uuid)," +
				"INDEX idx_mg_fact_entity (entity_id)," +
				"INDEX idx_mg_fact_content_key (entity_id, content_key)," +
				"INDEX idx_mg_fact_expires (entity_id, expires_at)," +
				"INDEX idx_mg_fact_slot (entity_id, slot)" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_process (" +
				"uuid        VARCHAR(36) PRIMARY KEY," +
				"external_id VARCHAR(255) NOT NULL UNIQUE," +
				"created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_process_attribute (" +
				"uuid       VARCHAR(36) PRIMARY KEY," +
				"process_id VARCHAR(36) NOT NULL," +
				"`key`      VARCHAR(255) NOT NULL," +
				"value      TEXT NOT NULL," +
				"created_at DATETIME NOT NULL," +
				"FOREIGN KEY (process_id) REFERENCES mg_process(uuid)" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_session (" +
				"uuid       VARCHAR(36) PRIMARY KEY," +
				"entity_id  VARCHAR(36) NOT NULL," +
				"process_id VARCHAR(36) NOT NULL DEFAULT ''," +
				"created_at DATETIME NOT NULL," +
				"expires_at DATETIME NOT NULL," +
				"INDEX idx_mg_session_lookup (entity_id, process_id, expires_at)" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_conversation (" +
				"uuid              VARCHAR(36) PRIMARY KEY," +
				"session_id        VARCHAR(36) NOT NULL," +
				"entity_id         VARCHAR(36) NOT NULL DEFAULT ''," +
				"summary           TEXT NOT NULL," +
				"summary_embedding LONGBLOB," +
				"created_at        DATETIME NOT NULL," +
				"updated_at        DATETIME NOT NULL," +
				"INDEX idx_mg_conv_session (session_id, created_at)" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_message (" +
				"uuid            VARCHAR(36) PRIMARY KEY," +
				"conversation_id VARCHAR(36) NOT NULL," +
				"role            VARCHAR(50) NOT NULL," +
				"content         TEXT NOT NULL," +
				"kind            VARCHAR(50) NOT NULL DEFAULT 'text'," +
				"created_at      DATETIME NOT NULL," +
				"FOREIGN KEY (conversation_id) REFERENCES mg_conversation(uuid)," +
				"INDEX idx_mg_msg_conv (conversation_id, created_at)" +
				")",
			"CREATE TABLE IF NOT EXISTS mg_schema_version (" +
				"id      INTEGER PRIMARY KEY DEFAULT 1," +
				"version INTEGER NOT NULL DEFAULT 1" +
				")",
		},

		SchemaRead:  `SELECT version FROM mg_schema_version WHERE id = 1`,
		SchemaWrite: `INSERT INTO mg_schema_version (id, version) VALUES (1, ?) ON DUPLICATE KEY UPDATE version = VALUES(version)`,

		EntityInsert:   `INSERT IGNORE INTO mg_entity (uuid, external_id) VALUES (?, ?)`,
		EntitySelectID: `SELECT uuid FROM mg_entity WHERE external_id = ?`,
		EntitySelect:   `SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = ?`,

		FactInsert: `INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		FactSelect: `SELECT uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at FROM mg_entity_fact WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?`,

		FactFindByKey:          `SELECT uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at FROM mg_entity_fact WHERE entity_id = ? AND content_key = ? LIMIT 1`,
		FactUpdateStatus:       `UPDATE mg_entity_fact SET temporal_status = ?, updated_at = ? WHERE uuid = ?`,
		FactReinforce:          `UPDATE mg_entity_fact SET reinforced_at = ?, reinforced_count = reinforced_count + 1, expires_at = ?, updated_at = ? WHERE uuid = ?`,
		FactPruneExpired:       `DELETE FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?`,
		FactDelete:             `DELETE FROM mg_entity_fact WHERE uuid = ? AND entity_id = ?`,
		FactDeleteAll:          `DELETE FROM mg_entity_fact WHERE entity_id = ?`,
		FactUpdateSignificance: `UPDATE mg_entity_fact SET significance = ?, updated_at = ? WHERE uuid = ?`,
		FactUpdateRecallUsage:  `UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?`,
		FactUpdateEmbedding:    `UPDATE mg_entity_fact SET embedding = ?, embedding_model = ?, updated_at = ? WHERE uuid = ?`,

		ConvInsert:          `INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
		ConvSelect:          `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE uuid = ?`,
		ConvSelectActive:    `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1`,
		ConvUpdateSummary:   `UPDATE mg_conversation SET summary = ?, summary_embedding = ?, updated_at = ? WHERE uuid = ?`,
		ConvSelectSummaries:    `SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC LIMIT ?`,
		ConvSelectUnsummarized: `SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE entity_id = ? AND summary = '' AND session_id != ? ORDER BY created_at DESC LIMIT 1`,
		ConvPruneSummaries:     `UPDATE mg_conversation SET summary = '', summary_embedding = NULL, updated_at = ? WHERE summary != '' AND created_at < ?`,

		MsgInsert:       `INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
		MsgSelect:       `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at ASC`,
		MsgSelectRecent: `SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?`,

		SessionInsert: `INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?)`,
		SessionSelect: `SELECT uuid, entity_id, process_id, created_at, expires_at FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1`,
		SessionSlide:  `UPDATE mg_session SET expires_at = ? WHERE uuid = ?`,

		ProcessInsert:   `INSERT IGNORE INTO mg_process (uuid, external_id) VALUES (?, ?)`,
		ProcessSelectID: `SELECT uuid FROM mg_process WHERE external_id = ?`,

		AttrInsert: "INSERT INTO mg_process_attribute (uuid, process_id, `key`, value, created_at) VALUES (?, ?, ?, ?, ?)",
	}
}
