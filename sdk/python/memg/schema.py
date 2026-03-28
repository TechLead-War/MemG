"""SQLite DDL for the MemG schema."""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS mg_entity (
    uuid TEXT PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS mg_entity_fact (
    uuid TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES mg_entity(uuid),
    content TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    fact_type TEXT NOT NULL DEFAULT 'identity',
    temporal_status TEXT NOT NULL DEFAULT 'current',
    significance INTEGER NOT NULL DEFAULT 5,
    content_key TEXT NOT NULL DEFAULT '',
    reference_time TEXT,
    expires_at TEXT,
    reinforced_at TEXT,
    reinforced_count INTEGER NOT NULL DEFAULT 0,
    tag TEXT NOT NULL DEFAULT '',
    slot TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    embedding_model TEXT NOT NULL DEFAULT '',
    source_role TEXT NOT NULL DEFAULT '',
    recall_count INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_mg_fact_entity ON mg_entity_fact(entity_id);
CREATE INDEX IF NOT EXISTS idx_mg_fact_content_key ON mg_entity_fact(entity_id, content_key);
CREATE INDEX IF NOT EXISTS idx_mg_fact_expires ON mg_entity_fact(entity_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_mg_fact_slot ON mg_entity_fact(entity_id, slot);
CREATE TABLE IF NOT EXISTS mg_session (
    uuid TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    process_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    entity_mentions TEXT NOT NULL DEFAULT '[]',
    message_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mg_session_lookup ON mg_session(entity_id, process_id, expires_at);
CREATE TABLE IF NOT EXISTS mg_conversation (
    uuid TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    entity_id TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    summary_embedding BLOB,
    summary_embedding_model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mg_conv_session ON mg_conversation(session_id, created_at);
CREATE TABLE IF NOT EXISTS mg_message (
    uuid TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES mg_conversation(uuid),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'text',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mg_msg_conv ON mg_message(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS mg_turn_summary (
    uuid TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    start_turn INTEGER NOT NULL,
    end_turn INTEGER NOT NULL,
    summary TEXT NOT NULL,
    summary_embedding BLOB,
    is_overview INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mg_turnsummary_conv ON mg_turn_summary(conversation_id, is_overview, created_at);
CREATE TABLE IF NOT EXISTS mg_artifact (
    uuid TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content TEXT NOT NULL,
    artifact_type TEXT NOT NULL DEFAULT 'code',
    language TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    description_embedding BLOB,
    superseded_by TEXT,
    turn_number INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mg_artifact_entity ON mg_artifact(entity_id, created_at);
CREATE INDEX IF NOT EXISTS idx_mg_artifact_conv ON mg_artifact(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS mg_process (
    uuid TEXT PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS mg_process_attribute (
    uuid TEXT PRIMARY KEY,
    process_id TEXT NOT NULL REFERENCES mg_process(uuid),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mg_slot_canonical (
    name TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS mg_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1
);
"""
