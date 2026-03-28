// Package sqlstore provides a generic SQL-backed implementation of store.Repository.
// Database-specific dialects supply the query templates via the Queries struct.
package sqlstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"memg/store"
)

// Queries holds every SQL statement a dialect must supply.
type Queries struct {
	// Schema DDL executed in order during Migrate.
	CreateTables []string

	// Entity
	EntityInsert   string // INSERT-ignore style upsert
	EntitySelectID string // SELECT uuid WHERE external_id = ?
	EntitySelect   string // SELECT uuid, external_id, created_at WHERE external_id = ?

	// Fact
	FactInsert string // INSERT with all columns including lifecycle metadata
	FactSelect string // SELECT WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?

	// Fact lifecycle
	FactFindByKey          string // SELECT WHERE entity_id = ? AND content_key = ?
	FactUpdateStatus       string // UPDATE SET temporal_status = ? WHERE uuid = ?
	FactReinforce          string // UPDATE SET reinforced_at = ?, reinforced_count = reinforced_count + 1, expires_at = ? WHERE uuid = ?
	FactPruneExpired       string // DELETE WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?
	FactDelete             string // DELETE WHERE uuid = ? AND entity_id = ?
	FactDeleteAll          string // DELETE WHERE entity_id = ?
	FactUpdateSignificance string // UPDATE SET significance = ?, updated_at = ? WHERE uuid = ?
	FactUpdateRecallUsage  string // UPDATE SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?
	FactUpdateEmbedding    string // UPDATE embedding = ?, embedding_model = ?, updated_at = ? WHERE uuid = ?

	// Conversation
	ConvInsert             string // INSERT (uuid, session_id, entity_id, created_at, updated_at)
	ConvSelect             string // SELECT WHERE uuid = ?
	ConvSelectActive       string // SELECT WHERE session_id = ? ORDER BY created_at DESC LIMIT 1
	ConvUpdateSummary      string // UPDATE SET summary = ?, summary_embedding = ?, updated_at = ? WHERE uuid = ?
	ConvSelectSummaries    string // SELECT WHERE entity_id != '' AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC LIMIT ?
	ConvSelectUnsummarized string // SELECT WHERE entity_id = ? AND summary = '' AND session_id != ? ORDER BY created_at DESC LIMIT 1
	ConvPruneSummaries     string // UPDATE SET summary = '', summary_embedding = NULL WHERE summary != '' AND created_at < ?

	// Message
	MsgInsert       string // INSERT (uuid, conversation_id, role, content, kind, created_at)
	MsgSelect       string // SELECT WHERE conversation_id = ? ORDER BY created_at ASC
	MsgSelectRecent string // SELECT ... ORDER BY created_at DESC LIMIT ? then reverse in Go

	// Session
	SessionInsert string // INSERT (uuid, entity_id, process_id, created_at, expires_at)
	SessionSelect string // SELECT active WHERE entity_id = ? AND process_id = ? AND expires_at > ?
	SessionSlide  string // UPDATE mg_session SET expires_at = ? WHERE uuid = ?

	// Process
	ProcessInsert   string // INSERT-ignore style upsert
	ProcessSelectID string // SELECT uuid WHERE external_id = ?

	// Attribute
	AttrInsert string // INSERT (uuid, process_id, key, value, created_at)

	// Canonical slots
	SlotCanonicalList       string // SELECT name, embedding, created_at FROM mg_slot_canonical
	SlotCanonicalInsert     string // INSERT INTO mg_slot_canonical (name, embedding, created_at)
	SlotCanonicalFindByName string // SELECT ... WHERE name = ?

	// Schema versioning
	SchemaRead  string
	SchemaWrite string
}

// Repository implements store.Repository on top of database/sql.
type Repository struct {
	db *sql.DB
	q  Queries
}

// New creates a Repository with the given SQL handle and query set.
func New(db *sql.DB, q Queries) *Repository {
	return &Repository{db: db, q: q}
}

// ---- Migrator ----

func (r *Repository) Migrate(ctx context.Context) error {
	for _, ddl := range r.q.CreateTables {
		if _, err := r.db.ExecContext(ctx, ddl); err != nil {
			return fmt.Errorf("sqlstore migrate: %w", err)
		}
	}
	return nil
}

func (r *Repository) SchemaVersion(ctx context.Context) (int, error) {
	var ver int
	err := r.db.QueryRowContext(ctx, r.q.SchemaRead).Scan(&ver)
	if err == sql.ErrNoRows {
		return 0, nil
	}
	return ver, err
}

// ---- EntityWriter ----

func (r *Repository) UpsertEntity(ctx context.Context, externalID string) (string, error) {
	id := uuid.New().String()
	if _, err := r.db.ExecContext(ctx, r.q.EntityInsert, id, externalID); err != nil {
		return "", fmt.Errorf("upsert entity: %w", err)
	}
	var resultID string
	if err := r.db.QueryRowContext(ctx, r.q.EntitySelectID, externalID).Scan(&resultID); err != nil {
		return "", fmt.Errorf("read entity uuid: %w", err)
	}
	return resultID, nil
}

// ---- EntityReader ----

func (r *Repository) LookupEntity(ctx context.Context, externalID string) (*store.Entity, error) {
	e := &store.Entity{}
	var createdAt flexTime
	err := r.db.QueryRowContext(ctx, r.q.EntitySelect, externalID).Scan(
		&e.UUID, &e.ExternalID, &createdAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("lookup entity: %w", err)
	}
	e.CreatedAt = createdAt.Time
	return e, nil
}

// ---- EntityLister ----

func (r *Repository) ListEntityUUIDs(ctx context.Context, limit int) ([]string, error) {
	query := "SELECT uuid FROM mg_entity ORDER BY created_at ASC LIMIT " + fmt.Sprintf("%d", limit)
	rows, err := r.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("list entity uuids: %w", err)
	}
	defer rows.Close()
	var uuids []string
	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return nil, fmt.Errorf("scan entity uuid: %w", err)
		}
		uuids = append(uuids, uuid)
	}
	return uuids, rows.Err()
}

// ---- FactWriter ----

func (r *Repository) InsertFact(ctx context.Context, entityUUID string, fact *store.Fact) error {
	fact.UUID = uuid.New().String()
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.FactInsert,
		fact.UUID, entityUUID, fact.Content, encodeEmbedding(fact.Embedding), now, now,
		string(fact.Type), string(fact.TemporalStatus), int(fact.Significance),
		fact.ContentKey, nullTime(fact.ReferenceTime), nullTime(fact.ExpiresAt),
		nullTime(fact.ReinforcedAt), fact.ReinforcedCount, fact.Tag,
		fact.Slot, fact.Confidence, fact.EmbeddingModel, fact.SourceRole,
		fact.RecallCount, nullTime(fact.LastRecalledAt),
	)
	if err != nil {
		return fmt.Errorf("insert fact: %w", err)
	}
	return nil
}

func (r *Repository) InsertFacts(ctx context.Context, entityUUID string, facts []*store.Fact) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, r.q.FactInsert)
	if err != nil {
		return fmt.Errorf("prepare insert: %w", err)
	}
	defer stmt.Close()

	now := time.Now().UTC()
	for _, f := range facts {
		f.UUID = uuid.New().String()
		if _, err := stmt.ExecContext(ctx,
			f.UUID, entityUUID, f.Content, encodeEmbedding(f.Embedding), now, now,
			string(f.Type), string(f.TemporalStatus), int(f.Significance),
			f.ContentKey, nullTime(f.ReferenceTime), nullTime(f.ExpiresAt),
			nullTime(f.ReinforcedAt), f.ReinforcedCount, f.Tag,
			f.Slot, f.Confidence, f.EmbeddingModel, f.SourceRole,
			f.RecallCount, nullTime(f.LastRecalledAt),
		); err != nil {
			return fmt.Errorf("insert fact %s: %w", f.UUID, err)
		}
	}
	return tx.Commit()
}

// ---- FactReader ----

func (r *Repository) ListFacts(ctx context.Context, entityUUID string, limit int) ([]*store.Fact, error) {
	rows, err := r.db.QueryContext(ctx, r.q.FactSelect, entityUUID, limit)
	if err != nil {
		return nil, fmt.Errorf("list facts: %w", err)
	}
	defer rows.Close()
	return scanFacts(rows)
}

// ---- FactManager ----

func (r *Repository) FindFactByKey(ctx context.Context, entityUUID, contentKey string) (*store.Fact, error) {
	f := &store.Fact{}
	var raw []byte
	var factType, temporalStatus string
	var significance int
	var createdAt, updatedAt flexTime
	var refTime, expiresAt, reinforcedAt flexTime
	var lastRecalledAt flexTime
	err := r.db.QueryRowContext(ctx, r.q.FactFindByKey, entityUUID, contentKey).Scan(
		&f.UUID, &f.Content, &raw, &createdAt, &updatedAt,
		&factType, &temporalStatus, &significance,
		&f.ContentKey, &refTime, &expiresAt,
		&reinforcedAt, &f.ReinforcedCount, &f.Tag,
		&f.Slot, &f.Confidence, &f.EmbeddingModel, &f.SourceRole,
		&f.RecallCount, &lastRecalledAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("find fact by key: %w", err)
	}
	f.Embedding = decodeEmbedding(raw)
	f.Type = store.FactType(factType)
	f.TemporalStatus = store.TemporalStatus(temporalStatus)
	f.Significance = store.Significance(significance)
	f.CreatedAt = createdAt.Time
	f.UpdatedAt = updatedAt.Time
	if refTime.Valid {
		f.ReferenceTime = &refTime.Time
	}
	if expiresAt.Valid {
		f.ExpiresAt = &expiresAt.Time
	}
	if reinforcedAt.Valid {
		f.ReinforcedAt = &reinforcedAt.Time
	}
	if lastRecalledAt.Valid {
		f.LastRecalledAt = &lastRecalledAt.Time
	}
	return f, nil
}

func (r *Repository) UpdateTemporalStatus(ctx context.Context, factUUID string, status store.TemporalStatus) error {
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.FactUpdateStatus, string(status), now, factUUID)
	if err != nil {
		return fmt.Errorf("update temporal status: %w", err)
	}
	return nil
}

func (r *Repository) ReinforceFact(ctx context.Context, factUUID string, newExpiresAt *time.Time) error {
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.FactReinforce, now, nullTime(newExpiresAt), now, factUUID)
	if err != nil {
		return fmt.Errorf("reinforce fact: %w", err)
	}
	return nil
}

func (r *Repository) PruneExpiredFacts(ctx context.Context, entityUUID string, now time.Time) (int64, error) {
	var res sql.Result
	var err error
	if entityUUID == "" {
		// Two-step batch prune for cross-dialect compatibility.
		// MySQL < 8.0.19 disallows self-referencing subqueries in DELETE,
		// so we SELECT the UUIDs first, then DELETE by UUID list.
		selectQuery := `SELECT uuid FROM mg_entity_fact WHERE expires_at IS NOT NULL AND expires_at < ` + r.placeholder(1) + ` LIMIT 1000`
		rows, qErr := r.db.QueryContext(ctx, selectQuery, now)
		if qErr != nil {
			return 0, fmt.Errorf("prune select: %w", qErr)
		}
		var uuids []string
		for rows.Next() {
			var id string
			if sErr := rows.Scan(&id); sErr != nil {
				rows.Close()
				return 0, fmt.Errorf("prune scan: %w", sErr)
			}
			uuids = append(uuids, id)
		}
		rows.Close()
		if rErr := rows.Err(); rErr != nil {
			return 0, fmt.Errorf("prune rows: %w", rErr)
		}
		if len(uuids) == 0 {
			return 0, nil
		}
		// Step 2: Delete by UUID list.
		placeholders := make([]string, len(uuids))
		args := make([]any, len(uuids))
		for i, id := range uuids {
			placeholders[i] = r.placeholder(i + 1)
			args[i] = id
		}
		deleteQuery := `DELETE FROM mg_entity_fact WHERE uuid IN (` + strings.Join(placeholders, ",") + `)`
		res, err = r.db.ExecContext(ctx, deleteQuery, args...)
	} else {
		res, err = r.db.ExecContext(ctx, r.q.FactPruneExpired, entityUUID, now)
	}
	if err != nil {
		return 0, fmt.Errorf("prune expired facts: %w", err)
	}
	return res.RowsAffected()
}

// placeholder returns a dialect-appropriate parameter placeholder for position n.
func (r *Repository) placeholder(n int) string {
	if containsDollarParam(r.q.FactInsert) {
		return fmt.Sprintf("$%d", n)
	}
	return "?"
}

func (r *Repository) UpdateSignificance(ctx context.Context, factUUID string, sig store.Significance) error {
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.FactUpdateSignificance, int(sig), now, factUUID)
	if err != nil {
		return fmt.Errorf("update significance: %w", err)
	}
	return nil
}

func (r *Repository) UpdateFactEmbedding(ctx context.Context, factUUID string, embedding []float32, model string) error {
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.FactUpdateEmbedding, encodeEmbedding(embedding), model, now, factUUID)
	if err != nil {
		return fmt.Errorf("update fact embedding: %w", err)
	}
	return nil
}

func (r *Repository) DeleteFact(ctx context.Context, entityUUID, factUUID string) error {
	_, err := r.db.ExecContext(ctx, r.q.FactDelete, factUUID, entityUUID)
	if err != nil {
		return fmt.Errorf("delete fact: %w", err)
	}
	return nil
}

func (r *Repository) DeleteEntityFacts(ctx context.Context, entityUUID string) (int64, error) {
	res, err := r.db.ExecContext(ctx, r.q.FactDeleteAll, entityUUID)
	if err != nil {
		return 0, fmt.Errorf("delete entity facts: %w", err)
	}
	return res.RowsAffected()
}

// ---- RecallUsageTracker ----

func (r *Repository) UpdateRecallUsage(ctx context.Context, factUUIDs []string) error {
	if len(factUUIDs) == 0 || r.q.FactUpdateRecallUsage == "" {
		return nil
	}
	now := time.Now().UTC()
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, r.q.FactUpdateRecallUsage)
	if err != nil {
		return fmt.Errorf("prepare recall usage update: %w", err)
	}
	defer stmt.Close()

	for _, id := range factUUIDs {
		if _, err := stmt.ExecContext(ctx, now, id); err != nil {
			return fmt.Errorf("update recall usage for %s: %w", id, err)
		}
	}
	return tx.Commit()
}

// ---- FactFilteredReader ----

func (r *Repository) ListFactsFiltered(ctx context.Context, entityUUID string, filter store.FactFilter, limit int) ([]*store.Fact, error) {
	query := `SELECT uuid, content, embedding, created_at, updated_at,
		fact_type, temporal_status, significance, content_key,
		reference_time, expires_at, reinforced_at, reinforced_count, tag,
		slot, confidence, embedding_model, source_role,
		recall_count, last_recalled_at
		FROM mg_entity_fact WHERE entity_id = ?`
	args := []any{entityUUID}

	if len(filter.Types) > 0 {
		placeholders := make([]string, len(filter.Types))
		for i, t := range filter.Types {
			placeholders[i] = "?"
			args = append(args, string(t))
		}
		query += " AND fact_type IN (" + joinStrings(placeholders, ",") + ")"
	}
	if len(filter.Statuses) > 0 {
		placeholders := make([]string, len(filter.Statuses))
		for i, s := range filter.Statuses {
			placeholders[i] = "?"
			args = append(args, string(s))
		}
		query += " AND temporal_status IN (" + joinStrings(placeholders, ",") + ")"
	}
	if len(filter.Tags) > 0 {
		placeholders := make([]string, len(filter.Tags))
		for i, t := range filter.Tags {
			placeholders[i] = "?"
			args = append(args, t)
		}
		query += " AND tag IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.MinSignificance > 0 {
		query += " AND significance >= ?"
		args = append(args, int(filter.MinSignificance))
	}
	if filter.ExcludeExpired {
		query += " AND (expires_at IS NULL OR expires_at > ?)"
		args = append(args, time.Now().UTC())
	}
	if filter.ReferenceTimeAfter != nil {
		query += " AND reference_time IS NOT NULL AND reference_time >= ?"
		args = append(args, *filter.ReferenceTimeAfter)
	}
	if filter.ReferenceTimeBefore != nil {
		query += " AND reference_time IS NOT NULL AND reference_time <= ?"
		args = append(args, *filter.ReferenceTimeBefore)
	}
	if len(filter.Slots) > 0 {
		placeholders := make([]string, len(filter.Slots))
		for i, sl := range filter.Slots {
			placeholders[i] = "?"
			args = append(args, sl)
		}
		query += " AND slot IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.MinConfidence > 0 {
		query += " AND confidence >= ?"
		args = append(args, filter.MinConfidence)
	}
	if len(filter.SourceRoles) > 0 {
		placeholders := make([]string, len(filter.SourceRoles))
		for i, sr := range filter.SourceRoles {
			placeholders[i] = "?"
			args = append(args, sr)
		}
		query += " AND source_role IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.UnembeddedOnly {
		query += " AND embedding IS NULL"
	}
	if filter.MaxSignificance > 0 {
		query += " AND significance <= ?"
		args = append(args, int(filter.MaxSignificance))
	}
	query += " ORDER BY created_at DESC LIMIT ?"
	args = append(args, limit)

	// Convert ? placeholders to dialect-specific ones for postgres ($1, $2, ...).
	query = r.rewritePlaceholders(query)

	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list facts filtered: %w", err)
	}
	defer rows.Close()
	return scanFacts(rows)
}

// ---- FactRecallReader ----

func (r *Repository) ListFactsForRecall(ctx context.Context, entityUUID string, filter store.FactFilter, limit int) ([]*store.Fact, error) {
	query := `SELECT uuid, content, embedding, created_at, updated_at,
		fact_type, temporal_status, significance, content_key,
		reference_time, expires_at, reinforced_at, reinforced_count, tag,
		slot, confidence, embedding_model, source_role,
		recall_count, last_recalled_at
		FROM mg_entity_fact WHERE entity_id = ?`
	args := []any{entityUUID}

	if len(filter.Types) > 0 {
		placeholders := make([]string, len(filter.Types))
		for i, t := range filter.Types {
			placeholders[i] = "?"
			args = append(args, string(t))
		}
		query += " AND fact_type IN (" + joinStrings(placeholders, ",") + ")"
	}
	if len(filter.Statuses) > 0 {
		placeholders := make([]string, len(filter.Statuses))
		for i, s := range filter.Statuses {
			placeholders[i] = "?"
			args = append(args, string(s))
		}
		query += " AND temporal_status IN (" + joinStrings(placeholders, ",") + ")"
	}
	if len(filter.Tags) > 0 {
		placeholders := make([]string, len(filter.Tags))
		for i, t := range filter.Tags {
			placeholders[i] = "?"
			args = append(args, t)
		}
		query += " AND tag IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.MinSignificance > 0 {
		query += " AND significance >= ?"
		args = append(args, int(filter.MinSignificance))
	}
	if filter.ExcludeExpired {
		query += " AND (expires_at IS NULL OR expires_at > ?)"
		args = append(args, time.Now().UTC())
	}
	if filter.ReferenceTimeAfter != nil {
		query += " AND reference_time IS NOT NULL AND reference_time >= ?"
		args = append(args, *filter.ReferenceTimeAfter)
	}
	if filter.ReferenceTimeBefore != nil {
		query += " AND reference_time IS NOT NULL AND reference_time <= ?"
		args = append(args, *filter.ReferenceTimeBefore)
	}
	if len(filter.Slots) > 0 {
		placeholders := make([]string, len(filter.Slots))
		for i, sl := range filter.Slots {
			placeholders[i] = "?"
			args = append(args, sl)
		}
		query += " AND slot IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.MinConfidence > 0 {
		query += " AND confidence >= ?"
		args = append(args, filter.MinConfidence)
	}
	if len(filter.SourceRoles) > 0 {
		placeholders := make([]string, len(filter.SourceRoles))
		for i, sr := range filter.SourceRoles {
			placeholders[i] = "?"
			args = append(args, sr)
		}
		query += " AND source_role IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.UnembeddedOnly {
		query += " AND embedding IS NULL"
	}
	if filter.MaxSignificance > 0 {
		query += " AND significance <= ?"
		args = append(args, int(filter.MaxSignificance))
	}
	query += " ORDER BY significance DESC, created_at DESC"
	if limit > 0 {
		query += " LIMIT ?"
		args = append(args, limit)
	}

	query = r.rewritePlaceholders(query)

	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list facts for recall: %w", err)
	}
	defer rows.Close()
	return scanFacts(rows)
}

// ---- FactMetadataReader ----

func (r *Repository) ListFactsMetadata(ctx context.Context, entityUUID string, filter store.FactFilter, limit int) ([]*store.Fact, error) {
	query := `SELECT uuid, content, created_at, updated_at,
		fact_type, temporal_status, significance, content_key,
		reference_time, expires_at, reinforced_at, reinforced_count, tag,
		slot, confidence, embedding_model, source_role,
		recall_count, last_recalled_at
		FROM mg_entity_fact WHERE entity_id = ?`
	args := []any{entityUUID}

	if len(filter.Types) > 0 {
		placeholders := make([]string, len(filter.Types))
		for i, t := range filter.Types {
			placeholders[i] = "?"
			args = append(args, string(t))
		}
		query += " AND fact_type IN (" + joinStrings(placeholders, ",") + ")"
	}
	if len(filter.Statuses) > 0 {
		placeholders := make([]string, len(filter.Statuses))
		for i, s := range filter.Statuses {
			placeholders[i] = "?"
			args = append(args, string(s))
		}
		query += " AND temporal_status IN (" + joinStrings(placeholders, ",") + ")"
	}
	if len(filter.Tags) > 0 {
		placeholders := make([]string, len(filter.Tags))
		for i, t := range filter.Tags {
			placeholders[i] = "?"
			args = append(args, t)
		}
		query += " AND tag IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.MinSignificance > 0 {
		query += " AND significance >= ?"
		args = append(args, int(filter.MinSignificance))
	}
	if filter.ExcludeExpired {
		query += " AND (expires_at IS NULL OR expires_at > ?)"
		args = append(args, time.Now().UTC())
	}
	if filter.ReferenceTimeAfter != nil {
		query += " AND reference_time IS NOT NULL AND reference_time >= ?"
		args = append(args, *filter.ReferenceTimeAfter)
	}
	if filter.ReferenceTimeBefore != nil {
		query += " AND reference_time IS NOT NULL AND reference_time <= ?"
		args = append(args, *filter.ReferenceTimeBefore)
	}
	if len(filter.Slots) > 0 {
		placeholders := make([]string, len(filter.Slots))
		for i, sl := range filter.Slots {
			placeholders[i] = "?"
			args = append(args, sl)
		}
		query += " AND slot IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.MinConfidence > 0 {
		query += " AND confidence >= ?"
		args = append(args, filter.MinConfidence)
	}
	if len(filter.SourceRoles) > 0 {
		placeholders := make([]string, len(filter.SourceRoles))
		for i, sr := range filter.SourceRoles {
			placeholders[i] = "?"
			args = append(args, sr)
		}
		query += " AND source_role IN (" + joinStrings(placeholders, ",") + ")"
	}
	if filter.UnembeddedOnly {
		query += " AND embedding IS NULL"
	}
	if filter.MaxSignificance > 0 {
		query += " AND significance <= ?"
		args = append(args, int(filter.MaxSignificance))
	}
	query += " ORDER BY created_at DESC LIMIT ?"
	args = append(args, limit)

	query = r.rewritePlaceholders(query)

	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list facts metadata: %w", err)
	}
	defer rows.Close()
	return scanFactsMetadata(rows)
}

// scanFactsMetadata scans fact rows without embedding data.
func scanFactsMetadata(rows *sql.Rows) ([]*store.Fact, error) {
	var out []*store.Fact
	for rows.Next() {
		f := &store.Fact{}
		var factType, temporalStatus string
		var significance int
		var createdAt, updatedAt flexTime
		var refTime, expiresAt, reinforcedAt flexTime
		var lastRecalledAt flexTime
		if err := rows.Scan(
			&f.UUID, &f.Content, &createdAt, &updatedAt,
			&factType, &temporalStatus, &significance,
			&f.ContentKey, &refTime, &expiresAt,
			&reinforcedAt, &f.ReinforcedCount, &f.Tag,
			&f.Slot, &f.Confidence, &f.EmbeddingModel, &f.SourceRole,
			&f.RecallCount, &lastRecalledAt,
		); err != nil {
			return nil, fmt.Errorf("scan fact metadata: %w", err)
		}
		f.Type = store.FactType(factType)
		f.TemporalStatus = store.TemporalStatus(temporalStatus)
		f.Significance = store.Significance(significance)
		f.CreatedAt = createdAt.Time
		f.UpdatedAt = updatedAt.Time
		if refTime.Valid {
			f.ReferenceTime = &refTime.Time
		}
		if expiresAt.Valid {
			f.ExpiresAt = &expiresAt.Time
		}
		if reinforcedAt.Valid {
			f.ReinforcedAt = &reinforcedAt.Time
		}
		if lastRecalledAt.Valid {
			f.LastRecalledAt = &lastRecalledAt.Time
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

// rewritePlaceholders converts ? placeholders to $N for postgres dialects.
// It checks whether the dialect uses $1-style params by inspecting FactInsert.
func (r *Repository) rewritePlaceholders(query string) string {
	if len(r.q.FactInsert) == 0 || !containsDollarParam(r.q.FactInsert) {
		return query // MySQL and SQLite use ? natively
	}
	var b strings.Builder
	n := 1
	for i := 0; i < len(query); i++ {
		if query[i] == '?' {
			b.WriteString(fmt.Sprintf("$%d", n))
			n++
		} else {
			b.WriteByte(query[i])
		}
	}
	return b.String()
}

func containsDollarParam(s string) bool {
	return strings.Contains(s, "$1")
}

func joinStrings(parts []string, sep string) string {
	return strings.Join(parts, sep)
}

// scanFacts reads fact rows including all lifecycle columns.
func scanFacts(rows *sql.Rows) ([]*store.Fact, error) {
	var out []*store.Fact
	for rows.Next() {
		f := &store.Fact{}
		var raw []byte
		var factType, temporalStatus string
		var significance int
		var createdAt, updatedAt flexTime
		var refTime, expiresAt, reinforcedAt flexTime
		var lastRecalledAt flexTime
		if err := rows.Scan(
			&f.UUID, &f.Content, &raw, &createdAt, &updatedAt,
			&factType, &temporalStatus, &significance,
			&f.ContentKey, &refTime, &expiresAt,
			&reinforcedAt, &f.ReinforcedCount, &f.Tag,
			&f.Slot, &f.Confidence, &f.EmbeddingModel, &f.SourceRole,
			&f.RecallCount, &lastRecalledAt,
		); err != nil {
			return nil, fmt.Errorf("scan fact: %w", err)
		}
		f.Embedding = decodeEmbedding(raw)
		f.Type = store.FactType(factType)
		f.TemporalStatus = store.TemporalStatus(temporalStatus)
		f.Significance = store.Significance(significance)
		f.CreatedAt = createdAt.Time
		f.UpdatedAt = updatedAt.Time
		if refTime.Valid {
			f.ReferenceTime = &refTime.Time
		}
		if expiresAt.Valid {
			f.ExpiresAt = &expiresAt.Time
		}
		if reinforcedAt.Valid {
			f.ReinforcedAt = &reinforcedAt.Time
		}
		if lastRecalledAt.Valid {
			f.LastRecalledAt = &lastRecalledAt.Time
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

// nullTime converts a *time.Time to a value suitable for a nullable SQL column.
func nullTime(t *time.Time) any {
	if t == nil {
		return nil
	}
	return *t
}

// ---- ConversationWriter ----

func (r *Repository) StartConversation(ctx context.Context, sessionUUID, entityUUID string) (string, error) {
	id := uuid.New().String()
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.ConvInsert, id, sessionUUID, entityUUID, now, now)
	if err != nil {
		return "", fmt.Errorf("start conversation: %w", err)
	}
	return id, nil
}

// ---- ConversationReader ----

func (r *Repository) FindConversation(ctx context.Context, id string) (*store.Conversation, error) {
	c := &store.Conversation{}
	var createdAt, updatedAt flexTime
	err := r.db.QueryRowContext(ctx, r.q.ConvSelect, id).Scan(
		&c.UUID, &c.SessionID, &c.EntityID, &c.Summary, &createdAt, &updatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("find conversation: %w", err)
	}
	c.CreatedAt = createdAt.Time
	c.UpdatedAt = updatedAt.Time
	return c, nil
}

func (r *Repository) ActiveConversation(ctx context.Context, sessionUUID string) (*store.Conversation, error) {
	c := &store.Conversation{}
	var createdAt, updatedAt flexTime
	err := r.db.QueryRowContext(ctx, r.q.ConvSelectActive, sessionUUID).Scan(
		&c.UUID, &c.SessionID, &c.EntityID, &c.Summary, &createdAt, &updatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("active conversation: %w", err)
	}
	c.CreatedAt = createdAt.Time
	c.UpdatedAt = updatedAt.Time
	return c, nil
}

func (r *Repository) UpdateConversationSummary(ctx context.Context, conversationUUID, summary string, embedding []float32, embeddingModel string) error {
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.ConvUpdateSummary, summary, encodeEmbedding(embedding), embeddingModel, now, conversationUUID)
	if err != nil {
		return fmt.Errorf("update conversation summary: %w", err)
	}
	return nil
}

func (r *Repository) ListConversationSummaries(ctx context.Context, entityUUID string, limit int) ([]*store.Conversation, error) {
	query := `SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at
		FROM mg_conversation
		WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL`
	args := []any{entityUUID}
	if limit > 0 {
		query += " ORDER BY created_at DESC LIMIT ?"
		args = append(args, limit)
	}
	query = r.rewritePlaceholders(query)

	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list conversation summaries: %w", err)
	}
	defer rows.Close()

	var out []*store.Conversation
	for rows.Next() {
		c := &store.Conversation{}
		var raw []byte
		var createdAt, updatedAt flexTime
		if err := rows.Scan(&c.UUID, &c.SessionID, &c.EntityID, &c.Summary, &raw, &c.SummaryEmbeddingModel, &createdAt, &updatedAt); err != nil {
			return nil, fmt.Errorf("scan conversation summary: %w", err)
		}
		c.CreatedAt = createdAt.Time
		c.UpdatedAt = updatedAt.Time
		c.SummaryEmbedding = decodeEmbedding(raw)
		out = append(out, c)
	}
	return out, rows.Err()
}

func (r *Repository) FindUnsummarizedConversation(ctx context.Context, entityUUID, excludeSessionUUID string) (*store.Conversation, error) {
	c := &store.Conversation{}
	var createdAt, updatedAt flexTime
	err := r.db.QueryRowContext(ctx, r.q.ConvSelectUnsummarized, entityUUID, excludeSessionUUID).Scan(
		&c.UUID, &c.SessionID, &c.EntityID, &c.Summary, &createdAt, &updatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("find unsummarized conversation: %w", err)
	}
	c.CreatedAt = createdAt.Time
	c.UpdatedAt = updatedAt.Time
	return c, nil
}

// ---- ConversationPruner ----

func (r *Repository) PruneStaleSummaries(ctx context.Context, olderThan time.Time) (int64, error) {
	now := time.Now().UTC()
	res, err := r.db.ExecContext(ctx, r.q.ConvPruneSummaries, now, olderThan)
	if err != nil {
		return 0, fmt.Errorf("prune stale summaries: %w", err)
	}
	return res.RowsAffected()
}

// ---- MessageWriter ----

func (r *Repository) AppendMessage(ctx context.Context, conversationUUID string, msg *store.Message) error {
	msg.UUID = uuid.New().String()
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.MsgInsert,
		msg.UUID, conversationUUID, msg.Role, msg.Content, msg.Kind, now,
	)
	if err != nil {
		return fmt.Errorf("append message: %w", err)
	}
	return nil
}

// ---- MessageReader ----

func (r *Repository) ReadMessages(ctx context.Context, conversationUUID string) ([]*store.Message, error) {
	rows, err := r.db.QueryContext(ctx, r.q.MsgSelect, conversationUUID)
	if err != nil {
		return nil, fmt.Errorf("read messages: %w", err)
	}
	defer rows.Close()

	var out []*store.Message
	for rows.Next() {
		m := &store.Message{}
		var createdAt flexTime
		if err := rows.Scan(&m.UUID, &m.ConversationID, &m.Role, &m.Content, &m.Kind, &createdAt); err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}
		m.CreatedAt = createdAt.Time
		out = append(out, m)
	}
	return out, rows.Err()
}

func (r *Repository) ReadRecentMessages(ctx context.Context, conversationUUID string, limit int) ([]*store.Message, error) {
	rows, err := r.db.QueryContext(ctx, r.q.MsgSelectRecent, conversationUUID, limit)
	if err != nil {
		return nil, fmt.Errorf("read recent messages: %w", err)
	}
	defer rows.Close()

	var out []*store.Message
	for rows.Next() {
		m := &store.Message{}
		var createdAt flexTime
		if err := rows.Scan(&m.UUID, &m.ConversationID, &m.Role, &m.Content, &m.Kind, &createdAt); err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}
		m.CreatedAt = createdAt.Time
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	// Reverse to get chronological order.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, nil
}

// ---- SessionWriter ----

func (r *Repository) EnsureSession(ctx context.Context, entityUUID, processUUID string, timeout time.Duration) (*store.Session, bool, error) {
	now := time.Now().UTC()

	// Check for an active (non-expired) session.
	s := &store.Session{}
	var createdAt, expiresAt flexTime
	var mentionsJSON string
	err := r.db.QueryRowContext(ctx, r.q.SessionSelect, entityUUID, processUUID, now).Scan(
		&s.UUID, &s.EntityID, &s.ProcessID, &createdAt, &expiresAt,
		&mentionsJSON, &s.MessageCount,
	)
	if err == nil {
		s.CreatedAt = createdAt.Time
		s.ExpiresAt = expiresAt.Time
		_ = json.Unmarshal([]byte(mentionsJSON), &s.EntityMentions)
		// Slide the expiry forward.
		newExpiry := now.Add(timeout)
		if _, slideErr := r.db.ExecContext(ctx, r.q.SessionSlide, newExpiry, s.UUID); slideErr != nil {
			return nil, false, fmt.Errorf("slide session expiry: %w", slideErr)
		}
		s.ExpiresAt = newExpiry
		return s, false, nil
	}
	if err != sql.ErrNoRows {
		return nil, false, fmt.Errorf("check active session: %w", err)
	}

	// No active session; create one.
	id := uuid.New().String()
	expires := now.Add(timeout)
	if _, err := r.db.ExecContext(ctx, r.q.SessionInsert, id, entityUUID, processUUID, now, expires, "[]", 0); err != nil {
		return nil, false, fmt.Errorf("create session: %w", err)
	}
	return &store.Session{
		UUID: id, EntityID: entityUUID, ProcessID: processUUID,
		CreatedAt: now, ExpiresAt: expires,
	}, true, nil
}

// ---- ProcessWriter ----

func (r *Repository) UpsertProcess(ctx context.Context, externalID string) (string, error) {
	id := uuid.New().String()
	if _, err := r.db.ExecContext(ctx, r.q.ProcessInsert, id, externalID); err != nil {
		return "", fmt.Errorf("upsert process: %w", err)
	}
	var resultID string
	if err := r.db.QueryRowContext(ctx, r.q.ProcessSelectID, externalID).Scan(&resultID); err != nil {
		return "", fmt.Errorf("read process uuid: %w", err)
	}
	return resultID, nil
}

// ---- ProcessAttributeWriter ----

func (r *Repository) InsertProcessAttribute(ctx context.Context, processUUID string, attr *store.Attribute) error {
	attr.UUID = uuid.New().String()
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.AttrInsert,
		attr.UUID, processUUID, attr.Key, attr.Value, now,
	)
	if err != nil {
		return fmt.Errorf("insert attribute: %w", err)
	}
	return nil
}

// ---- CanonicalSlotStore ----

func (r *Repository) ListCanonicalSlots(ctx context.Context) ([]*store.CanonicalSlot, error) {
	rows, err := r.db.QueryContext(ctx, r.q.SlotCanonicalList)
	if err != nil {
		return nil, fmt.Errorf("list canonical slots: %w", err)
	}
	defer rows.Close()

	var out []*store.CanonicalSlot
	for rows.Next() {
		s := &store.CanonicalSlot{}
		var raw []byte
		var createdAt flexTime
		if err := rows.Scan(&s.Name, &raw, &createdAt); err != nil {
			return nil, fmt.Errorf("scan canonical slot: %w", err)
		}
		s.Embedding = decodeEmbedding(raw)
		if createdAt.Valid {
			s.CreatedAt = createdAt.Time
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func (r *Repository) InsertCanonicalSlot(ctx context.Context, slot *store.CanonicalSlot) error {
	now := time.Now().UTC()
	_, err := r.db.ExecContext(ctx, r.q.SlotCanonicalInsert, slot.Name, encodeEmbedding(slot.Embedding), now)
	if err != nil {
		return fmt.Errorf("insert canonical slot: %w", err)
	}
	return nil
}

func (r *Repository) FindCanonicalSlotByName(ctx context.Context, name string) (*store.CanonicalSlot, error) {
	s := &store.CanonicalSlot{}
	var raw []byte
	var createdAt flexTime
	err := r.db.QueryRowContext(ctx, r.q.SlotCanonicalFindByName, name).Scan(&s.Name, &raw, &createdAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("find canonical slot: %w", err)
	}
	s.Embedding = decodeEmbedding(raw)
	if createdAt.Valid {
		s.CreatedAt = createdAt.Time
	}
	return s, nil
}

// ---- TurnSummaryWriter ----

func (r *Repository) InsertTurnSummary(ctx context.Context, ts *store.TurnSummary) error {
	ts.UUID = uuid.New().String()
	now := time.Now().UTC()
	isOverview := 0
	if ts.IsOverview {
		isOverview = 1
	}
	query := `INSERT INTO mg_turn_summary (uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
	query = r.rewritePlaceholders(query)
	_, err := r.db.ExecContext(ctx, query,
		ts.UUID, ts.ConversationID, ts.EntityID,
		ts.StartTurn, ts.EndTurn, ts.Summary,
		encodeEmbedding(ts.SummaryEmbedding), isOverview, now,
	)
	if err != nil {
		return fmt.Errorf("insert turn summary: %w", err)
	}
	ts.CreatedAt = now
	return nil
}

func (r *Repository) DeleteTurnSummaries(ctx context.Context, conversationID string, uuids []string) error {
	if len(uuids) == 0 {
		return nil
	}
	placeholders := make([]string, len(uuids))
	args := make([]any, 0, len(uuids)+1)
	args = append(args, conversationID)
	for i, id := range uuids {
		placeholders[i] = "?"
		args = append(args, id)
	}
	query := `DELETE FROM mg_turn_summary WHERE conversation_id = ? AND uuid IN (` + strings.Join(placeholders, ",") + `)`
	query = r.rewritePlaceholders(query)
	_, err := r.db.ExecContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("delete turn summaries: %w", err)
	}
	return nil
}

// ---- TurnSummaryReader ----

func (r *Repository) ListTurnSummaries(ctx context.Context, conversationID string) ([]*store.TurnSummary, error) {
	query := `SELECT uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at FROM mg_turn_summary WHERE conversation_id = ? ORDER BY is_overview ASC, start_turn ASC`
	query = r.rewritePlaceholders(query)
	rows, err := r.db.QueryContext(ctx, query, conversationID)
	if err != nil {
		return nil, fmt.Errorf("list turn summaries: %w", err)
	}
	defer rows.Close()

	var out []*store.TurnSummary
	for rows.Next() {
		ts := &store.TurnSummary{}
		var raw []byte
		var isOverview int
		var createdAt flexTime
		if err := rows.Scan(
			&ts.UUID, &ts.ConversationID, &ts.EntityID,
			&ts.StartTurn, &ts.EndTurn, &ts.Summary,
			&raw, &isOverview, &createdAt,
		); err != nil {
			return nil, fmt.Errorf("scan turn summary: %w", err)
		}
		ts.SummaryEmbedding = decodeEmbedding(raw)
		ts.IsOverview = isOverview != 0
		ts.CreatedAt = createdAt.Time
		out = append(out, ts)
	}
	return out, rows.Err()
}

func (r *Repository) CountTurnSummaries(ctx context.Context, conversationID string) (int, error) {
	query := `SELECT COUNT(*) FROM mg_turn_summary WHERE conversation_id = ? AND is_overview = 0`
	query = r.rewritePlaceholders(query)
	var count int
	err := r.db.QueryRowContext(ctx, query, conversationID).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count turn summaries: %w", err)
	}
	return count, nil
}

// ---- ArtifactWriter ----

func (r *Repository) InsertArtifact(ctx context.Context, a *store.Artifact) error {
	a.UUID = uuid.New().String()
	now := time.Now().UTC()
	query := `INSERT INTO mg_artifact (uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
	query = r.rewritePlaceholders(query)
	var supersededBy any
	if a.SupersededBy != "" {
		supersededBy = a.SupersededBy
	}
	_, err := r.db.ExecContext(ctx, query,
		a.UUID, a.ConversationID, a.EntityID, a.Content,
		a.ArtifactType, a.Language, a.Description,
		encodeEmbedding(a.DescriptionEmbedding), supersededBy,
		a.TurnNumber, now,
	)
	if err != nil {
		return fmt.Errorf("insert artifact: %w", err)
	}
	a.CreatedAt = now
	return nil
}

func (r *Repository) SupersedeArtifact(ctx context.Context, oldUUID, newUUID string) error {
	query := `UPDATE mg_artifact SET superseded_by = ? WHERE uuid = ?`
	query = r.rewritePlaceholders(query)
	_, err := r.db.ExecContext(ctx, query, newUUID, oldUUID)
	if err != nil {
		return fmt.Errorf("supersede artifact: %w", err)
	}
	return nil
}

// ---- ArtifactReader ----

func (r *Repository) ListActiveArtifacts(ctx context.Context, entityID, conversationID string) ([]*store.Artifact, error) {
	query := `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = ? AND conversation_id = ? AND superseded_by IS NULL ORDER BY created_at DESC`
	query = r.rewritePlaceholders(query)
	rows, err := r.db.QueryContext(ctx, query, entityID, conversationID)
	if err != nil {
		return nil, fmt.Errorf("list active artifacts: %w", err)
	}
	defer rows.Close()
	return scanArtifacts(rows)
}

func (r *Repository) ListActiveArtifactsByEntity(ctx context.Context, entityID string) ([]*store.Artifact, error) {
	query := `SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at FROM mg_artifact WHERE entity_id = ? AND superseded_by IS NULL ORDER BY created_at DESC`
	query = r.rewritePlaceholders(query)
	rows, err := r.db.QueryContext(ctx, query, entityID)
	if err != nil {
		return nil, fmt.Errorf("list active artifacts by entity: %w", err)
	}
	defer rows.Close()
	return scanArtifacts(rows)
}

func scanArtifacts(rows *sql.Rows) ([]*store.Artifact, error) {
	var out []*store.Artifact
	for rows.Next() {
		a := &store.Artifact{}
		var raw []byte
		var supersededBy sql.NullString
		var createdAt flexTime
		if err := rows.Scan(
			&a.UUID, &a.ConversationID, &a.EntityID, &a.Content,
			&a.ArtifactType, &a.Language, &a.Description,
			&raw, &supersededBy, &a.TurnNumber, &createdAt,
		); err != nil {
			return nil, fmt.Errorf("scan artifact: %w", err)
		}
		a.DescriptionEmbedding = decodeEmbedding(raw)
		if supersededBy.Valid {
			a.SupersededBy = supersededBy.String
		}
		a.CreatedAt = createdAt.Time
		out = append(out, a)
	}
	return out, rows.Err()
}

// ---- SessionMetadataWriter ----

func (r *Repository) UpdateSessionMentions(ctx context.Context, sessionUUID string, mentions []string) error {
	data, err := json.Marshal(mentions)
	if err != nil {
		return fmt.Errorf("marshal mentions: %w", err)
	}
	query := `UPDATE mg_session SET entity_mentions = ? WHERE uuid = ?`
	query = r.rewritePlaceholders(query)
	_, err = r.db.ExecContext(ctx, query, string(data), sessionUUID)
	if err != nil {
		return fmt.Errorf("update session mentions: %w", err)
	}
	return nil
}

func (r *Repository) IncrementSessionMessageCount(ctx context.Context, sessionUUID string) error {
	query := `UPDATE mg_session SET message_count = message_count + 1 WHERE uuid = ?`
	query = r.rewritePlaceholders(query)
	_, err := r.db.ExecContext(ctx, query, sessionUUID)
	if err != nil {
		return fmt.Errorf("increment session message count: %w", err)
	}
	return nil
}

// ---- SessionMetadataReader ----

func (r *Repository) GetSessionMentions(ctx context.Context, sessionUUID string) ([]string, error) {
	query := `SELECT entity_mentions FROM mg_session WHERE uuid = ?`
	query = r.rewritePlaceholders(query)
	var mentionsJSON string
	err := r.db.QueryRowContext(ctx, query, sessionUUID).Scan(&mentionsJSON)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get session mentions: %w", err)
	}
	var mentions []string
	if err := json.Unmarshal([]byte(mentionsJSON), &mentions); err != nil {
		return nil, fmt.Errorf("unmarshal session mentions: %w", err)
	}
	return mentions, nil
}

// ---- io.Closer ----

func (r *Repository) Close() error {
	return r.db.Close()
}
