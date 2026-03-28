"""MySQL store for MemG. Requires mysql-connector-python."""

from __future__ import annotations

import json
import logging
import struct
import uuid as uuid_mod
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from .store import Store
from .types import Artifact, Conversation, Entity, Fact, FactFilter, Message, Session, TurnSummary

logger = logging.getLogger("memg")

_MYSQL_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS mg_entity (
        uuid VARCHAR(36) PRIMARY KEY, external_id VARCHAR(255) NOT NULL UNIQUE,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS mg_entity_fact (
        uuid VARCHAR(36) PRIMARY KEY,
        entity_id VARCHAR(36) NOT NULL,
        content TEXT NOT NULL, embedding LONGBLOB,
        created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
        fact_type VARCHAR(50) NOT NULL DEFAULT 'identity',
        temporal_status VARCHAR(50) NOT NULL DEFAULT 'current',
        significance INTEGER NOT NULL DEFAULT 5,
        content_key VARCHAR(255) NOT NULL DEFAULT '',
        reference_time DATETIME, expires_at DATETIME,
        reinforced_at DATETIME, reinforced_count INTEGER NOT NULL DEFAULT 0,
        tag VARCHAR(100) NOT NULL DEFAULT '', slot VARCHAR(100) NOT NULL DEFAULT '',
        confidence DOUBLE NOT NULL DEFAULT 1.0,
        embedding_model VARCHAR(255) NOT NULL DEFAULT '',
        source_role VARCHAR(50) NOT NULL DEFAULT '',
        recall_count INTEGER NOT NULL DEFAULT 0, last_recalled_at DATETIME,
        FOREIGN KEY (entity_id) REFERENCES mg_entity(uuid),
        INDEX idx_mg_fact_entity (entity_id),
        INDEX idx_mg_fact_content_key (entity_id, content_key),
        INDEX idx_mg_fact_expires (entity_id, expires_at),
        INDEX idx_mg_fact_slot (entity_id, slot))""",
    """CREATE TABLE IF NOT EXISTS mg_process (
        uuid VARCHAR(36) PRIMARY KEY, external_id VARCHAR(255) NOT NULL UNIQUE,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS mg_process_attribute (
        uuid VARCHAR(36) PRIMARY KEY,
        process_id VARCHAR(36) NOT NULL,
        `key` VARCHAR(255) NOT NULL, value TEXT NOT NULL,
        created_at DATETIME NOT NULL,
        FOREIGN KEY (process_id) REFERENCES mg_process(uuid))""",
    """CREATE TABLE IF NOT EXISTS mg_session (
        uuid VARCHAR(36) PRIMARY KEY, entity_id VARCHAR(36) NOT NULL,
        process_id VARCHAR(36) NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL, expires_at DATETIME NOT NULL,
        entity_mentions TEXT NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0,
        INDEX idx_mg_session_lookup (entity_id, process_id, expires_at))""",
    """CREATE TABLE IF NOT EXISTS mg_conversation (
        uuid VARCHAR(36) PRIMARY KEY, session_id VARCHAR(36) NOT NULL,
        entity_id VARCHAR(36) NOT NULL DEFAULT '', summary TEXT NOT NULL,
        summary_embedding LONGBLOB,
        summary_embedding_model VARCHAR(255) NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
        INDEX idx_mg_conv_session (session_id, created_at))""",
    """CREATE TABLE IF NOT EXISTS mg_message (
        uuid VARCHAR(36) PRIMARY KEY,
        conversation_id VARCHAR(36) NOT NULL,
        role VARCHAR(50) NOT NULL, content TEXT NOT NULL,
        kind VARCHAR(50) NOT NULL DEFAULT 'text', created_at DATETIME NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES mg_conversation(uuid),
        INDEX idx_mg_msg_conv (conversation_id, created_at))""",
    """CREATE TABLE IF NOT EXISTS mg_turn_summary (
        uuid VARCHAR(36) PRIMARY KEY,
        conversation_id VARCHAR(36) NOT NULL,
        entity_id VARCHAR(36) NOT NULL,
        start_turn INTEGER NOT NULL,
        end_turn INTEGER NOT NULL,
        summary TEXT NOT NULL,
        summary_embedding LONGBLOB,
        is_overview INTEGER NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL,
        INDEX idx_mg_turnsummary_conv (conversation_id, is_overview, created_at))""",
    """CREATE TABLE IF NOT EXISTS mg_artifact (
        uuid VARCHAR(36) PRIMARY KEY,
        conversation_id VARCHAR(36) NOT NULL,
        entity_id VARCHAR(36) NOT NULL,
        content TEXT NOT NULL,
        artifact_type VARCHAR(50) NOT NULL DEFAULT 'code',
        language VARCHAR(50) NOT NULL DEFAULT '',
        description TEXT NOT NULL,
        description_embedding LONGBLOB,
        superseded_by VARCHAR(36),
        turn_number INTEGER NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL,
        INDEX idx_mg_artifact_entity (entity_id, created_at),
        INDEX idx_mg_artifact_conv (conversation_id, created_at))""",
    """CREATE TABLE IF NOT EXISTS mg_slot_canonical (
        name VARCHAR(255) PRIMARY KEY,
        embedding LONGBLOB NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _new_uuid() -> str:
    return str(uuid_mod.uuid4())


def _encode(vec: Optional[List[float]]) -> Optional[bytes]:
    if not vec:
        return None
    return struct.pack("<" + "f" * len(vec), *vec)


def _decode(data) -> Optional[List[float]]:
    if data is None or len(data) == 0:
        return None
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))


def _dt_val(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


_FACT_COLS = "uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at"


class MySQLStore(Store):
    """MySQL 8+ store for MemG. Requires mysql-connector-python."""

    def __init__(self, dsn: str = "", **connect_kwargs) -> None:
        try:
            import mysql.connector
        except ImportError:
            raise ImportError("mysql-connector-python required. Install with: pip install mysql-connector-python")
        if dsn:
            from urllib.parse import urlparse
            u = urlparse(dsn)
            connect_kwargs = {
                "host": u.hostname or "localhost",
                "port": u.port or 3306,
                "user": u.username or "root",
                "password": u.password or "",
                "database": u.path.lstrip("/") if u.path else "",
                **connect_kwargs,
            }
        import threading as _threading
        self._connect_kwargs = {**connect_kwargs, "autocommit": True}
        self._mysql = mysql.connector
        self._local = _threading.local()
        # Create one connection for schema setup.
        self._conn_direct = self._mysql.connect(**self._connect_kwargs)
        self._create_schema()
        self._conn_direct.close()

    @property
    def _conn(self):
        """Thread-local connection. Each thread gets its own."""
        conn = getattr(self._local, "conn", None)
        if conn is None or not conn.is_connected():
            conn = self._mysql.connect(**self._connect_kwargs)
            self._local.conn = conn
        return conn

    def _create_schema(self) -> None:
        cur = self._conn_direct.cursor()
        for ddl in _MYSQL_SCHEMA:
            cur.execute(ddl)
        cur.close()

    def _row_to_fact(self, row: tuple) -> Fact:
        return Fact(
            uuid=row[0], content=row[1], embedding=_decode(row[2]),
            created_at=_dt_val(row[3]), updated_at=_dt_val(row[4]),
            fact_type=row[5] or "identity", temporal_status=row[6] or "current",
            significance=row[7] if row[7] is not None else 5,
            content_key=row[8] or "", reference_time=_dt_val(row[9]),
            expires_at=_dt_val(row[10]), reinforced_at=_dt_val(row[11]),
            reinforced_count=row[12] if row[12] is not None else 0,
            tag=row[13] or "", slot=row[14] or "",
            confidence=row[15] if row[15] is not None else 1.0,
            embedding_model=row[16] or "", source_role=row[17] or "",
            recall_count=row[18] if row[18] is not None else 0,
            last_recalled_at=_dt_val(row[19]),
        )

    # ---- Entity ----

    def upsert_entity(self, external_id: str) -> str:
        uid = _new_uuid()
        cur = self._conn.cursor()
        cur.execute("INSERT IGNORE INTO mg_entity (uuid, external_id) VALUES (%s, %s)", (uid, external_id))
        cur.execute("SELECT uuid FROM mg_entity WHERE external_id = %s", (external_id,))
        result = cur.fetchone()[0]
        cur.close()
        return result

    def lookup_entity(self, external_id: str) -> Optional[Entity]:
        cur = self._conn.cursor()
        cur.execute("SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = %s", (external_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        return Entity(uuid=row[0], external_id=row[1], created_at=_dt_val(row[2]))

    def list_entity_uuids(self, limit: int = 100) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT uuid FROM mg_entity ORDER BY created_at ASC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]

    # ---- Fact metadata ----

    _FACT_METADATA_COLS = "uuid, content, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at"

    def _row_to_fact_metadata(self, row: tuple) -> Fact:
        return Fact(
            uuid=row[0], content=row[1], embedding=None,
            created_at=_dt_val(row[2]), updated_at=_dt_val(row[3]),
            fact_type=row[4] or "identity", temporal_status=row[5] or "current",
            significance=row[6] if row[6] is not None else 5,
            content_key=row[7] or "", reference_time=_dt_val(row[8]),
            expires_at=_dt_val(row[9]), reinforced_at=_dt_val(row[10]),
            reinforced_count=row[11] if row[11] is not None else 0,
            tag=row[12] or "", slot=row[13] or "",
            confidence=row[14] if row[14] is not None else 1.0,
            embedding_model=row[15] or "", source_role=row[16] or "",
            recall_count=row[17] if row[17] is not None else 0,
            last_recalled_at=_dt_val(row[18]),
        )

    def list_facts_metadata(self, entity_uuid: str, filt: FactFilter, limit: int = 100) -> List[Fact]:
        q = f"SELECT {self._FACT_METADATA_COLS} FROM mg_entity_fact WHERE entity_id = %s"
        args: list = [entity_uuid]
        if filt.types:
            q += f" AND fact_type IN ({','.join(['%s'] * len(filt.types))})"
            args.extend(filt.types)
        if filt.statuses:
            q += f" AND temporal_status IN ({','.join(['%s'] * len(filt.statuses))})"
            args.extend(filt.statuses)
        if filt.tags:
            q += f" AND tag IN ({','.join(['%s'] * len(filt.tags))})"
            args.extend(filt.tags)
        if filt.min_significance > 0:
            q += " AND significance >= %s"
            args.append(filt.min_significance)
        if filt.exclude_expired:
            q += " AND (expires_at IS NULL OR expires_at > %s)"
            args.append(_fmt(_now()))
        if filt.slots:
            q += f" AND slot IN ({','.join(['%s'] * len(filt.slots))})"
            args.extend(filt.slots)
        if filt.min_confidence > 0:
            q += " AND confidence >= %s"
            args.append(filt.min_confidence)
        if filt.source_roles:
            q += f" AND source_role IN ({','.join(['%s'] * len(filt.source_roles))})"
            args.extend(filt.source_roles)
        if filt.unembedded_only:
            q += " AND embedding IS NULL"
        if filt.max_significance > 0:
            q += " AND significance <= %s"
            args.append(filt.max_significance)
        if filt.reference_time_after:
            q += " AND reference_time IS NOT NULL AND reference_time >= %s"
            args.append(filt.reference_time_after)
        if filt.reference_time_before:
            q += " AND reference_time IS NOT NULL AND reference_time <= %s"
            args.append(filt.reference_time_before)
        q += " ORDER BY created_at DESC"
        if limit > 0:
            q += " LIMIT %s"
            args.append(limit)
        cur = self._conn.cursor()
        cur.execute(q, tuple(args))
        rows = cur.fetchall()
        cur.close()
        return [self._row_to_fact_metadata(r) for r in rows]

    # ---- Process ----

    def upsert_process(self, external_id: str) -> str:
        uid = _new_uuid()
        cur = self._conn.cursor()
        cur.execute("INSERT IGNORE INTO mg_process (uuid, external_id) VALUES (%s, %s)", (uid, external_id))
        cur.execute("SELECT uuid FROM mg_process WHERE external_id = %s", (external_id,))
        result = cur.fetchone()[0]
        cur.close()
        return result

    def insert_process_attribute(self, process_uuid: str, key: str, value: str) -> None:
        uid = _new_uuid()
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO mg_process_attribute (uuid, process_id, `key`, value, created_at) VALUES (%s, %s, %s, %s, %s)",
            (uid, process_uuid, key, value, now),
        )
        cur.close()

    # ---- Canonical Slots ----

    def list_canonical_slots(self) -> List[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT name, embedding, created_at FROM mg_slot_canonical")
        rows = cur.fetchall()
        cur.close()
        return [
            {"name": r[0], "embedding": _decode(r[1]), "created_at": _dt_val(r[2])}
            for r in rows
        ]

    def insert_canonical_slot(self, name: str, embedding: List[float]) -> None:
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute(
            "INSERT IGNORE INTO mg_slot_canonical (name, embedding, created_at) VALUES (%s, %s, %s)",
            (name, _encode(embedding), now),
        )
        cur.close()

    def find_canonical_slot_by_name(self, name: str) -> Optional[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT name, embedding, created_at FROM mg_slot_canonical WHERE name = %s", (name,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        return {"name": row[0], "embedding": _decode(row[1]), "created_at": _dt_val(row[2])}

    # ---- Fact ----

    def insert_fact(self, entity_uuid: str, fact: Fact) -> None:
        fact.uuid = _new_uuid()
        now = _now()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO mg_entity_fact (uuid, entity_id, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (fact.uuid, entity_uuid, fact.content, _encode(fact.embedding),
             _fmt(now), _fmt(now), fact.fact_type, fact.temporal_status, fact.significance,
             fact.content_key, _fmt(fact.reference_time), _fmt(fact.expires_at),
             _fmt(fact.reinforced_at), fact.reinforced_count, fact.tag, fact.slot,
             fact.confidence, fact.embedding_model, fact.source_role,
             fact.recall_count, _fmt(fact.last_recalled_at)),
        )
        cur.close()

    def insert_facts(self, entity_uuid: str, facts: List[Fact]) -> None:
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            for f in facts:
                self.insert_fact(entity_uuid, f)
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
        finally:
            cur.close()

    def _query_facts(self, query: str, args: tuple) -> List[Fact]:
        cur = self._conn.cursor()
        cur.execute(query, args)
        rows = cur.fetchall()
        cur.close()
        return [self._row_to_fact(r) for r in rows]

    def list_facts(self, entity_uuid: str, limit: int = 100) -> List[Fact]:
        return self._query_facts(f"SELECT {_FACT_COLS} FROM mg_entity_fact WHERE entity_id = %s ORDER BY created_at DESC LIMIT %s", (entity_uuid, limit))

    def _build_filter(self, entity_uuid: str, filt: FactFilter, limit: int, order_by: str = "ORDER BY created_at DESC") -> Tuple[str, list]:
        q = f"SELECT {_FACT_COLS} FROM mg_entity_fact WHERE entity_id = %s"
        args: list = [entity_uuid]
        if filt.types:
            q += f" AND fact_type IN ({','.join(['%s'] * len(filt.types))})"
            args.extend(filt.types)
        if filt.statuses:
            q += f" AND temporal_status IN ({','.join(['%s'] * len(filt.statuses))})"
            args.extend(filt.statuses)
        if filt.tags:
            q += f" AND tag IN ({','.join(['%s'] * len(filt.tags))})"
            args.extend(filt.tags)
        if filt.min_significance > 0:
            q += " AND significance >= %s"
            args.append(filt.min_significance)
        if filt.exclude_expired:
            q += " AND (expires_at IS NULL OR expires_at > %s)"
            args.append(_fmt(_now()))
        if filt.slots:
            q += f" AND slot IN ({','.join(['%s'] * len(filt.slots))})"
            args.extend(filt.slots)
        if filt.min_confidence > 0:
            q += " AND confidence >= %s"
            args.append(filt.min_confidence)
        if filt.source_roles:
            q += f" AND source_role IN ({','.join(['%s'] * len(filt.source_roles))})"
            args.extend(filt.source_roles)
        if filt.unembedded_only:
            q += " AND embedding IS NULL"
        if filt.max_significance > 0:
            q += " AND significance <= %s"
            args.append(filt.max_significance)
        if filt.reference_time_after:
            q += " AND reference_time IS NOT NULL AND reference_time >= %s"
            args.append(filt.reference_time_after)
        if filt.reference_time_before:
            q += " AND reference_time IS NOT NULL AND reference_time <= %s"
            args.append(filt.reference_time_before)
        if order_by:
            q += f" {order_by}"
        if limit > 0:
            q += " LIMIT %s"
            args.append(limit)
        return q, args

    def list_facts_filtered(self, entity_uuid: str, filt: FactFilter, limit: int = 100) -> List[Fact]:
        q, args = self._build_filter(entity_uuid, filt, limit)
        return self._query_facts(q, tuple(args))

    def list_facts_for_recall(self, entity_uuid: str, filt: FactFilter, limit: int = 50) -> List[Fact]:
        q, args = self._build_filter(entity_uuid, filt, limit, order_by="ORDER BY significance DESC, created_at DESC")
        return self._query_facts(q, tuple(args))

    def find_fact_by_key(self, entity_uuid: str, content_key: str) -> Optional[Fact]:
        cur = self._conn.cursor()
        cur.execute(f"SELECT {_FACT_COLS} FROM mg_entity_fact WHERE entity_id = %s AND content_key = %s LIMIT 1", (entity_uuid, content_key))
        row = cur.fetchone()
        cur.close()
        return self._row_to_fact(row) if row else None

    def reinforce_fact(self, fact_uuid: str, new_expires_at: Optional[datetime] = None) -> None:
        now = _now()
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_entity_fact SET reinforced_at = %s, reinforced_count = reinforced_count + 1, expires_at = %s, updated_at = %s WHERE uuid = %s", (_fmt(now), _fmt(new_expires_at), _fmt(now), fact_uuid))
        cur.close()

    def update_temporal_status(self, fact_uuid: str, status: str) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_entity_fact SET temporal_status = %s, updated_at = %s WHERE uuid = %s", (status, _fmt(_now()), fact_uuid))
        cur.close()

    def update_significance(self, fact_uuid: str, significance: int) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_entity_fact SET significance = %s, updated_at = %s WHERE uuid = %s", (significance, _fmt(_now()), fact_uuid))
        cur.close()

    def delete_fact(self, entity_uuid: str, fact_uuid: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM mg_entity_fact WHERE uuid = %s AND entity_id = %s", (fact_uuid, entity_uuid))
        cur.close()

    def delete_entity_facts(self, entity_uuid: str) -> int:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM mg_entity_fact WHERE entity_id = %s", (entity_uuid,))
        count = cur.rowcount
        cur.close()
        return count

    def prune_expired_facts(self, entity_uuid: str, now: Optional[datetime] = None) -> int:
        if now is None:
            now = _now()
        total = 0
        while True:
            cur = self._conn.cursor()
            if entity_uuid:
                cur.execute("SELECT uuid FROM mg_entity_fact WHERE entity_id = %s AND expires_at IS NOT NULL AND expires_at < %s LIMIT 1000", (entity_uuid, _fmt(now)))
            else:
                cur.execute("SELECT uuid FROM mg_entity_fact WHERE expires_at IS NOT NULL AND expires_at < %s LIMIT 1000", (_fmt(now),))
            rows = cur.fetchall()
            if not rows:
                cur.close()
                break
            uuids = [r[0] for r in rows]
            placeholders = ",".join(["%s"] * len(uuids))
            cur.execute(f"DELETE FROM mg_entity_fact WHERE uuid IN ({placeholders})", uuids)
            total += len(uuids)
            cur.close()
            if len(uuids) < 1000:
                break
        return total

    def prune_stale_summaries(self, days: int = 90) -> int:
        cutoff = _fmt(_now() - timedelta(days=days))
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_conversation SET summary = '', summary_embedding = NULL WHERE created_at < %s AND summary != ''", (cutoff,))
        count = cur.rowcount
        cur.close()
        return count

    def update_recall_usage(self, fact_uuids: List[str]) -> None:
        if not fact_uuids:
            return
        now = _fmt(_now())
        cur = self._conn.cursor()
        for uid in fact_uuids:
            cur.execute("UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = %s WHERE uuid = %s", (now, uid))
        cur.close()

    # ---- Session ----

    def ensure_session(self, entity_uuid: str, process_uuid: str, timeout_seconds: int = 1800) -> Tuple[Session, bool]:
        now = _now()
        cur = self._conn.cursor()
        cur.execute("SELECT uuid, entity_id, process_id, created_at, expires_at, entity_mentions, message_count FROM mg_session WHERE entity_id = %s AND process_id = %s AND expires_at > %s ORDER BY created_at DESC LIMIT 1", (entity_uuid, process_uuid, _fmt(now)))
        row = cur.fetchone()
        if row:
            new_exp = now + timedelta(seconds=timeout_seconds)
            cur.execute("UPDATE mg_session SET expires_at = %s WHERE uuid = %s", (_fmt(new_exp), row[0]))
            cur.close()
            mentions = json.loads(row[5]) if row[5] else []
            return Session(uuid=row[0], entity_id=row[1], process_id=row[2] or "", created_at=_dt_val(row[3]), expires_at=new_exp, entity_mentions=mentions, message_count=row[6] if row[6] is not None else 0), False
        uid = _new_uuid()
        expires = now + timedelta(seconds=timeout_seconds)
        cur.execute("INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at, entity_mentions) VALUES (%s, %s, %s, %s, %s, %s)", (uid, entity_uuid, process_uuid, _fmt(now), _fmt(expires), "[]"))
        cur.close()
        return Session(uuid=uid, entity_id=entity_uuid, process_id=process_uuid, created_at=now, expires_at=expires), True

    # ---- Conversation ----

    def start_conversation(self, session_uuid: str, entity_uuid: str = "") -> str:
        uid = _new_uuid()
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute("INSERT INTO mg_conversation (uuid, session_id, entity_id, summary, created_at, updated_at) VALUES (%s, %s, %s, '', %s, %s)", (uid, session_uuid, entity_uuid, now, now))
        cur.close()
        return uid

    def active_conversation(self, session_uuid: str) -> Optional[Conversation]:
        cur = self._conn.cursor()
        cur.execute("SELECT uuid, session_id, entity_id, summary, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE session_id = %s ORDER BY created_at DESC LIMIT 1", (session_uuid,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        return Conversation(uuid=row[0], session_id=row[1], entity_id=row[2] or "", summary=row[3] or "", summary_embedding_model=row[4] or "", created_at=_dt_val(row[5]), updated_at=_dt_val(row[6]))

    def append_message(self, conversation_uuid: str, msg: Message) -> None:
        msg.uuid = _new_uuid()
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute("INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (%s, %s, %s, %s, %s, %s)", (msg.uuid, conversation_uuid, msg.role, msg.content, msg.kind or "text", now))
        cur.execute("UPDATE mg_conversation SET updated_at = %s WHERE uuid = %s", (now, conversation_uuid))
        cur.close()

    def read_messages(self, conversation_uuid: str) -> List[Message]:
        cur = self._conn.cursor()
        cur.execute("SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = %s ORDER BY created_at ASC", (conversation_uuid,))
        rows = cur.fetchall()
        cur.close()
        return [Message(uuid=r[0], conversation_id=r[1], role=r[2], content=r[3], kind=r[4] or "text", created_at=_dt_val(r[5])) for r in rows]

    def read_recent_messages(self, conversation_uuid: str, limit: int = 20) -> List[Message]:
        cur = self._conn.cursor()
        cur.execute("SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s", (conversation_uuid, limit))
        rows = cur.fetchall()
        cur.close()
        msgs = [Message(uuid=r[0], conversation_id=r[1], role=r[2], content=r[3], kind=r[4] or "text", created_at=_dt_val(r[5])) for r in rows]
        msgs.reverse()
        return msgs

    def list_conversation_summaries(self, entity_uuid: str, limit: int = 100) -> List[Conversation]:
        q = "SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at FROM mg_conversation WHERE entity_id = %s AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC"
        args: list = [entity_uuid]
        if limit > 0:
            q += " LIMIT %s"
            args.append(limit)
        cur = self._conn.cursor()
        cur.execute(q, args)
        rows = cur.fetchall()
        cur.close()
        return [Conversation(uuid=r[0], session_id=r[1], entity_id=r[2] or "", summary=r[3] or "", summary_embedding=_decode(r[4]), summary_embedding_model=r[5] or "", created_at=_dt_val(r[6]), updated_at=_dt_val(r[7])) for r in rows]

    def update_conversation_summary(self, conversation_uuid: str, summary: str, embedding: Optional[List[float]] = None, embedding_model: str = "") -> None:
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_conversation SET summary = %s, summary_embedding = %s, summary_embedding_model = %s, updated_at = %s WHERE uuid = %s", (summary, _encode(embedding), embedding_model, now, conversation_uuid))
        cur.close()

    def find_unsummarized_conversation(self, entity_uuid: str, exclude_session_uuid: str) -> Optional[Conversation]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT uuid, session_id, entity_id, summary, summary_embedding, summary_embedding_model, created_at, updated_at "
            "FROM mg_conversation WHERE entity_id = %s AND session_id != %s AND (summary IS NULL OR summary = '') "
            "ORDER BY created_at DESC LIMIT 1",
            (entity_uuid, exclude_session_uuid),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        return Conversation(uuid=row[0], session_id=row[1], entity_id=row[2] or "", summary=row[3] or "", summary_embedding=_decode(row[4]), summary_embedding_model=row[5] or "", created_at=_dt_val(row[6]), updated_at=_dt_val(row[7]))

    def list_unembedded_facts(self, entity_uuid: str, limit: int = 50) -> List[Fact]:
        return self._query_facts(
            f"SELECT {_FACT_COLS} FROM mg_entity_fact WHERE entity_id = %s AND embedding IS NULL ORDER BY created_at DESC LIMIT %s",
            (entity_uuid, limit),
        )

    def update_fact_embedding(self, fact_uuid: str, embedding: List[float], model: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE mg_entity_fact SET embedding = %s, embedding_model = %s, updated_at = %s WHERE uuid = %s",
            (_encode(embedding), model, _fmt(_now()), fact_uuid),
        )
        cur.close()

    # ---- Turn Summary ----

    def insert_turn_summary(self, ts: TurnSummary) -> None:
        ts.uuid = _new_uuid()
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO mg_turn_summary (uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (ts.uuid, ts.conversation_id, ts.entity_id, ts.start_turn, ts.end_turn,
             ts.summary, _encode(ts.summary_embedding),
             1 if ts.is_overview else 0, now),
        )
        cur.close()

    def list_turn_summaries(self, conversation_id: str) -> List[TurnSummary]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT uuid, conversation_id, entity_id, start_turn, end_turn, summary, summary_embedding, is_overview, created_at "
            "FROM mg_turn_summary WHERE conversation_id = %s ORDER BY is_overview ASC, start_turn ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return [
            TurnSummary(
                uuid=r[0], conversation_id=r[1], entity_id=r[2],
                start_turn=r[3], end_turn=r[4], summary=r[5],
                summary_embedding=_decode(r[6]),
                is_overview=bool(r[7]), created_at=_dt_val(r[8]),
            )
            for r in rows
        ]

    def count_turn_summaries(self, conversation_id: str) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM mg_turn_summary WHERE conversation_id = %s AND is_overview = 0",
            (conversation_id,),
        )
        count = cur.fetchone()[0]
        cur.close()
        return count

    def delete_turn_summaries(self, conversation_id: str, uuids: List[str]) -> None:
        if not uuids:
            return
        placeholders = ",".join(["%s"] * len(uuids))
        cur = self._conn.cursor()
        cur.execute(
            f"DELETE FROM mg_turn_summary WHERE conversation_id = %s AND uuid IN ({placeholders})",
            [conversation_id] + uuids,
        )
        cur.close()

    # ---- Artifact ----

    def insert_artifact(self, a: Artifact) -> None:
        a.uuid = _new_uuid()
        now = _fmt(_now())
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO mg_artifact (uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (a.uuid, a.conversation_id, a.entity_id, a.content, a.artifact_type,
             a.language, a.description, _encode(a.description_embedding),
             a.superseded_by, a.turn_number, now),
        )
        cur.close()

    def supersede_artifact(self, old_uuid: str, new_uuid: str) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_artifact SET superseded_by = %s WHERE uuid = %s", (new_uuid, old_uuid))
        cur.close()

    def list_active_artifacts(self, entity_id: str, conversation_id: str) -> List[Artifact]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at "
            "FROM mg_artifact WHERE entity_id = %s AND conversation_id = %s AND superseded_by IS NULL ORDER BY created_at DESC",
            (entity_id, conversation_id),
        )
        rows = cur.fetchall()
        cur.close()
        return [self._row_to_artifact(r) for r in rows]

    def list_active_artifacts_by_entity(self, entity_id: str) -> List[Artifact]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT uuid, conversation_id, entity_id, content, artifact_type, language, description, description_embedding, superseded_by, turn_number, created_at "
            "FROM mg_artifact WHERE entity_id = %s AND superseded_by IS NULL ORDER BY created_at DESC",
            (entity_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return [self._row_to_artifact(r) for r in rows]

    def _row_to_artifact(self, row: tuple) -> Artifact:
        return Artifact(
            uuid=row[0], conversation_id=row[1], entity_id=row[2],
            content=row[3], artifact_type=row[4] or "code",
            language=row[5] or "", description=row[6] or "",
            description_embedding=_decode(row[7]),
            superseded_by=row[8], turn_number=row[9] if row[9] is not None else 0,
            created_at=_dt_val(row[10]),
        )

    # ---- Session metadata ----

    def update_session_mentions(self, session_uuid: str, mentions: List[str]) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_session SET entity_mentions = %s WHERE uuid = %s", (json.dumps(mentions), session_uuid))
        cur.close()

    def increment_session_message_count(self, session_uuid: str) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE mg_session SET message_count = message_count + 1 WHERE uuid = %s", (session_uuid,))
        cur.close()

    def get_session_mentions(self, session_uuid: str) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT entity_mentions FROM mg_session WHERE uuid = %s", (session_uuid,))
        row = cur.fetchone()
        cur.close()
        if row is None or not row[0]:
            return []
        return json.loads(row[0])

    def close(self) -> None:
        try:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
        except Exception:
            pass
