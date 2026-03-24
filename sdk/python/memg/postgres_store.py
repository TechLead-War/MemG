"""PostgreSQL store for MemG. Requires psycopg2."""

from __future__ import annotations

import logging
import struct
import uuid as uuid_mod
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from .store import Store
from .types import Conversation, Entity, Fact, FactFilter, Message, Session

logger = logging.getLogger("memg")

_ISO_FMT = "%Y-%m-%dT%H:%M:%S"

_POSTGRES_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS mg_entity (
        uuid TEXT PRIMARY KEY, external_id TEXT NOT NULL UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""",
    """CREATE TABLE IF NOT EXISTS mg_entity_fact (
        uuid TEXT PRIMARY KEY, entity_id TEXT NOT NULL REFERENCES mg_entity(uuid),
        content TEXT NOT NULL, embedding BYTEA,
        created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
        fact_type TEXT NOT NULL DEFAULT 'identity',
        temporal_status TEXT NOT NULL DEFAULT 'current',
        significance INTEGER NOT NULL DEFAULT 5,
        content_key TEXT NOT NULL DEFAULT '',
        reference_time TIMESTAMPTZ, expires_at TIMESTAMPTZ,
        reinforced_at TIMESTAMPTZ, reinforced_count INTEGER NOT NULL DEFAULT 0,
        tag TEXT NOT NULL DEFAULT '', slot TEXT NOT NULL DEFAULT '',
        confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        embedding_model TEXT NOT NULL DEFAULT '',
        source_role TEXT NOT NULL DEFAULT '',
        recall_count INTEGER NOT NULL DEFAULT 0, last_recalled_at TIMESTAMPTZ)""",
    "CREATE INDEX IF NOT EXISTS idx_mg_fact_entity ON mg_entity_fact(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_mg_fact_content_key ON mg_entity_fact(entity_id, content_key)",
    "CREATE INDEX IF NOT EXISTS idx_mg_fact_expires ON mg_entity_fact(entity_id, expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_mg_fact_slot ON mg_entity_fact(entity_id, slot)",
    """CREATE TABLE IF NOT EXISTS mg_session (
        uuid TEXT PRIMARY KEY, entity_id TEXT NOT NULL,
        process_id TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL, expires_at TIMESTAMPTZ NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_mg_session_lookup ON mg_session(entity_id, process_id, expires_at)",
    """CREATE TABLE IF NOT EXISTS mg_conversation (
        uuid TEXT PRIMARY KEY, session_id TEXT NOT NULL,
        entity_id TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
        summary_embedding BYTEA,
        created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_mg_conv_session ON mg_conversation(session_id, created_at)",
    """CREATE TABLE IF NOT EXISTS mg_message (
        uuid TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES mg_conversation(uuid),
        role TEXT NOT NULL, content TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'text', created_at TIMESTAMPTZ NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_mg_msg_conv ON mg_message(conversation_id, created_at)",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid_mod.uuid4())


def _encode(vec: Optional[List[float]]) -> Optional[bytes]:
    if not vec:
        return None
    return struct.pack("<" + "f" * len(vec), *vec)


def _decode(data) -> Optional[List[float]]:
    if data is None:
        return None
    if isinstance(data, memoryview):
        data = bytes(data)
    if not data:
        return None
    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))


def _dt_val(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    return None


_FACT_COLS = "uuid, content, embedding, created_at, updated_at, fact_type, temporal_status, significance, content_key, reference_time, expires_at, reinforced_at, reinforced_count, tag, slot, confidence, embedding_model, source_role, recall_count, last_recalled_at"


class PostgresStore(Store):
    """PostgreSQL store for MemG. Requires psycopg2."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 required. Install with: pip install psycopg2-binary")
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._create_schema()

    def _create_schema(self) -> None:
        with self._conn.cursor() as cur:
            for ddl in _POSTGRES_SCHEMA:
                cur.execute(ddl)

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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO mg_entity (uuid, external_id) VALUES (%s, %s) ON CONFLICT (external_id) DO NOTHING",
                (uid, external_id),
            )
            cur.execute("SELECT uuid FROM mg_entity WHERE external_id = %s", (external_id,))
            return cur.fetchone()[0]

    def lookup_entity(self, external_id: str) -> Optional[Entity]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = %s", (external_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Entity(uuid=row[0], external_id=row[1], created_at=_dt_val(row[2]))

    # ---- Fact ----

    def insert_fact(self, entity_uuid: str, fact: Fact) -> None:
        fact.uuid = _new_uuid()
        now = _now()
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO mg_entity_fact ({_FACT_COLS.replace('uuid,', 'uuid, entity_id,').replace('content,', 'content,')}) VALUES ({', '.join(['%s'] * 21)})",
                (fact.uuid, entity_uuid, fact.content, _encode(fact.embedding),
                 now, now, fact.fact_type, fact.temporal_status, fact.significance,
                 fact.content_key, fact.reference_time, fact.expires_at,
                 fact.reinforced_at, fact.reinforced_count, fact.tag, fact.slot,
                 fact.confidence, fact.embedding_model, fact.source_role,
                 fact.recall_count, fact.last_recalled_at),
            )

    def insert_facts(self, entity_uuid: str, facts: List[Fact]) -> None:
        for f in facts:
            self.insert_fact(entity_uuid, f)

    def _query_facts(self, query: str, args: tuple) -> List[Fact]:
        with self._conn.cursor() as cur:
            cur.execute(query, args)
            return [self._row_to_fact(r) for r in cur.fetchall()]

    def list_facts(self, entity_uuid: str, limit: int = 100) -> List[Fact]:
        return self._query_facts(
            f"SELECT {_FACT_COLS} FROM mg_entity_fact WHERE entity_id = %s ORDER BY created_at DESC LIMIT %s",
            (entity_uuid, limit),
        )

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
            args.append(_now())
        if filt.slots:
            q += f" AND slot IN ({','.join(['%s'] * len(filt.slots))})"
            args.extend(filt.slots)
        if filt.min_confidence > 0:
            q += " AND confidence >= %s"
            args.append(filt.min_confidence)
        if filt.source_roles:
            q += f" AND source_role IN ({','.join(['%s'] * len(filt.source_roles))})"
            args.extend(filt.source_roles)
        if order_by:
            q += f" {order_by}"
        if limit > 0:
            q += " LIMIT %s"
            args.append(limit)
        return q, args

    def list_facts_filtered(self, entity_uuid: str, filt: FactFilter, limit: int = 100) -> List[Fact]:
        q, args = self._build_filter(entity_uuid, filt, limit)
        return self._query_facts(q, tuple(args))

    def list_facts_for_recall(self, entity_uuid: str, filt: FactFilter, limit: int = 10000) -> List[Fact]:
        q, args = self._build_filter(entity_uuid, filt, limit, order_by="")
        return self._query_facts(q, tuple(args))

    def find_fact_by_key(self, entity_uuid: str, content_key: str) -> Optional[Fact]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_FACT_COLS} FROM mg_entity_fact WHERE entity_id = %s AND content_key = %s LIMIT 1",
                (entity_uuid, content_key),
            )
            row = cur.fetchone()
            return self._row_to_fact(row) if row else None

    def reinforce_fact(self, fact_uuid: str, new_expires_at: Optional[datetime] = None) -> None:
        now = _now()
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE mg_entity_fact SET reinforced_at = %s, reinforced_count = reinforced_count + 1, expires_at = %s, updated_at = %s WHERE uuid = %s",
                (now, new_expires_at, now, fact_uuid),
            )

    def update_temporal_status(self, fact_uuid: str, status: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("UPDATE mg_entity_fact SET temporal_status = %s, updated_at = %s WHERE uuid = %s", (status, _now(), fact_uuid))

    def update_significance(self, fact_uuid: str, significance: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute("UPDATE mg_entity_fact SET significance = %s, updated_at = %s WHERE uuid = %s", (significance, _now(), fact_uuid))

    def delete_fact(self, entity_uuid: str, fact_uuid: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM mg_entity_fact WHERE uuid = %s AND entity_id = %s", (fact_uuid, entity_uuid))

    def delete_entity_facts(self, entity_uuid: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM mg_entity_fact WHERE entity_id = %s", (entity_uuid,))
            return cur.rowcount

    def prune_expired_facts(self, entity_uuid: str, now: Optional[datetime] = None) -> int:
        if now is None:
            now = _now()
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM mg_entity_fact WHERE entity_id = %s AND expires_at IS NOT NULL AND expires_at < %s", (entity_uuid, now))
            return cur.rowcount

    def update_recall_usage(self, fact_uuids: List[str]) -> None:
        if not fact_uuids:
            return
        now = _now()
        with self._conn.cursor() as cur:
            for uid in fact_uuids:
                cur.execute("UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = %s WHERE uuid = %s", (now, uid))

    # ---- Session ----

    def ensure_session(self, entity_uuid: str, process_uuid: str, timeout_seconds: int = 1800) -> Tuple[Session, bool]:
        now = _now()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT uuid, entity_id, process_id, created_at, expires_at FROM mg_session WHERE entity_id = %s AND process_id = %s AND expires_at > %s ORDER BY created_at DESC LIMIT 1",
                (entity_uuid, process_uuid, now),
            )
            row = cur.fetchone()
            if row:
                new_exp = now + timedelta(seconds=timeout_seconds)
                cur.execute("UPDATE mg_session SET expires_at = %s WHERE uuid = %s", (new_exp, row[0]))
                return Session(uuid=row[0], entity_id=row[1], process_id=row[2] or "", created_at=_dt_val(row[3]), expires_at=new_exp), False

            uid = _new_uuid()
            expires = now + timedelta(seconds=timeout_seconds)
            cur.execute("INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) VALUES (%s, %s, %s, %s, %s)", (uid, entity_uuid, process_uuid, now, expires))
            return Session(uuid=uid, entity_id=entity_uuid, process_id=process_uuid, created_at=now, expires_at=expires), True

    # ---- Conversation ----

    def start_conversation(self, session_uuid: str, entity_uuid: str = "") -> str:
        uid = _new_uuid()
        now = _now()
        with self._conn.cursor() as cur:
            cur.execute("INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)", (uid, session_uuid, entity_uuid, now, now))
        return uid

    def active_conversation(self, session_uuid: str) -> Optional[Conversation]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT uuid, session_id, entity_id, summary, created_at, updated_at FROM mg_conversation WHERE session_id = %s ORDER BY created_at DESC LIMIT 1", (session_uuid,))
            row = cur.fetchone()
            if not row:
                return None
            return Conversation(uuid=row[0], session_id=row[1], entity_id=row[2] or "", summary=row[3] or "", created_at=_dt_val(row[4]), updated_at=_dt_val(row[5]))

    def append_message(self, conversation_uuid: str, msg: Message) -> None:
        msg.uuid = _new_uuid()
        now = _now()
        with self._conn.cursor() as cur:
            cur.execute("INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) VALUES (%s, %s, %s, %s, %s, %s)", (msg.uuid, conversation_uuid, msg.role, msg.content, msg.kind or "text", now))
            cur.execute("UPDATE mg_conversation SET updated_at = %s WHERE uuid = %s", (now, conversation_uuid))

    def read_messages(self, conversation_uuid: str) -> List[Message]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = %s ORDER BY created_at ASC", (conversation_uuid,))
            return [Message(uuid=r[0], conversation_id=r[1], role=r[2], content=r[3], kind=r[4] or "text", created_at=_dt_val(r[5])) for r in cur.fetchall()]

    def read_recent_messages(self, conversation_uuid: str, limit: int = 20) -> List[Message]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT uuid, conversation_id, role, content, kind, created_at FROM mg_message WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s", (conversation_uuid, limit))
            msgs = [Message(uuid=r[0], conversation_id=r[1], role=r[2], content=r[3], kind=r[4] or "text", created_at=_dt_val(r[5])) for r in cur.fetchall()]
            msgs.reverse()
            return msgs

    def list_conversation_summaries(self, entity_uuid: str, limit: int = 100) -> List[Conversation]:
        q = "SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at FROM mg_conversation WHERE entity_id = %s AND summary != '' AND summary_embedding IS NOT NULL ORDER BY created_at DESC"
        args: list = [entity_uuid]
        if limit > 0:
            q += " LIMIT %s"
            args.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(q, args)
            return [Conversation(uuid=r[0], session_id=r[1], entity_id=r[2] or "", summary=r[3] or "", summary_embedding=_decode(r[4]), created_at=_dt_val(r[5]), updated_at=_dt_val(r[6])) for r in cur.fetchall()]

    def update_conversation_summary(self, conversation_uuid: str, summary: str, embedding: Optional[List[float]] = None) -> None:
        now = _now()
        with self._conn.cursor() as cur:
            cur.execute("UPDATE mg_conversation SET summary = %s, summary_embedding = %s, updated_at = %s WHERE uuid = %s", (summary, _encode(embedding), now, conversation_uuid))

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
