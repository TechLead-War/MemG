"""Store interface and SQLite implementation for MemG."""

from __future__ import annotations

import logging
import os
import sqlite3
import struct
import uuid as uuid_mod
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .schema import SQLITE_SCHEMA
from .types import (
    Conversation,
    Entity,
    Fact,
    FactFilter,
    Message,
    Session,
)

logger = logging.getLogger("memg")

_ISO_FMT = "%Y-%m-%dT%H:%M:%S"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_ISO_FMT)


def _parse_dt(val: Optional[str]) -> Optional[datetime]:
    if val is None or val == "":
        return None
    for fmt in (_ISO_FMT, "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _encode_embedding(vec: Optional[List[float]]) -> Optional[bytes]:
    if vec is None or len(vec) == 0:
        return None
    return struct.pack("<" + "f" * len(vec), *vec)


def _decode_embedding(data: Optional[bytes]) -> Optional[List[float]]:
    if data is None or len(data) == 0:
        return None
    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))


def _new_uuid() -> str:
    return str(uuid_mod.uuid4())


class Store(ABC):
    """Abstract store interface for pluggable persistence backends.

    Implement this to use a custom database (MySQL, Postgres, etc.)
    instead of the default SQLite store.
    """

    # Entity
    @abstractmethod
    def upsert_entity(self, external_id: str) -> str: ...
    @abstractmethod
    def lookup_entity(self, external_id: str) -> Optional[Entity]: ...

    # Fact CRUD
    @abstractmethod
    def insert_fact(self, entity_uuid: str, fact: Fact) -> None: ...
    @abstractmethod
    def insert_facts(self, entity_uuid: str, facts: List[Fact]) -> None: ...
    @abstractmethod
    def list_facts(self, entity_uuid: str, limit: int = 100) -> List[Fact]: ...
    @abstractmethod
    def list_facts_filtered(self, entity_uuid: str, filt: FactFilter, limit: int = 100) -> List[Fact]: ...
    @abstractmethod
    def list_facts_for_recall(self, entity_uuid: str, filt: FactFilter, limit: int = 10000) -> List[Fact]: ...
    @abstractmethod
    def find_fact_by_key(self, entity_uuid: str, content_key: str) -> Optional[Fact]: ...

    # Fact lifecycle
    @abstractmethod
    def reinforce_fact(self, fact_uuid: str, new_expires_at: Optional[datetime] = None) -> None: ...
    @abstractmethod
    def update_temporal_status(self, fact_uuid: str, status: str) -> None: ...
    @abstractmethod
    def update_significance(self, fact_uuid: str, significance: int) -> None: ...
    @abstractmethod
    def delete_fact(self, entity_uuid: str, fact_uuid: str) -> None: ...
    @abstractmethod
    def delete_entity_facts(self, entity_uuid: str) -> int: ...
    @abstractmethod
    def prune_expired_facts(self, entity_uuid: str, now: Optional[datetime] = None) -> int: ...
    @abstractmethod
    def update_recall_usage(self, fact_uuids: List[str]) -> None: ...
    @abstractmethod
    def list_unembedded_facts(self, entity_uuid: str, limit: int = 50) -> List[Fact]: ...
    @abstractmethod
    def update_fact_embedding(self, fact_uuid: str, embedding: List[float], model: str) -> None: ...

    # Session
    @abstractmethod
    def ensure_session(self, entity_uuid: str, process_uuid: str, timeout_seconds: int = 1800) -> Tuple[Session, bool]: ...

    # Conversation
    @abstractmethod
    def start_conversation(self, session_uuid: str, entity_uuid: str = "") -> str: ...
    @abstractmethod
    def active_conversation(self, session_uuid: str) -> Optional[Conversation]: ...
    @abstractmethod
    def append_message(self, conversation_uuid: str, msg: Message) -> None: ...
    @abstractmethod
    def read_messages(self, conversation_uuid: str) -> List[Message]: ...
    @abstractmethod
    def read_recent_messages(self, conversation_uuid: str, limit: int = 20) -> List[Message]: ...
    @abstractmethod
    def list_conversation_summaries(self, entity_uuid: str, limit: int = 100) -> List[Conversation]: ...
    @abstractmethod
    def update_conversation_summary(self, conversation_uuid: str, summary: str, embedding: Optional[List[float]] = None) -> None: ...
    @abstractmethod
    def find_unsummarized_conversation(self, entity_uuid: str, exclude_session_uuid: str) -> Optional[Conversation]: ...

    # Lifecycle
    @abstractmethod
    def close(self) -> None: ...


class SQLiteStore(Store):
    """SQLite-backed repository for MemG.

    Thread-safe: uses WAL mode and check_same_thread=False.
    """

    def __init__(self, db_path: str) -> None:
        db_path = os.path.expanduser(db_path)
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_schema()

    def _create_schema(self) -> None:
        for statement in SQLITE_SCHEMA.strip().split(";"):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)

    # ---- Entity ----

    def upsert_entity(self, external_id: str) -> str:
        uid = _new_uuid()
        self._conn.execute(
            "INSERT OR IGNORE INTO mg_entity (uuid, external_id) VALUES (?, ?)",
            (uid, external_id),
        )
        row = self._conn.execute(
            "SELECT uuid FROM mg_entity WHERE external_id = ?",
            (external_id,),
        ).fetchone()
        return row[0]

    def lookup_entity(self, external_id: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT uuid, external_id, created_at FROM mg_entity WHERE external_id = ?",
            (external_id,),
        ).fetchone()
        if row is None:
            return None
        return Entity(
            uuid=row[0],
            external_id=row[1],
            created_at=_parse_dt(row[2]),
        )

    # ---- Fact ----

    def insert_fact(self, entity_uuid: str, fact: Fact) -> None:
        fact.uuid = _new_uuid()
        now = _now_iso()
        self._conn.execute(
            """INSERT INTO mg_entity_fact
            (uuid, entity_id, content, embedding, created_at, updated_at,
             fact_type, temporal_status, significance, content_key,
             reference_time, expires_at, reinforced_at, reinforced_count,
             tag, slot, confidence, embedding_model, source_role,
             recall_count, last_recalled_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fact.uuid, entity_uuid, fact.content,
                _encode_embedding(fact.embedding),
                now, now,
                fact.fact_type, fact.temporal_status, fact.significance,
                fact.content_key,
                fact.reference_time.strftime(_ISO_FMT) if fact.reference_time else None,
                fact.expires_at.strftime(_ISO_FMT) if fact.expires_at else None,
                fact.reinforced_at.strftime(_ISO_FMT) if fact.reinforced_at else None,
                fact.reinforced_count, fact.tag, fact.slot, fact.confidence,
                fact.embedding_model, fact.source_role,
                fact.recall_count,
                fact.last_recalled_at.strftime(_ISO_FMT) if fact.last_recalled_at else None,
            ),
        )

    def insert_facts(self, entity_uuid: str, facts: List[Fact]) -> None:
        for f in facts:
            self.insert_fact(entity_uuid, f)

    def _row_to_fact(self, row: tuple) -> Fact:
        return Fact(
            uuid=row[0],
            content=row[1],
            embedding=_decode_embedding(row[2]),
            created_at=_parse_dt(row[3]),
            updated_at=_parse_dt(row[4]),
            fact_type=row[5] or "identity",
            temporal_status=row[6] or "current",
            significance=row[7] if row[7] is not None else 5,
            content_key=row[8] or "",
            reference_time=_parse_dt(row[9]),
            expires_at=_parse_dt(row[10]),
            reinforced_at=_parse_dt(row[11]),
            reinforced_count=row[12] if row[12] is not None else 0,
            tag=row[13] or "",
            slot=row[14] or "",
            confidence=row[15] if row[15] is not None else 1.0,
            embedding_model=row[16] or "",
            source_role=row[17] or "",
            recall_count=row[18] if row[18] is not None else 0,
            last_recalled_at=_parse_dt(row[19]),
        )

    _FACT_COLUMNS = (
        "uuid, content, embedding, created_at, updated_at, "
        "fact_type, temporal_status, significance, content_key, "
        "reference_time, expires_at, reinforced_at, reinforced_count, "
        "tag, slot, confidence, embedding_model, source_role, "
        "recall_count, last_recalled_at"
    )

    def list_facts(self, entity_uuid: str, limit: int = 100) -> List[Fact]:
        rows = self._conn.execute(
            f"SELECT {self._FACT_COLUMNS} FROM mg_entity_fact "
            "WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?",
            (entity_uuid, limit),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def _build_filter_query(
        self, entity_uuid: str, filt: FactFilter, limit: int, order_by: str = "ORDER BY created_at DESC"
    ) -> Tuple[str, list]:
        query = f"SELECT {self._FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ?"
        args: list = [entity_uuid]

        if filt.types:
            placeholders = ",".join("?" for _ in filt.types)
            query += f" AND fact_type IN ({placeholders})"
            args.extend(filt.types)
        if filt.statuses:
            placeholders = ",".join("?" for _ in filt.statuses)
            query += f" AND temporal_status IN ({placeholders})"
            args.extend(filt.statuses)
        if filt.tags:
            placeholders = ",".join("?" for _ in filt.tags)
            query += f" AND tag IN ({placeholders})"
            args.extend(filt.tags)
        if filt.min_significance > 0:
            query += " AND significance >= ?"
            args.append(filt.min_significance)
        if filt.exclude_expired:
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            args.append(_now_iso())
        if filt.slots:
            placeholders = ",".join("?" for _ in filt.slots)
            query += f" AND slot IN ({placeholders})"
            args.extend(filt.slots)
        if filt.min_confidence > 0:
            query += " AND confidence >= ?"
            args.append(filt.min_confidence)
        if filt.source_roles:
            placeholders = ",".join("?" for _ in filt.source_roles)
            query += f" AND source_role IN ({placeholders})"
            args.extend(filt.source_roles)

        query += f" {order_by}"
        if limit > 0:
            query += " LIMIT ?"
            args.append(limit)
        return query, args

    def list_facts_filtered(self, entity_uuid: str, filt: FactFilter, limit: int = 100) -> List[Fact]:
        query, args = self._build_filter_query(entity_uuid, filt, limit)
        rows = self._conn.execute(query, args).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def list_facts_for_recall(self, entity_uuid: str, filt: FactFilter, limit: int = 10000) -> List[Fact]:
        query, args = self._build_filter_query(entity_uuid, filt, limit, order_by="")
        rows = self._conn.execute(query, args).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def find_fact_by_key(self, entity_uuid: str, content_key: str) -> Optional[Fact]:
        row = self._conn.execute(
            f"SELECT {self._FACT_COLUMNS} FROM mg_entity_fact "
            "WHERE entity_id = ? AND content_key = ? LIMIT 1",
            (entity_uuid, content_key),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_fact(row)

    def reinforce_fact(self, fact_uuid: str, new_expires_at: Optional[datetime] = None) -> None:
        now = _now_iso()
        expires_val = new_expires_at.strftime(_ISO_FMT) if new_expires_at else None
        self._conn.execute(
            "UPDATE mg_entity_fact SET reinforced_at = ?, reinforced_count = reinforced_count + 1, "
            "expires_at = ?, updated_at = ? WHERE uuid = ?",
            (now, expires_val, now, fact_uuid),
        )

    def update_temporal_status(self, fact_uuid: str, status: str) -> None:
        now = _now_iso()
        self._conn.execute(
            "UPDATE mg_entity_fact SET temporal_status = ?, updated_at = ? WHERE uuid = ?",
            (status, now, fact_uuid),
        )

    def update_significance(self, fact_uuid: str, significance: int) -> None:
        now = _now_iso()
        self._conn.execute(
            "UPDATE mg_entity_fact SET significance = ?, updated_at = ? WHERE uuid = ?",
            (significance, now, fact_uuid),
        )

    def update_fact_embedding(self, fact_uuid: str, embedding: List[float], model: str) -> None:
        now = _now_iso()
        self._conn.execute(
            "UPDATE mg_entity_fact SET embedding = ?, embedding_model = ?, updated_at = ? WHERE uuid = ?",
            (_encode_embedding(embedding), model, now, fact_uuid),
        )

    def delete_fact(self, entity_uuid: str, fact_uuid: str) -> None:
        self._conn.execute(
            "DELETE FROM mg_entity_fact WHERE uuid = ? AND entity_id = ?",
            (fact_uuid, entity_uuid),
        )

    def delete_entity_facts(self, entity_uuid: str) -> int:
        cursor = self._conn.execute(
            "DELETE FROM mg_entity_fact WHERE entity_id = ?",
            (entity_uuid,),
        )
        return cursor.rowcount

    def prune_expired_facts(self, entity_uuid: str, now: Optional[datetime] = None) -> int:
        if now is None:
            now = datetime.now(timezone.utc)
        now_str = now.strftime(_ISO_FMT)
        cursor = self._conn.execute(
            "DELETE FROM mg_entity_fact WHERE entity_id = ? AND expires_at IS NOT NULL AND expires_at < ?",
            (entity_uuid, now_str),
        )
        return cursor.rowcount

    def update_recall_usage(self, fact_uuids: List[str]) -> None:
        if not fact_uuids:
            return
        now = _now_iso()
        for uid in fact_uuids:
            self._conn.execute(
                "UPDATE mg_entity_fact SET recall_count = recall_count + 1, last_recalled_at = ? WHERE uuid = ?",
                (now, uid),
            )

    def list_unembedded_facts(self, entity_uuid: str, limit: int = 50) -> List[Fact]:
        rows = self._conn.execute(
            f"SELECT {self._FACT_COLUMNS} FROM mg_entity_fact WHERE entity_id = ? AND embedding IS NULL ORDER BY created_at DESC LIMIT ?",
            (entity_uuid, limit),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ---- Session ----

    def ensure_session(
        self, entity_uuid: str, process_uuid: str, timeout_seconds: int = 1800
    ) -> Tuple[Session, bool]:
        now = datetime.now(timezone.utc)
        now_str = now.strftime(_ISO_FMT)

        row = self._conn.execute(
            "SELECT uuid, entity_id, process_id, created_at, expires_at "
            "FROM mg_session WHERE entity_id = ? AND process_id = ? AND expires_at > ? "
            "ORDER BY created_at DESC LIMIT 1",
            (entity_uuid, process_uuid, now_str),
        ).fetchone()

        if row is not None:
            from datetime import timedelta
            new_expires = now + timedelta(seconds=timeout_seconds)
            new_expires_str = new_expires.strftime(_ISO_FMT)
            self._conn.execute(
                "UPDATE mg_session SET expires_at = ? WHERE uuid = ?",
                (new_expires_str, row[0]),
            )
            session = Session(
                uuid=row[0],
                entity_id=row[1],
                process_id=row[2] or "",
                created_at=_parse_dt(row[3]),
                expires_at=new_expires,
            )
            return session, False

        from datetime import timedelta
        uid = _new_uuid()
        expires = now + timedelta(seconds=timeout_seconds)
        self._conn.execute(
            "INSERT INTO mg_session (uuid, entity_id, process_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, entity_uuid, process_uuid, now_str, expires.strftime(_ISO_FMT)),
        )
        session = Session(
            uuid=uid,
            entity_id=entity_uuid,
            process_id=process_uuid,
            created_at=now,
            expires_at=expires,
        )
        return session, True

    # ---- Conversation ----

    def start_conversation(self, session_uuid: str, entity_uuid: str = "") -> str:
        uid = _new_uuid()
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO mg_conversation (uuid, session_id, entity_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, session_uuid, entity_uuid, now, now),
        )
        return uid

    def active_conversation(self, session_uuid: str) -> Optional[Conversation]:
        row = self._conn.execute(
            "SELECT uuid, session_id, entity_id, summary, created_at, updated_at "
            "FROM mg_conversation WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_uuid,),
        ).fetchone()
        if row is None:
            return None
        return Conversation(
            uuid=row[0],
            session_id=row[1],
            entity_id=row[2] or "",
            summary=row[3] or "",
            created_at=_parse_dt(row[4]),
            updated_at=_parse_dt(row[5]),
        )

    def append_message(self, conversation_uuid: str, msg: Message) -> None:
        msg.uuid = _new_uuid()
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO mg_message (uuid, conversation_id, role, content, kind, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (msg.uuid, conversation_uuid, msg.role, msg.content, msg.kind or "text", now),
        )
        self._conn.execute(
            "UPDATE mg_conversation SET updated_at = ? WHERE uuid = ?",
            (now, conversation_uuid),
        )

    def read_messages(self, conversation_uuid: str) -> List[Message]:
        rows = self._conn.execute(
            "SELECT uuid, conversation_id, role, content, kind, created_at "
            "FROM mg_message WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_uuid,),
        ).fetchall()
        return [
            Message(
                uuid=r[0],
                conversation_id=r[1],
                role=r[2],
                content=r[3],
                kind=r[4] or "text",
                created_at=_parse_dt(r[5]),
            )
            for r in rows
        ]

    def read_recent_messages(self, conversation_uuid: str, limit: int = 20) -> List[Message]:
        rows = self._conn.execute(
            "SELECT uuid, conversation_id, role, content, kind, created_at "
            "FROM mg_message WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
            (conversation_uuid, limit),
        ).fetchall()
        messages = [
            Message(
                uuid=r[0],
                conversation_id=r[1],
                role=r[2],
                content=r[3],
                kind=r[4] or "text",
                created_at=_parse_dt(r[5]),
            )
            for r in rows
        ]
        messages.reverse()
        return messages

    def list_conversation_summaries(self, entity_uuid: str, limit: int = 100) -> List[Conversation]:
        query = (
            "SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at "
            "FROM mg_conversation WHERE entity_id = ? AND summary != '' AND summary_embedding IS NOT NULL "
            "ORDER BY created_at DESC"
        )
        args: list = [entity_uuid]
        if limit > 0:
            query += " LIMIT ?"
            args.append(limit)
        rows = self._conn.execute(query, args).fetchall()
        return [
            Conversation(
                uuid=r[0],
                session_id=r[1],
                entity_id=r[2] or "",
                summary=r[3] or "",
                summary_embedding=_decode_embedding(r[4]),
                created_at=_parse_dt(r[5]),
                updated_at=_parse_dt(r[6]),
            )
            for r in rows
        ]

    def update_conversation_summary(
        self, conversation_uuid: str, summary: str, embedding: Optional[List[float]] = None
    ) -> None:
        now = _now_iso()
        self._conn.execute(
            "UPDATE mg_conversation SET summary = ?, summary_embedding = ?, updated_at = ? WHERE uuid = ?",
            (summary, _encode_embedding(embedding), now, conversation_uuid),
        )

    def find_unsummarized_conversation(self, entity_uuid: str, exclude_session_uuid: str) -> Optional[Conversation]:
        row = self._conn.execute(
            "SELECT uuid, session_id, entity_id, summary, summary_embedding, created_at, updated_at "
            "FROM mg_conversation "
            "WHERE entity_id = ? AND session_id != ? AND (summary IS NULL OR summary = '') "
            "ORDER BY created_at DESC LIMIT 1",
            (entity_uuid, exclude_session_uuid),
        ).fetchone()
        if not row:
            return None
        return Conversation(
            uuid=row[0], session_id=row[1], entity_id=row[2] or "",
            summary=row[3] or "", summary_embedding=_decode_embedding(row[4]),
            created_at=_parse_dt(row[5]), updated_at=_parse_dt(row[6]),
        )

    # ---- Close ----

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
