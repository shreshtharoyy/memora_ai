
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteMemoryStore:

    def __init__(self, db_path: str = "data/memora.db") -> None:
        self.db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    scope TEXT NOT NULL DEFAULT 'general',
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (
                            status IN (
                                'active',
                                'superseded',
                                'invalidated'
                            )
                        ),
                    created_at TEXT NOT NULL,
                    supersedes_id INTEGER
                        REFERENCES memories(id)
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user_slot
                ON memories(user_id, slot, status);

                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_memory(
        self,
        user_id: str,
        slot: str,
        value: str,
        *,
        category: str = "general",
        scope: str = "general",
        source: str = "user",
        confidence: float | None = None,
    ) -> int:
        if not user_id.strip():
            raise ValueError("user_id cannot be empty.")
        if not slot.strip():
            raise ValueError("slot cannot be empty.")
        if not value.strip():
            raise ValueError("value cannot be empty.")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    user_id, slot, value, category, scope,
                    source, confidence, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    user_id,
                    slot,
                    value,
                    category,
                    scope,
                    source,
                    confidence,
                    self._now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to retrieve the inserted memory ID.")

            return cursor.lastrowid

    def get_active_by_slot(
        self,
        user_id: str,
        slot: str,
        *,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT *
            FROM memories
            WHERE user_id = ?
              AND slot = ?
              AND status = 'active'
        """
        params: list[Any] = [user_id, slot]

        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)

        query += " ORDER BY id DESC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def supersede_memory(
        self,
        old_memory_id: int,
        new_value: str,
        *,
        source: str = "user",
        confidence: float | None = None,
    ) -> int:
        """Replace an active fact while preserving its history."""
        if not new_value.strip():
            raise ValueError("new_value cannot be empty.")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        with self._connect() as connection:
            old_memory = connection.execute(
                """
                SELECT *
                FROM memories
                WHERE id = ? AND status = 'active'
                """,
                (old_memory_id,),
            ).fetchone()

            if old_memory is None:
                raise ValueError(
                    f"No active memory found with ID {old_memory_id}."
                )

            connection.execute(
                """
                UPDATE memories
                SET status = 'superseded'
                WHERE id = ? AND status = 'active'
                """,
                (old_memory_id,),
            )

            cursor = connection.execute(
                """
                INSERT INTO memories (
                    user_id, slot, value, category, scope,
                    source, confidence, status, created_at,
                    supersedes_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    old_memory["user_id"],
                    old_memory["slot"],
                    new_value,
                    old_memory["category"],
                    old_memory["scope"],
                    source,
                    confidence,
                    self._now(),
                    old_memory_id,
                ),
            )

            if cursor.lastrowid is None:
                raise RuntimeError("Failed to retrieve the inserted memory ID.")

            return cursor.lastrowid

    def invalidate_memory(self, memory_id: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memories
                SET status = 'invalidated'
                WHERE id = ? AND status = 'active'
                """,
                (memory_id,),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"No active memory found with ID {memory_id}."
                )

    def list_memories(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]

        if not include_inactive:
            query += " AND status = 'active'"

        query += " ORDER BY id DESC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            return [dict(row) for row in rows]
