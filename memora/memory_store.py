
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
            connection.execute(
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
                        CHECK (status IN (
                            'active', 'superseded', 'invalidated'
                        )),
                    created_at TEXT NOT NULL,
                    supersedes_id INTEGER REFERENCES memories(id)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_slot_status
                ON memories(user_id, slot, status)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at)
                """
            )

            # FTS5 index for full-text memory retrieval.
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(
                    slot,
                    value,
                    category,
                    scope,
                    content='memories',
                    content_rowid='id'
                )
                """
            )

            # Keep the FTS5 index synchronized with the memories table.
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai
                AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memories_fts(
                        rowid, slot, value, category, scope
                    )
                    VALUES (
                        new.id, new.slot, new.value,
                        new.category, new.scope
                    );
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ad
                AFTER DELETE ON memories
                BEGIN
                    INSERT INTO memories_fts(
                        memories_fts, rowid, slot, value, category, scope
                    )
                    VALUES (
                        'delete', old.id, old.slot, old.value,
                        old.category, old.scope
                    );
                END;

                CREATE TRIGGER IF NOT EXISTS memories_au
                AFTER UPDATE OF slot, value, category, scope ON memories
                BEGIN
                    INSERT INTO memories_fts(
                        memories_fts, rowid, slot, value, category, scope
                    )
                    VALUES (
                        'delete', old.id, old.slot, old.value,
                        old.category, old.scope
                    );

                    INSERT INTO memories_fts(
                        rowid, slot, value, category, scope
                    )
                    VALUES (
                        new.id, new.slot, new.value,
                        new.category, new.scope
                    );
                END;
                """
            )

            # Populate the index for records that existed before FTS5 was added.
            connection.execute(
                """
                INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        if self.db_path != ":memory:":
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
                    user_id, slot, value, category, scope, source,
                    confidence, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    user_id, slot, value, category, scope, source,
                    confidence, self._now(),
                ),
            )

            if cursor.lastrowid is None:
                raise RuntimeError("Failed to retrieve the inserted memory ID.")

            return cursor.lastrowid

    def get_active_by_slot(
        self,
        user_id: str,
        slot: str,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM memories
            WHERE user_id = ? AND slot = ? AND status = 'active'
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
        source: str = "user",
        confidence: float | None = None,
    ) -> int:
        if not new_value.strip():
            raise ValueError("new_value cannot be empty.")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        with self._connect() as connection:
            old_memory = connection.execute(
                """
                SELECT * FROM memories
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
                WHERE id = ?
                """,
                (old_memory_id,),
            )

            cursor = connection.execute(
                """
                INSERT INTO memories (
                    user_id, slot, value, category, scope, source,
                    confidence, status, created_at, supersedes_id
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

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:

        if not user_id.strip():
            raise ValueError("user_id cannot be empty.")
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        # Convert ordinary query text into safely quoted FTS5 terms.
        terms = query.split()
        fts_query = " OR ".join(
            '"' + term.replace('"', '""') + '"'
            for term in terms
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memories.*, bm25(memories_fts) AS search_rank
                FROM memories_fts
                JOIN memories ON memories.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                  AND memories.user_id = ?
                  AND memories.status = 'active'
                ORDER BY search_rank ASC, memories.id DESC
                LIMIT ?
                """,
                (fts_query, user_id, limit),
            ).fetchall()

            return [dict(row) for row in rows]
