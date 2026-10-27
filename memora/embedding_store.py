
import sqlite3
from pathlib import Path
import numpy as np

class EmbeddingStore:

    def __init__(
        self,
        db_path: str = "data/memora.db",
        dimensions: int = 384,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self.db_path = db_path
        self.dimensions = dimensions
        self.model_name = model_name

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def get_embeddings(
        self,
        memory_ids: list[int],
    ) -> dict[int, np.ndarray]:

        if not memory_ids:
            return {}

        placeholders = ", ".join("?" for _ in memory_ids)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT memory_id, embedding
                FROM memory_embeddings
                WHERE model_name = ?
                  AND dimensions = ?
                  AND memory_id IN ({placeholders})
                """,
                [self.model_name, self.dimensions, *memory_ids],
            ).fetchall()

        return {
            int(memory_id): np.frombuffer(
                embedding, dtype=np.float32
            ).copy()
            for memory_id, embedding in rows
        }

    def save_embeddings(
        self,
        embeddings: dict[int, np.ndarray],
    ) -> None:

        if not embeddings:
            return

        rows = []

        for memory_id, vector in embeddings.items():
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)

            if vector.size != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"received {vector.size}."
                )

            rows.append(
                (
                    memory_id,
                    self.model_name,
                    self.dimensions,
                    vector.tobytes(),
                )
            )

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO memory_embeddings (
                    memory_id, model_name, dimensions, embedding
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    model_name = excluded.model_name,
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding
                """,
                rows,
            )
