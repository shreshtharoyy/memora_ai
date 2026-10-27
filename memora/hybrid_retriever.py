
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from memora.embedding_store import EmbeddingStore
from memora.memory_store import SQLiteMemoryStore


class HybridMemoryRetriever:

    def __init__(
        self,
        store: SQLiteMemoryStore | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        self.store = (
            store if store is not None else SQLiteMemoryStore()
        )
        self.model_name = model_name
        self.rrf_k = rrf_k

        # Force CPU inference; no dedicated GPU is required.
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        self.embedding_store = EmbeddingStore(
            db_path=self.store.db_path,
            dimensions=384,
            model_name=model_name,
        )

    @staticmethod
    def _memory_text(memory: dict[str, Any]) -> str:
        """Convert a memory record into searchable text."""
        return (
            f"Slot: {memory['slot']}. "
            f"Value: {memory['value']}. "
            f"Category: {memory['category']}. "
            f"Scope: {memory['scope']}."
        )

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Combine FTS5 and semantic rankings using RRF."""

        if not user_id.strip():
            raise ValueError("user_id cannot be empty.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit <= 0:
            raise ValueError("limit must be positive.")

        # 1. Retrieve keyword matches from the existing FTS5 search.
        keyword_results = self.store.search_memories(
            user_id=user_id,
            query=query,
            limit=max(limit * 3, 10),
        )

        # 2. Get this user's active memories for semantic retrieval.
        memories = self.store.list_memories(
            user_id=user_id,
            include_inactive=False,
        )

        semantic_results: list[dict[str, Any]] = []

        if memories:
            memory_ids = [int(memory["id"]) for memory in memories]

            # 3. Load cached embeddings from SQLite.
            cached = self.embedding_store.get_embeddings(memory_ids)

            # 4. Generate embeddings only for uncached memories.
            missing = [
                memory
                for memory in memories
                if int(memory["id"]) not in cached
            ]

            for start in range(0, len(missing), 8):
                batch = missing[start : start + 8]
                texts = [self._memory_text(memory) for memory in batch]

                vectors = self.model.encode(
                    texts,
                    batch_size=8,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                new_embeddings = {
                    int(memory["id"]): np.asarray(
                        vector, dtype=np.float32
                    )
                    for memory, vector in zip(batch, vectors)
                }

                self.embedding_store.save_embeddings(new_embeddings)
                cached.update(new_embeddings)

            # 5. Embed the query and compare it with cached vectors.
            query_embedding = self.model.encode(
                query,
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            matrix = np.stack([
                cached[memory_id] for memory_id in memory_ids
            ])

            similarities = matrix @ query_embedding
            ranked_indices = np.argsort(-similarities)

            semantic_results = [
                memories[int(index)]
                for index in ranked_indices[: max(limit * 3, 10)]
            ]

        # 6. Combine the keyword and semantic rankings with RRF.
        combined: dict[int, dict[str, Any]] = {}
        scores: dict[int, float] = {}

        for ranked_list in (keyword_results, semantic_results):
            for rank, memory in enumerate(ranked_list, start=1):
                memory_id = int(memory["id"])
                combined[memory_id] = memory

                scores[memory_id] = scores.get(memory_id, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )

        # 7. Return the highest-ranked unique memories.
        ranked_ids = sorted(
            scores,
            key=lambda memory_id: scores[memory_id],
            reverse=True,
        )

        results = []

        for memory_id in ranked_ids[:limit]:
            result = dict(combined[memory_id])
            result["retrieval_score"] = round(scores[memory_id], 6)
            results.append(result)

        return results
