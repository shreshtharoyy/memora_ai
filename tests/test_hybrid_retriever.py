
import numpy as np

from memora.hybrid_retriever import HybridMemoryRetriever
from memora.memory_store import SQLiteMemoryStore



class FakeModel:
    """Lightweight test double; does not load a real ML model."""

    def __init__(self, *args, **kwargs):
        self.encode_calls = 0

    def encode(
        self,
        sentences,
        batch_size=32,
        normalize_embeddings=False,
        convert_to_numpy=False,
        show_progress_bar=False,
    ):
        self.encode_calls += 1

        single_input = isinstance(sentences, str)

        if single_input:
            sentences = [sentences]

        vectors = []

        for sentence in sentences:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = (
                1.0 if "python" in sentence.lower() else 0.0
            )
            vector[1] = (
                1.0 if "language" in sentence.lower() else 0.0
            )

            norm = np.linalg.norm(vector)

            if normalize_embeddings and norm > 0:
                vector = vector / norm

            vectors.append(vector)

        result = np.asarray(vectors, dtype=np.float32)

        # A string input returns one vector.
        # A list input always returns a 2D array, even for one item.
        return result[0] if single_input else result

def test_hybrid_search_returns_only_the_requested_users_memories(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "memora.hybrid_retriever.SentenceTransformer",
        FakeModel,
    )

    store = SQLiteMemoryStore(str(tmp_path / "hybrid_test.db"))

    store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
        category="preference",
    )

    store.add_memory(
        user_id="user_2",
        slot="preferred_language",
        value="Python",
        category="preference",
    )

    retriever = HybridMemoryRetriever(store=store)

    results = retriever.search(
        user_id="user_1",
        query="Python",
        limit=5,
    )

    assert results
    assert all(item["user_id"] == "user_1" for item in results)
    assert any(item["value"] == "Python" for item in results)
    assert all("retrieval_score" in item for item in results)


def test_embeddings_are_cached_between_searches(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "memora.hybrid_retriever.SentenceTransformer",
        FakeModel,
    )

    store = SQLiteMemoryStore(str(tmp_path / "cache_test.db"))

    store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
        category="preference",
    )

    retriever = HybridMemoryRetriever(store=store)

    retriever.search("user_1", "Python")
    calls_after_first_search = retriever.model.encode_calls

    retriever.search("user_1", "Python")
    calls_after_second_search = retriever.model.encode_calls

    # First search encodes the memory and the query.
    # Second search should encode only the query.
    assert calls_after_first_search == 2
    assert calls_after_second_search == 3

    cached = retriever.embedding_store.get_embeddings([1])
    assert 1 in cached
    assert cached[1].shape == (384,)
