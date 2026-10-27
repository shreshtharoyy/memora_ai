
from memora.memory_pipeline import MemoryIngestionPipeline
from memora.memory_store import SQLiteMemoryStore


def main() -> None:
    store = SQLiteMemoryStore("data/pipeline_test.db")
    pipeline = MemoryIngestionPipeline(store=store)

    conversation = (
        "I prefer Python for AI projects. "
        "I am currently learning LangGraph."
    )

    memory_ids = pipeline.ingest(
        user_id="pipeline_test_user",
        conversation=conversation,
    )

    print(f"Saved memory IDs: {memory_ids}")

    memories = store.list_memories("pipeline_test_user")

    for memory in memories:
        print(memory)


def test_search_memories_finds_matching_active_memory(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "search.db"))
    store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
        category="preference",
    )
    store.add_memory(
        user_id="user_1",
        slot="favourite_food",
        value="Pizza",
        category="preference",
    )

    results = store.search_memories("user_1", "Python")

    assert len(results) == 1
    assert results[0]["value"] == "Python"


def test_search_memories_does_not_return_another_users_data(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "search_users.db"))
    store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
    )
    store.add_memory(
        user_id="user_2",
        slot="preferred_language",
        value="Python",
    )

    results = store.search_memories("user_1", "Python")

    assert len(results) == 1
    assert results[0]["user_id"] == "user_1"


def test_search_memories_excludes_superseded_records(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "search_updates.db"))
    old_id = store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
    )
    store.supersede_memory(old_id, "TypeScript")

    results = store.search_memories("user_1", "Python")

    assert all(memory["status"] == "active" for memory in results)
    assert all(memory["value"] != "Python" for memory in results)


if __name__ == "__main__":
    main()
