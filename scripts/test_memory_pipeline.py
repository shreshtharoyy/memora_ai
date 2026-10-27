
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


if __name__ == "__main__":
    main()
