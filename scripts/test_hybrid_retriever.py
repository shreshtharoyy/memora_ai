from memora.hybrid_retriever import HybridMemoryRetriever
from memora.memory_store import SQLiteMemoryStore


def main() -> None:
    store = SQLiteMemoryStore()
    retriever = HybridMemoryRetriever(store=store)

    user_id = input("Enter your user ID: ").strip()
    query = input("Enter your search query: ").strip()

    if not user_id or not query:
        print("User ID and search query cannot be empty.")
        return

    print("\nSearching memories...\n")

    results = retriever.search(
        user_id=user_id,
        query=query,
        limit=5,
    )

    if not results:
        print("No matching memories found.")
        return

    for index, memory in enumerate(results, start=1):
        print(f"Result {index}")
        print(f"Memory: {memory['value']}")
        print(f"Slot: {memory['slot']}")
        print(f"Category: {memory['category']}")
        print(f"Retrieval score: {memory['retrieval_score']:.6f}")
        print("-" * 35)


if __name__ == "__main__":
    main()
