
from memora.hybrid_retriever import HybridMemoryRetriever
from memora.langchain_retriever import HybridLangChainRetriever
from memora.memory_store import SQLiteMemoryStore

def main() -> None:
    store = SQLiteMemoryStore()
    hybrid_retriever = HybridMemoryRetriever(store=store)

    user_id = input("Enter your user ID: ").strip()

    retriever = HybridLangChainRetriever(
        memory_retriever=hybrid_retriever,
        user_id=user_id,
        result_limit=5,
    )

    query = input("Enter your search query: ").strip()

    if not user_id or not query:
        print("User ID and query cannot be empty.")
        return

    documents = retriever.invoke(query)

    if not documents:
        print("No matching memories found.")
        return

    for index, document in enumerate(documents, start=1):
        print(f"\nResult {index}")
        print(f"Memory: {document.page_content}")
        print(f"Metadata: {document.metadata}")


if __name__ == "__main__":
    main()
