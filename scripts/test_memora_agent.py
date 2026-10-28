
def main() -> None:
    user_id = input("Enter your user ID: ").strip()
    if not user_id:
        print("User ID cannot be empty.")
        return

    print("Initializing Memora agent...")
    from memora.agent import MemoraAgent

    agent = MemoraAgent(user_id=user_id)

    print("\nMemora is ready. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            result = agent.invoke(query)
            print(f"\nMemora: {result['answer']}")
        except Exception as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
