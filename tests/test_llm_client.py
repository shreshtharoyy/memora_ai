
from memora.llm_client import GroqLLMClient

def main() -> None:
    llm = GroqLLMClient()

    result = llm.generate(
        prompt="Reply with exactly: Reusable client works.",
        system_prompt="Follow the user's instructions precisely.",
        max_completion_tokens=50,
    )

    print("Model:", result.model)
    print("Response:", result.text)
    print("Prompt tokens:", result.prompt_tokens)
    print("Completion tokens:", result.completion_tokens)
    print("Total tokens:", result.total_tokens)
    print("Latency (ms):", result.latency_ms)


if __name__ == "__main__":
    main()
