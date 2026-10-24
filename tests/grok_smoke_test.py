
import os

from dotenv import load_dotenv
from groq import Groq


def main() -> None:
    # Load environment variables from the .env file.
    load_dotenv()

    # Read credentials and model configuration.
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    # Stop early if the API key is missing.
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    # Create the Groq API client.
    client = Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=2,
    )

    # Send a test prompt to the selected model.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Memora AI is connected.",
            }
        ],
        max_completion_tokens=50,
    )

    # Display the model and its response.
    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message.content}")

    # Usage information may be None, so check before accessing it.
    usage = response.usage

    if usage is not None:
        print(f"Prompt tokens: {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
    else:
        print("Token usage information was not returned.")


if __name__ == "__main__":
    main()
    