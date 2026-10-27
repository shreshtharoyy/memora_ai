
from memora.fact_extractor import FactExtractor


def main() -> None:
    extractor = FactExtractor()

    conversation = (
        "I prefer Python for AI projects. "
        "I am currently learning LangGraph."
    )

    facts = extractor.extract(conversation)

    for fact in facts:
        print(f"Slot: {fact.slot}")
        print(f"Value: {fact.value}")
        print(f"Category: {fact.category}")
        print(f"Scope: {fact.scope}")
        print(f"Source: {fact.source}")
        print(f"Confidence: {fact.confidence}")
        print("-" * 30)


if __name__ == "__main__":
    main()
