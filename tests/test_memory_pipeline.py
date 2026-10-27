
from pathlib import Path

from memora.fact_extractor import FactCandidate
from memora.memory_pipeline import MemoryIngestionPipeline
from memora.memory_store import SQLiteMemoryStore


class FakeFactExtractor:
    """Return predefined facts without calling an LLM."""

    def __init__(self, facts: list[FactCandidate]) -> None:
        self.facts = facts

    def extract(self, conversation: str) -> list[FactCandidate]:
        return self.facts


def make_fact(slot: str, value: str) -> FactCandidate:
    return FactCandidate(
        slot=slot,
        value=value,
        category="preference",
        scope="general",
        source="user",
        confidence=0.95,
    )


def make_pipeline(
    tmp_path: Path,
    facts: list[FactCandidate],
) -> tuple[MemoryIngestionPipeline, SQLiteMemoryStore, FakeFactExtractor]:
    store = SQLiteMemoryStore(str(tmp_path / "test_memory.db"))
    extractor = FakeFactExtractor(facts)
    pipeline = MemoryIngestionPipeline(extractor=extractor, store=store)
    return pipeline, store, extractor


def test_adds_new_memory(tmp_path: Path) -> None:
    pipeline, store, _ = make_pipeline(
        tmp_path,
        [make_fact("preferred_language", "Python")],
    )

    memory_ids = pipeline.ingest("user_1", "I prefer Python.")

    memories = store.list_memories("user_1")

    assert len(memory_ids) == 1
    assert len(memories) == 1
    assert memories[0]["value"] == "Python"
    assert memories[0]["status"] == "active"


def test_updates_changed_memory(tmp_path: Path) -> None:
    pipeline, store, extractor = make_pipeline(
        tmp_path,
        [make_fact("preferred_language", "Python")],
    )

    pipeline.ingest("user_1", "I prefer Python.")

    extractor.facts = [make_fact("preferred_language", "TypeScript")]
    memory_ids = pipeline.ingest("user_1", "I now prefer TypeScript.")

    memories = store.list_memories("user_1", include_inactive=True)
    active_memories = [
        memory for memory in memories if memory["status"] == "active"
    ]
    old_memories = [
        memory for memory in memories if memory["status"] == "superseded"
    ]

    assert len(memory_ids) == 1
    assert len(active_memories) == 1
    assert active_memories[0]["value"] == "TypeScript"
    assert len(old_memories) == 1
    assert old_memories[0]["value"] == "Python"
    assert active_memories[0]["supersedes_id"] == old_memories[0]["id"]


def test_skips_duplicate_memory(tmp_path: Path) -> None:
    pipeline, store, _ = make_pipeline(
        tmp_path,
        [make_fact("preferred_language", "Python")],
    )

    first_ids = pipeline.ingest("user_1", "I prefer Python.")
    second_ids = pipeline.ingest("user_1", "I prefer Python again.")

    memories = store.list_memories("user_1")

    assert len(first_ids) == 1
    assert second_ids == []
    assert len(memories) == 1
