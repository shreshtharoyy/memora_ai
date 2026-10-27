
from typing import Any
from memora.fact_extractor import FactExtractor
from memora.memory_store import SQLiteMemoryStore

class MemoryIngestionPipeline:

    def __init__(
        self,
        extractor: FactExtractor | None = None,
        store: SQLiteMemoryStore | None = None,
    ) -> None:
        self.extractor = extractor if extractor is not None else FactExtractor()
        self.store = store if store is not None else SQLiteMemoryStore()

    def ingest(self, user_id: str, conversation: str) -> list[int]:

        if not user_id.strip():
            raise ValueError("user_id cannot be empty.")

        if not conversation.strip():
            raise ValueError("Conversation cannot be empty.")

        facts = self.extractor.extract(conversation)
        memory_ids: list[int] = []

        for fact in facts:
            existing = self.store.get_active_by_slot(
                user_id=user_id,
                slot=fact.slot,
                scope=fact.scope,
            )

            if any(memory["value"] == fact.value for memory in existing):
                continue

            if existing:
                old_memory = existing[0]
                memory_id = self.store.supersede_memory(
                    old_memory_id=old_memory["id"],
                    new_value=fact.value,
                    source=fact.source,
                    confidence=fact.confidence,
                )
            else:
                memory_id = self.store.add_memory(
                    user_id=user_id,
                    slot=fact.slot,
                    value=fact.value,
                    category=fact.category,
                    scope=fact.scope,
                    source=fact.source,
                    confidence=fact.confidence,
                )

            memory_ids.append(memory_id)

        return memory_ids
