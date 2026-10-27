
import pytest
from memora.memory_store import SQLiteMemoryStore

@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_memora.db"
    return SQLiteMemoryStore(db_path=str(db_path))


def test_add_and_retrieve_memory(store):
    memory_id = store.add_memory(
        user_id="user_001",
        slot="preferred_language",
        value="Python",
        source="user",
        confidence=0.95,
    )

    memories = store.get_active_by_slot(
        user_id="user_001",
        slot="preferred_language",
    )

    assert memory_id > 0
    assert len(memories) == 1
    assert memories[0]["value"] == "Python"
    assert memories[0]["status"] == "active"
    assert memories[0]["confidence"] == 0.95


def test_supersede_memory_preserves_history(store):
    old_id = store.add_memory(
        user_id="user_001",
        slot="preferred_language",
        value="Java",
    )

    new_id = store.supersede_memory(
        old_memory_id=old_id,
        new_value="Python",
    )

    active_memories = store.get_active_by_slot(
        user_id="user_001",
        slot="preferred_language",
    )

    all_memories = store.list_memories(
        user_id="user_001",
        include_inactive=True,
    )

    assert new_id != old_id
    assert len(active_memories) == 1
    assert active_memories[0]["value"] == "Python"
    assert len(all_memories) == 2

    old_memory = next(m for m in all_memories if m["id"] == old_id)

    assert old_memory["status"] == "superseded"
    assert active_memories[0]["supersedes_id"] == old_id


def test_invalidate_memory(store):
    memory_id = store.add_memory(
        user_id="user_001",
        slot="preferred_language",
        value="Java",
    )

    store.invalidate_memory(memory_id)

    active_memories = store.get_active_by_slot(
        user_id="user_001",
        slot="preferred_language",
    )

    all_memories = store.list_memories(
        user_id="user_001",
        include_inactive=True,
    )

    assert active_memories == []
    assert len(all_memories) == 1
    assert all_memories[0]["status"] == "invalidated"


def test_empty_memory_value_is_rejected(store):
    with pytest.raises(ValueError, match="value cannot be empty"):
        store.add_memory(
            user_id="user_001",
            slot="preferred_language",
            value="   ",
        )
