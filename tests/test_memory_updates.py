
import pytest

from memora.memory_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_memora.db"
    return SQLiteMemoryStore(db_path=str(db_path))


def test_new_memory_is_saved(store):
    memory_id = store.add_memory(
        user_id="user_1",
        slot="currently_learning",
        value="LangChain",
        category="learning",
    )

    memories = store.get_active_by_slot(
        user_id="user_1",
        slot="currently_learning",
        scope="general",
    )

    assert memory_id > 0
    assert any(memory["value"] == "LangChain" for memory in memories)


def test_updated_memory_supersedes_old_value(store):
    old_id = store.add_memory(
        user_id="user_1",
        slot="currently_learning",
        value="Python basics",
        category="learning",
    )

    new_id = store.supersede_memory(
        old_memory_id=old_id,
        new_value="LangChain",
        source="conversation",
        confidence=1.0,
    )

    active_memories = store.get_active_by_slot(
        user_id="user_1",
        slot="currently_learning",
        scope="general",
    )

    assert new_id > 0
    assert any(memory["value"] == "LangChain" for memory in active_memories)
    assert not any(
        memory["value"] == "Python basics" for memory in active_memories
    )


def test_unrelated_memory_is_preserved(store):
    store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
        category="preference",
    )

    old_id = store.add_memory(
        user_id="user_1",
        slot="currently_learning",
        value="Python basics",
        category="learning",
    )

    store.supersede_memory(
        old_memory_id=old_id,
        new_value="LangChain",
        source="conversation",
        confidence=1.0,
    )

    preferences = store.get_active_by_slot(
        user_id="user_1",
        slot="preferred_language",
        scope="general",
    )

    learning = store.get_active_by_slot(
        user_id="user_1",
        slot="currently_learning",
        scope="general",
    )

    assert any(memory["value"] == "Python" for memory in preferences)
    assert any(memory["value"] == "LangChain" for memory in learning)


def test_users_memories_are_isolated(store):
    store.add_memory(
        user_id="user_1",
        slot="preferred_language",
        value="Python",
        category="preference",
    )

    store.add_memory(
        user_id="user_2",
        slot="preferred_language",
        value="Java",
        category="preference",
    )

    user_1_memories = store.list_memories(
        user_id="user_1",
        include_inactive=False,
    )

    user_2_memories = store.list_memories(
        user_id="user_2",
        include_inactive=False,
    )

    assert any(memory["value"] == "Python" for memory in user_1_memories)
    assert not any(memory["value"] == "Java" for memory in user_1_memories)

    assert any(memory["value"] == "Java" for memory in user_2_memories)
    assert not any(memory["value"] == "Python" for memory in user_2_memories)
