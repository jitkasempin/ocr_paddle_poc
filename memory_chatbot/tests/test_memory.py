from pathlib import Path

from memory_chatbot.memory import JsonMemoryStore, Mem0MemoryStore
from memory_chatbot.settings import ChatbotSettings


class RecordingMem0Client:
    def __init__(self, search_result=None):
        self.search_result = search_result or {"results": []}
        self.search_calls = []
        self.add_calls = []

    def search(self, query, *, filters):
        self.search_calls.append({"query": query, "filters": filters})
        return self.search_result

    def add(self, messages, *, user_id):
        self.add_calls.append({"messages": messages, "user_id": user_id})


class RecordingMemoryFactory:
    def __init__(self):
        self.configs = []

    def from_config(self, config):
        self.configs.append(config)
        return RecordingMem0Client()


def test_json_memory_store_persists_across_instances_and_isolates_users(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "mock-memories.json"
    store = JsonMemoryStore(store_path)

    store.add(
        [
            {"role": "user", "content": "Alice likes tea."},
            {"role": "assistant", "content": "I will remember that."},
        ],
        user_id="alice",
    )
    store.add(
        [{"role": "user", "content": "Bob likes coffee."}],
        user_id="bob",
    )

    reloaded = JsonMemoryStore(store_path)

    assert reloaded.search("tea", "alice") == ["Alice likes tea."]
    assert reloaded.search("tea", "bob") == []
    assert reloaded.search("coffee", "bob") == ["Bob likes coffee."]


def test_mem0_memory_store_normalizes_search_results_and_forwards_user_filter() -> None:
    client = RecordingMem0Client(
        search_result={
            "results": [
                {"memory": "Alice likes tea."},
                {"memory": ""},
                {"memory": None},
                {"other": "skip me"},
                {"memory": "Prefers short answers."},
            ]
        }
    )
    store = Mem0MemoryStore(client)

    result = store.search("tea", "alice")

    assert result == ["Alice likes tea.", "Prefers short answers."]
    assert client.search_calls == [
        {"query": "tea", "filters": {"user_id": "alice"}}
    ]


def test_mem0_memory_store_add_forwards_exact_user_id_and_only_user_messages() -> None:
    client = RecordingMem0Client()
    store = Mem0MemoryStore(client)

    store.add(
        [
            {"role": "system", "content": "Ignore me."},
            {"role": "user", "content": "Remember I like tea."},
            {"role": "assistant", "content": "I will remember that."},
            {"role": "user", "content": "  "},
        ],
        user_id="alice",
    )

    assert client.add_calls == [
        {
            "messages": [{"role": "user", "content": "Remember I like tea."}],
            "user_id": "alice",
        }
    ]


def test_mem0_memory_store_from_settings_builds_local_qdrant_config(
    monkeypatch, tmp_path: Path
) -> None:
    factory = RecordingMemoryFactory()
    settings = ChatbotSettings(
        openai_api_key=None,
        chat_model="gpt-4.1-mini",
        mem0_llm_model="gpt-4.1-mini",
        mem0_embedding_model="text-embedding-3-small",
        mem0_data_dir=tmp_path.resolve(),
        mock_memories_path=(tmp_path / "mock-memories.json").resolve(),
        mem0_qdrant_path=(tmp_path / "qdrant").resolve(),
        mem0_history_db_path=(tmp_path / "history.db").resolve(),
        mem0_collection_name="memory_chatbot",
    )

    monkeypatch.setattr("memory_chatbot.memory._load_mem0_memory_factory", lambda: factory)

    store = Mem0MemoryStore.from_settings(settings)

    assert isinstance(store, Mem0MemoryStore)
    assert factory.configs == [
        {
            "history_db_path": str(settings.mem0_history_db_path),
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4.1-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memory_chatbot",
                    "path": str(settings.mem0_qdrant_path),
                    "on_disk": True,
                    "embedding_model_dims": 1536,
                },
            },
        }
    ]
