from pathlib import Path

from memory_chatbot.settings import ChatbotSettings


def test_from_env_loads_explicit_values_and_creates_data_dir(
    monkeypatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "custom-data" / "memories"

    monkeypatch.setenv("CHAT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("MEM0_LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEM0_DATA_DIR", str(data_dir))

    settings = ChatbotSettings.from_env()

    assert settings.chat_model == "gpt-4.1-mini"
    assert settings.mem0_llm_model == "gpt-4.1-mini"
    assert settings.mem0_embedding_model == "text-embedding-3-small"
    assert settings.mem0_data_dir == data_dir.resolve()
    assert settings.mem0_data_dir.is_dir()
    assert settings.mock_memories_path == data_dir.resolve() / "mock-memories.json"
    assert settings.mem0_qdrant_path == data_dir.resolve() / "qdrant"
    assert settings.mem0_history_db_path == data_dir.resolve() / "history.db"
    assert settings.mem0_collection_name == "memory_chatbot"
