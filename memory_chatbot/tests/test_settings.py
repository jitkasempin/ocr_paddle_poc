from pathlib import Path

import pytest

import memory_chatbot.settings as settings_module
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


def test_from_env_strips_explicit_openai_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEM0_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "  test-key  ")

    settings = ChatbotSettings.from_env()

    assert settings.openai_api_key == "test-key"


def test_from_env_treats_whitespace_only_openai_api_key_as_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MEM0_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "  \t  ")

    settings = ChatbotSettings.from_env()

    assert settings.openai_api_key is None


def test_from_env_rejects_repo_local_mem0_data_dir_outside_default_subtree(
    monkeypatch, tmp_path: Path
) -> None:
    fake_repo_root = tmp_path / "repo"
    fake_settings_file = fake_repo_root / "memory_chatbot" / "settings.py"
    unsafe_data_dir = fake_repo_root / "unsafe" / "data"

    monkeypatch.setattr(settings_module, "__file__", str(fake_settings_file))
    monkeypatch.setenv("MEM0_DATA_DIR", str(unsafe_data_dir))

    with pytest.raises(ValueError, match="MEM0_DATA_DIR must be outside the repository"):
        ChatbotSettings.from_env()

    assert not unsafe_data_dir.exists()


def test_from_env_allows_repo_local_mem0_data_dir_within_default_subtree(
    monkeypatch, tmp_path: Path
) -> None:
    fake_repo_root = tmp_path / "repo"
    fake_settings_file = fake_repo_root / "memory_chatbot" / "settings.py"
    allowed_data_dir = fake_repo_root / "memory_chatbot" / ".data" / "nested"

    monkeypatch.setattr(settings_module, "__file__", str(fake_settings_file))
    monkeypatch.setenv("MEM0_DATA_DIR", str(allowed_data_dir))

    settings = ChatbotSettings.from_env()

    assert settings.mem0_data_dir == allowed_data_dir.resolve()
    assert settings.mem0_data_dir.is_dir()
