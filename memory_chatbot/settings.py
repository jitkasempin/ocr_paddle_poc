from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChatbotSettings:
    openai_api_key: str | None
    chat_model: str
    mem0_llm_model: str
    mem0_embedding_model: str
    mem0_data_dir: Path
    mock_memories_path: Path
    mem0_qdrant_path: Path
    mem0_history_db_path: Path
    mem0_collection_name: str

    @classmethod
    def from_env(cls) -> ChatbotSettings:
        package_dir = Path(__file__).resolve().parent
        raw_data_dir = os.getenv("MEM0_DATA_DIR")
        data_dir = Path(raw_data_dir) if raw_data_dir else package_dir / ".data"
        resolved_data_dir = data_dir.resolve()
        resolved_data_dir.mkdir(parents=True, exist_ok=True)

        raw_openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_key = raw_openai_api_key if raw_openai_api_key else None

        return cls(
            openai_api_key=openai_api_key,
            chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
            mem0_llm_model=os.getenv("MEM0_LLM_MODEL", "gpt-4.1-mini"),
            mem0_embedding_model=os.getenv(
                "MEM0_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            mem0_data_dir=resolved_data_dir,
            mock_memories_path=(resolved_data_dir / "mock-memories.json").resolve(),
            mem0_qdrant_path=(resolved_data_dir / "qdrant").resolve(),
            mem0_history_db_path=(resolved_data_dir / "history.db").resolve(),
            mem0_collection_name="memory_chatbot",
        )
