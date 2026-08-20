from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from memory_chatbot.settings import ChatbotSettings


class MemoryStore(Protocol):
    def search(self, query: str, user_id: str) -> list[str]:
        ...

    def add(self, messages: Sequence[Mapping[str, str]], user_id: str) -> None:
        ...


def _load_mem0_memory_factory() -> Any:
    try:
        from mem0 import Memory
    except ImportError as exc:  # pragma: no cover - exercised only with live dependency
        raise RuntimeError(
            "Mem0 OSS is not installed. Install mem0ai before using the live backend."
        ) from exc

    return Memory


class JsonMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, user_id: str) -> list[str]:
        memories = self._read_data().get(user_id, [])
        normalized_query = query.strip().lower()
        if not normalized_query:
            return list(memories)
        return [
            memory for memory in memories if normalized_query in memory.lower()
        ]

    def add(self, messages: Sequence[Mapping[str, str]], user_id: str) -> None:
        user_memories = [
            content
            for message in messages
            if message.get("role") == "user"
            for content in [message.get("content", "").strip()]
            if content
        ]
        if not user_memories:
            return

        data = self._read_data()
        data.setdefault(user_id, []).extend(user_memories)
        self._write_data(data)

    def _read_data(self) -> dict[str, list[str]]:
        if not self.path.exists():
            return {}

        raw_data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            return {}

        normalized: dict[str, list[str]] = {}
        for user_id, memories in raw_data.items():
            if not isinstance(user_id, str) or not isinstance(memories, list):
                continue
            normalized[user_id] = [
                memory for memory in memories if isinstance(memory, str) and memory
            ]
        return normalized

    def _write_data(self, data: dict[str, list[str]]) -> None:
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


class Mem0MemoryStore:
    def __init__(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: ChatbotSettings) -> Mem0MemoryStore:
        memory_factory = _load_mem0_memory_factory()
        client = memory_factory.from_config(
            {
                "history_db_path": str(settings.mem0_history_db_path),
                "llm": {
                    "provider": "openai",
                    "config": {"model": settings.mem0_llm_model},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": settings.mem0_embedding_model},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": settings.mem0_collection_name,
                        "path": str(settings.mem0_qdrant_path),
                        "on_disk": True,
                        "embedding_model_dims": 1536,
                    },
                },
            }
        )
        return cls(client)

    def search(self, query: str, user_id: str) -> list[str]:
        result = self.client.search(query, filters={"user_id": user_id})
        if not isinstance(result, dict):
            return []

        memories: list[str] = []
        for item in result.get("results", []):
            if not isinstance(item, Mapping):
                continue
            memory = item.get("memory")
            if isinstance(memory, str) and memory:
                memories.append(memory)
        return memories

    def add(self, messages: Sequence[Mapping[str, str]], user_id: str) -> None:
        remembered_messages = [
            {"role": role, "content": content}
            for message in messages
            for role in [message.get("role")]
            if role in {"user", "assistant"}
            for content in [message.get("content", "").strip()]
            if content
        ]
        if not remembered_messages:
            return
        self.client.add(remembered_messages, user_id=user_id)
