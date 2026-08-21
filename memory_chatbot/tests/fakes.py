from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import Field


class RecordingChatModel(SimpleChatModel):
    reply_prefix: str = "assistant:"
    calls: list[list[dict[str, str]]] = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "recording-chat-model"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        self.calls.append(
            [
                {
                    "role": message.type,
                    "content": str(message.text),
                }
                for message in messages
            ]
        )

        latest_human = next(
            str(message.text)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        return f"{self.reply_prefix} {latest_human}"


class RecordingMemoryStore:
    def __init__(
        self,
        *,
        memories_by_user: Mapping[str, Sequence[str]] | None = None,
        search_error: Exception | None = None,
        add_error: Exception | None = None,
    ) -> None:
        self.memories_by_user = {
            user_id: list(memories)
            for user_id, memories in (memories_by_user or {}).items()
        }
        self.search_error = search_error
        self.add_error = add_error
        self.search_calls: list[dict[str, str]] = []
        self.add_calls: list[dict[str, object]] = []

    def search(self, query: str, user_id: str) -> list[str]:
        self.search_calls.append({"query": query, "user_id": user_id})
        if self.search_error is not None:
            raise self.search_error
        return list(self.memories_by_user.get(user_id, []))

    def add(self, messages: Sequence[Mapping[str, str]], user_id: str) -> None:
        normalized_messages = [dict(message) for message in messages]
        self.add_calls.append(
            {"messages": deepcopy(normalized_messages), "user_id": user_id}
        )
        if self.add_error is not None:
            raise self.add_error

        remembered_user_messages = [
            message["content"]
            for message in normalized_messages
            if message.get("role") == "user" and message.get("content")
        ]
        if not remembered_user_messages:
            return

        self.memories_by_user.setdefault(user_id, []).extend(remembered_user_messages)
