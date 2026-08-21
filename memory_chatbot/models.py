from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from memory_chatbot.settings import ChatbotSettings


class MockChatModel(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        latest_human_message = next(
            str(message.text)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        memories = _extract_memories(messages)

        if _looks_like_preference_question(latest_human_message):
            if not memories:
                return "You haven't shared any preferences yet."
            return f"You told me these preferences: {'; '.join(memories)}"

        return f"Noted. I'll remember: {latest_human_message}"


def create_openai_chat_model(settings: ChatbotSettings):
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai.")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "langchain-openai is not installed. Install the standalone requirements first."
        ) from exc

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.chat_model,
        temperature=0,
    )


def looks_like_preference_question(message: str) -> bool:
    return _looks_like_preference_question(message)


def _extract_memories(messages: Sequence[BaseMessage]) -> list[str]:
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        text = str(message.text)
        start_token = "<memories-json>"
        end_token = "</memories-json>"
        start = text.find(start_token)
        end = text.find(end_token)
        if start == -1 or end == -1 or end <= start:
            continue
        memory_block = text[start + len(start_token) : end].strip()
        try:
            decoded = json.loads(memory_block)
        except json.JSONDecodeError:
            return []
        if not isinstance(decoded, list):
            return []
        return [
            memory.strip()
            for memory in decoded
            if isinstance(memory, str) and memory.strip()
        ]
    return []


def _looks_like_preference_question(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False

    question_starts = (
        "what",
        "which",
        "do i",
        "did i",
        "could you",
        "can you",
        "tell me",
    )
    memory_keywords = (
        "preference",
        "preferences",
        "prefer",
        "like",
        "likes",
        "remember",
        "know about me",
    )
    return normalized.startswith(question_starts) and any(
        keyword in normalized for keyword in memory_keywords
    )
