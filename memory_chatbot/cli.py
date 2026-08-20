from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv

from memory_chatbot.chatbot import PreferenceChatbot
from memory_chatbot.memory import JsonMemoryStore, Mem0MemoryStore
from memory_chatbot.models import MockChatModel, create_openai_chat_model, looks_like_preference_question
from memory_chatbot.settings import ChatbotSettings

Backend = Literal["mock", "openai"]


class DurableMockMemoryStore:
    def __init__(self, store: JsonMemoryStore) -> None:
        self._store = store

    def search(self, query: str, user_id: str) -> list[str]:
        memories = self._store.search(query, user_id)
        if memories or not looks_like_preference_question(query):
            return memories
        return self._store.search("", user_id)

    def add(self, messages, user_id: str) -> None:
        self._store.add(messages, user_id)


def build_chatbot(settings: ChatbotSettings, backend: Backend) -> PreferenceChatbot:
    if backend == "mock":
        memory_store = DurableMockMemoryStore(JsonMemoryStore(settings.mock_memories_path))
        return PreferenceChatbot(model=MockChatModel(), memory_store=memory_store)

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai.")
    return PreferenceChatbot(
        model=create_openai_chat_model(settings),
        memory_store=Mem0MemoryStore.from_settings(settings),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    settings = ChatbotSettings.from_env()
    thread_id = args.thread_id or str(uuid4())

    try:
        chatbot = build_chatbot(settings, backend=args.backend)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    while True:
        try:
            user_message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye.")
            return 0

        normalized_message = user_message.strip()
        if not normalized_message:
            continue
        if normalized_message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0

        try:
            reply = chatbot.respond(
                user_id=args.user_id,
                thread_id=thread_id,
                message=normalized_message,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(reply)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory_chatbot",
        description="Standalone LangGraph + Mem0 memory chatbot demo.",
    )
    parser.add_argument("--user-id", required=True, help="Stable user identifier for durable memories.")
    parser.add_argument(
        "--thread-id",
        help="Optional LangGraph thread identifier. Defaults to a generated UUID per session.",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "openai"),
        default="mock",
        help="Chat backend to use. Default: mock.",
    )
    return parser
