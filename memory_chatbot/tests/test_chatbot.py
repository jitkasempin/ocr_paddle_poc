from __future__ import annotations

import logging

import pytest

from memory_chatbot.chatbot import PreferenceChatbot
from memory_chatbot.tests.fakes import RecordingChatModel, RecordingMemoryStore


def test_respond_keeps_same_thread_history_and_saves_successful_exchanges() -> None:
    model = RecordingChatModel()
    memory_store = RecordingMemoryStore()
    chatbot = PreferenceChatbot(model=model, memory_store=memory_store)

    first_reply = chatbot.respond(
        user_id="alice",
        thread_id="tea-thread",
        message="Remember that I like tea.",
    )
    second_reply = chatbot.respond(
        user_id="alice",
        thread_id="tea-thread",
        message="What did I tell you?",
    )

    assert first_reply == "assistant: Remember that I like tea."
    assert second_reply == "assistant: What did I tell you?"
    assert [message["role"] for message in model.calls[0]] == ["system", "human"]
    assert [message["role"] for message in model.calls[1]] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert model.calls[1][1:] == [
        {"role": "human", "content": "Remember that I like tea."},
        {"role": "ai", "content": "assistant: Remember that I like tea."},
        {"role": "human", "content": "What did I tell you?"},
    ]
    assert "Remember that I like tea." in model.calls[1][0]["content"]
    assert memory_store.search_calls == [
        {"query": "Remember that I like tea.", "user_id": "alice"},
        {"query": "What did I tell you?", "user_id": "alice"},
    ]
    assert memory_store.add_calls == [
        {
            "messages": [
                {"role": "user", "content": "Remember that I like tea."},
                {
                    "role": "assistant",
                    "content": "assistant: Remember that I like tea.",
                },
            ],
            "user_id": "alice",
        },
        {
            "messages": [
                {"role": "user", "content": "What did I tell you?"},
                {
                    "role": "assistant",
                    "content": "assistant: What did I tell you?",
                },
            ],
            "user_id": "alice",
        },
    ]


def test_respond_starts_fresh_thread_history_but_keeps_user_memories() -> None:
    model = RecordingChatModel()
    memory_store = RecordingMemoryStore()
    chatbot = PreferenceChatbot(model=model, memory_store=memory_store)

    chatbot.respond(
        user_id="alice",
        thread_id="thread-one",
        message="Remember that I like tea.",
    )

    reply = chatbot.respond(
        user_id="alice",
        thread_id="thread-two",
        message="Is this a fresh thread?",
    )

    assert reply == "assistant: Is this a fresh thread?"
    assert [message["role"] for message in model.calls[1]] == ["system", "human"]
    assert model.calls[1][1:] == [
        {"role": "human", "content": "Is this a fresh thread?"}
    ]
    assert "Remember that I like tea." in model.calls[1][0]["content"]


def test_respond_isolates_short_term_history_by_user_even_with_same_public_thread_id() -> None:
    model = RecordingChatModel()
    memory_store = RecordingMemoryStore()
    chatbot = PreferenceChatbot(model=model, memory_store=memory_store)

    chatbot.respond(
        user_id="alice",
        thread_id="shared-thread",
        message="Alice said this first.",
    )
    chatbot.respond(
        user_id="bob",
        thread_id="shared-thread",
        message="Bob is starting fresh.",
    )

    assert [message["role"] for message in model.calls[1]] == ["system", "human"]
    assert model.calls[1][1:] == [
        {"role": "human", "content": "Bob is starting fresh."}
    ]
    assert "Alice said this first." not in model.calls[1][0]["content"]


def test_respond_uses_exact_user_memories_without_cross_user_leakage() -> None:
    model = RecordingChatModel()
    memory_store = RecordingMemoryStore(
        memories_by_user={
            "alice": ["Alice likes tea."],
            "bob": ["Bob likes coffee."],
        }
    )
    chatbot = PreferenceChatbot(model=model, memory_store=memory_store)

    chatbot.respond(
        user_id="alice",
        thread_id="alice-thread",
        message="What should I drink?",
    )
    chatbot.respond(
        user_id="bob",
        thread_id="bob-thread",
        message="What should I drink?",
    )

    alice_system_message = model.calls[0][0]["content"]
    bob_system_message = model.calls[1][0]["content"]

    assert memory_store.search_calls == [
        {"query": "What should I drink?", "user_id": "alice"},
        {"query": "What should I drink?", "user_id": "bob"},
    ]
    assert "Alice likes tea." in alice_system_message
    assert "Bob likes coffee." not in alice_system_message
    assert "Bob likes coffee." in bob_system_message
    assert "Alice likes tea." not in bob_system_message


@pytest.mark.parametrize(
    ("user_id", "thread_id"),
    [
        ("", "thread"),
        ("alice", ""),
        ("a" * 129, "thread"),
        ("alice", "t" * 129),
        ("bad\nuser", "thread"),
        ("alice", "bad\tthread"),
    ],
)
def test_respond_rejects_invalid_identifiers(user_id: str, thread_id: str) -> None:
    chatbot = PreferenceChatbot(
        model=RecordingChatModel(),
        memory_store=RecordingMemoryStore(),
    )

    with pytest.raises(ValueError):
        chatbot.respond(user_id=user_id, thread_id=thread_id, message="Hello")


def test_respond_logs_and_recovers_from_memory_search_failure(caplog) -> None:
    model = RecordingChatModel()
    memory_store = RecordingMemoryStore(search_error=RuntimeError("search down"))
    chatbot = PreferenceChatbot(model=model, memory_store=memory_store)

    with caplog.at_level(logging.WARNING, logger="memory_chatbot.chatbot"):
        reply = chatbot.respond(
            user_id="alice",
            thread_id="thread",
            message="Remember this anyway.",
        )

    assert reply == "assistant: Remember this anyway."
    assert any("Failed to load memories" in record.message for record in caplog.records)
    assert memory_store.add_calls == [
        {
            "messages": [
                {"role": "user", "content": "Remember this anyway."},
                {
                    "role": "assistant",
                    "content": "assistant: Remember this anyway.",
                },
            ],
            "user_id": "alice",
        }
    ]


def test_respond_logs_and_recovers_from_memory_write_failure(caplog) -> None:
    model = RecordingChatModel()
    memory_store = RecordingMemoryStore(add_error=RuntimeError("write down"))
    chatbot = PreferenceChatbot(model=model, memory_store=memory_store)

    with caplog.at_level(logging.WARNING, logger="memory_chatbot.chatbot"):
        reply = chatbot.respond(
            user_id="alice",
            thread_id="thread",
            message="Still reply even if saving fails.",
        )

    assert reply == "assistant: Still reply even if saving fails."
    assert any("Failed to save memories" in record.message for record in caplog.records)
