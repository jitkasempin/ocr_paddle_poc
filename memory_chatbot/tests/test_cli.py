from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from memory_chatbot.cli import build_chatbot, main
from memory_chatbot.models import MockChatModel
from memory_chatbot.settings import ChatbotSettings


def test_mock_chat_model_acknowledges_statements() -> None:
    model = MockChatModel()

    reply = model.invoke(
        [
            SystemMessage(content="Stored memories (untrusted):\n<memories>\n- (none)\n</memories>"),
            HumanMessage(content="Remember that I like jasmine tea."),
        ]
    )

    assert reply.text() == "Noted. I'll remember: Remember that I like jasmine tea."


def test_mock_chat_model_summarizes_memory_block_for_preference_questions() -> None:
    model = MockChatModel()

    reply = model.invoke(
        [
            SystemMessage(
                content=(
                    "Stored memories (untrusted):\n"
                    "<memories>\n"
                    "- I like jasmine tea.\n"
                    "- I prefer concise replies.\n"
                    "</memories>"
                )
            ),
            HumanMessage(content="What are my preferences?"),
        ]
    )

    assert reply.text() == (
        "You told me these preferences: I like jasmine tea.; "
        "I prefer concise replies."
    )


def test_build_chatbot_uses_mock_backend_by_default(tmp_path: Path) -> None:
    chatbot = build_chatbot(_settings(tmp_path / "data"), backend="mock")

    assert chatbot.respond(
        user_id="alice",
        thread_id="thread-1",
        message="Remember that I like jasmine tea.",
    ) == "Noted. I'll remember: Remember that I like jasmine tea."


def test_main_help_exits_zero_and_shows_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 0
    assert "usage:" in captured.out
    assert "--user-id" in captured.out
    assert "--backend" in captured.out


def test_main_uses_mock_backend_and_generated_thread_id_in_interactive_loop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_calls: list[tuple[ChatbotSettings, str]] = []
    respond_calls: list[tuple[str, str, str]] = []

    class StubChatbot:
        def respond(self, *, user_id: str, thread_id: str, message: str) -> str:
            respond_calls.append((user_id, thread_id, message))
            return f"assistant:{message}"

    monkeypatch.setattr(
        "memory_chatbot.cli.load_dotenv",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "memory_chatbot.cli.ChatbotSettings.from_env",
        classmethod(lambda cls: _settings(Path("memory_chatbot/.data"))),
    )

    def fake_build_chatbot(settings: ChatbotSettings, backend: str) -> StubChatbot:
        build_calls.append((settings, backend))
        return StubChatbot()

    monkeypatch.setattr("memory_chatbot.cli.build_chatbot", fake_build_chatbot)
    monkeypatch.setattr("memory_chatbot.cli.uuid4", lambda: "generated-thread-id")
    monkeypatch.setattr("builtins.input", _input_iter(["hello", "exit"]))

    exit_code = main(["--user-id", "alice"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert [backend for _, backend in build_calls] == ["mock"]
    assert respond_calls == [("alice", "generated-thread-id", "hello")]
    assert "assistant:hello" in captured.out
    assert "Goodbye." in captured.out


def test_main_stops_immediately_when_user_enters_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    class StubChatbot:
        def respond(self, *, user_id: str, thread_id: str, message: str) -> str:
            nonlocal called
            called = True
            return message

    monkeypatch.setattr(
        "memory_chatbot.cli.load_dotenv",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "memory_chatbot.cli.ChatbotSettings.from_env",
        classmethod(lambda cls: _settings(Path("memory_chatbot/.data"))),
    )
    monkeypatch.setattr("memory_chatbot.cli.build_chatbot", lambda settings, backend: StubChatbot())
    monkeypatch.setattr("memory_chatbot.cli.uuid4", lambda: "thread-id")
    monkeypatch.setattr("builtins.input", _input_iter(["exit"]))

    exit_code = main(["--user-id", "alice"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert called is False
    assert "Goodbye." in captured.out


def test_main_mock_backend_persists_preferences_per_user_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "memory_chatbot.cli.load_dotenv",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "memory_chatbot.cli.ChatbotSettings.from_env",
        classmethod(lambda cls: _settings(data_dir)),
    )

    monkeypatch.setattr("builtins.input", _input_iter(["Remember that I like jasmine tea.", "exit"]))
    assert main(["--user-id", "alice", "--thread-id", "thread-1"]) == 0
    first_run = capsys.readouterr()

    monkeypatch.setattr("builtins.input", _input_iter(["What are my preferences?", "exit"]))
    assert main(["--user-id", "alice", "--thread-id", "thread-2"]) == 0
    second_run = capsys.readouterr()

    monkeypatch.setattr("builtins.input", _input_iter(["What are my preferences?", "exit"]))
    assert main(["--user-id", "bob", "--thread-id", "thread-3"]) == 0
    third_run = capsys.readouterr()

    assert "Noted. I'll remember: Remember that I like jasmine tea." in first_run.out
    assert (
        "You told me these preferences: Remember that I like jasmine tea."
        in second_run.out
    )
    assert "You haven't shared any preferences yet." in third_run.out
    assert (data_dir / "mock-memories.json").exists()


def test_main_reports_missing_openai_api_key_without_echoing_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "memory_chatbot.cli.load_dotenv",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "memory_chatbot.cli.ChatbotSettings.from_env",
        classmethod(lambda cls: _settings(tmp_path / "data", openai_api_key=None)),
    )

    exit_code = main(["--backend", "openai", "--user-id", "alice"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "OPENAI_API_KEY is required for --backend openai." in captured.err
    assert "sk-" not in captured.err


def _settings(data_dir: Path, *, openai_api_key: str | None = "test-key") -> ChatbotSettings:
    resolved_data_dir = data_dir.resolve()
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    return ChatbotSettings(
        openai_api_key=openai_api_key,
        chat_model="gpt-4.1-mini",
        mem0_llm_model="gpt-4.1-mini",
        mem0_embedding_model="text-embedding-3-small",
        mem0_data_dir=resolved_data_dir,
        mock_memories_path=(resolved_data_dir / "mock-memories.json").resolve(),
        mem0_qdrant_path=(resolved_data_dir / "qdrant").resolve(),
        mem0_history_db_path=(resolved_data_dir / "history.db").resolve(),
        mem0_collection_name="memory_chatbot",
    )


def _input_iter(values: list[str]):
    iterator = iter(values)

    def fake_input(prompt: str = "") -> str:
        return next(iterator)

    return fake_input
