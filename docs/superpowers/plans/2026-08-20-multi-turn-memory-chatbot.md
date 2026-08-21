# Multi-turn Memory Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable LangGraph chatbot that remembers durable preferences separately for each user through Mem0 OSS.

**Architecture:** A standalone Python package compiles a one-node LangGraph with thread-scoped checkpoints. Each model call recalls user-filtered facts, invokes an injected LangChain chat model, and writes the completed exchange back to the same user scope. Offline mock mode is the default; an optional live backend uses OpenAI and Mem0 OSS through the same interfaces.

**Tech Stack:** Python 3.12, LangChain Core, LangChain OpenAI, LangGraph, Mem0 OSS, Qdrant local storage, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-multi-turn-memory-chatbot-design.md`

## Global Constraints

- Use LangChain/LangGraph as the conversation framework.
- Use Mem0 OSS for durable, per-user long-term memory.
- Never commit API keys or generated memory data.
- Keep the implementation small, injectable, and testable without network calls.
- Preserve existing user changes and avoid modifying unrelated OCR code.
- Support Python 3.12.

---

### Task 1: Configuration and memory adapters

**Files:**
- Create: `memory_chatbot/__init__.py`
- Create: `memory_chatbot/settings.py`
- Create: `memory_chatbot/memory.py`
- Create: `memory_chatbot/tests/test_settings.py`
- Create: `memory_chatbot/tests/test_memory.py`

**Interfaces:**
- Consumes: environment variables from the design spec.
- Produces: `ChatbotSettings.from_env()`, `MemoryStore.search(query, user_id)`, `MemoryStore.add(messages, user_id)`, `JsonMemoryStore`, and `Mem0MemoryStore.from_settings(settings)`.

- [ ] **Step 1: Write failing settings and adapter tests**

Test that explicit environment values are loaded, the data directory is created, JSON memories survive a new store instance and remain isolated by user, Mem0 result dictionaries become `list[str]`, and the wrapped client receives exact `filters={"user_id": ...}` and `user_id=...` arguments.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest memory_chatbot/tests/test_settings.py memory_chatbot/tests/test_memory.py -q`

Expected: collection fails because package modules do not exist.

- [ ] **Step 3: Implement minimal settings and adapter**

Use `Path.resolve()`, create the configured directory, write JSON atomically through a temporary sibling file, and store only user-authored content under the exact user ID. Configure Qdrant with a fixed `collection_name`, explicit `path`, `on_disk=True`, and 1536 embedding dimensions, and configure a project-local SQLite history path. Normalize only non-empty string `memory` fields from `result["results"]`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest memory_chatbot/tests/test_settings.py memory_chatbot/tests/test_memory.py -q`

Expected: PASS.

### Task 2: LangGraph conversation workflow

**Files:**
- Create: `memory_chatbot/chatbot.py`
- Create: `memory_chatbot/tests/fakes.py`
- Create: `memory_chatbot/tests/test_chatbot.py`

**Interfaces:**
- Consumes: a LangChain `BaseChatModel` and the `MemoryStore` protocol from Task 1.
- Produces: `PreferenceChatbot(model, memory_store)` and `respond(*, user_id, thread_id, message) -> str`.

- [ ] **Step 1: Write failing workflow tests**

Cover multi-turn history in one thread, fresh history in another thread, exact user-specific recall, saved exchanges, identifier validation, and graceful memory failures. Use a recording fake model that returns deterministic `AIMessage` objects.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest memory_chatbot/tests/test_chatbot.py -q`

Expected: collection fails because `PreferenceChatbot` does not exist.

- [ ] **Step 3: Implement the graph**

Compile `StateGraph(MessagesState)` with `InMemorySaver`, add one `chat` node from `START` to `END`, pass `user_id` and `thread_id` through `configurable`, prepend a guarded system message containing normalized memories, call the model, then write the successful user/assistant pair to memory.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest memory_chatbot/tests/test_chatbot.py -q`

Expected: PASS.

### Task 3: Runnable CLI and documentation

**Files:**
- Create: `memory_chatbot/cli.py`
- Create: `memory_chatbot/models.py`
- Create: `memory_chatbot/__main__.py`
- Create: `memory_chatbot/.env.example`
- Create: `memory_chatbot/requirements.txt`
- Create: `memory_chatbot/README.md`
- Create: `memory_chatbot/tests/test_cli.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `ChatbotSettings`, `Mem0MemoryStore`, and `PreferenceChatbot`.
- Produces: `MockChatModel`, `build_chatbot(settings, backend)`, `main(argv=None) -> int`, and `python -m memory_chatbot --user-id <id>`.

- [ ] **Step 1: Write failing CLI tests**

Test `--help`, argument parsing, mock mode as the default, injected chatbot loop behavior, `exit` handling, deterministic preference recall, and a clear missing-key error for the live backend without exposing secrets.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest memory_chatbot/tests/test_cli.py -q`

Expected: collection fails because CLI modules do not exist.

- [ ] **Step 3: Implement CLI and operator docs**

Implement a deterministic LangChain-compatible mock model that acknowledges statements and summarizes the delimited memory block when asked about preferences. Default to the mock model and `JsonMemoryStore`; for `--backend openai`, initialize `ChatOpenAI` with `CHAT_MODEL` and construct the Mem0 adapter. Generate a UUID thread when none is supplied, loop on `input()`, and print concise recoverable errors. Document isolated installation, both backends, the distinction between `user_id` and `thread_id`, a two-user verification scenario, and local-demo limitations.

- [ ] **Step 4: Run CLI tests and help smoke check**

Run: `python -m pytest memory_chatbot/tests/test_cli.py -q`

Run: `python -m memory_chatbot --help`

Expected: tests pass and help exits zero.

### Task 4: Whole-feature verification

**Files:**
- Modify only files from Tasks 1-3 if verification exposes defects.

**Interfaces:**
- Consumes: the complete feature.
- Produces: evidence that every acceptance criterion not requiring a billable key is satisfied.

- [ ] **Step 1: Run all chatbot tests**

Run: `python -m pytest memory_chatbot/tests -q`

Expected: PASS.

- [ ] **Step 2: Run static checks**

Run: `python -m compileall -q memory_chatbot`

Run: `python -m ruff check memory_chatbot`

Expected: both exit zero.

- [ ] **Step 3: Inspect the final diff and requirement coverage**

Confirm no secret or generated database is tracked, every Mem0 read/write carries the same validated user ID, and the README contains a reproducible Alice/Bob memory-isolation walkthrough.

- [ ] **Step 4: Record live-test limitation honestly**

Do not run a networked smoke test. Verify mock mode end to end and report the optional OpenAI + Mem0 live path as not run by user choice. Do not print or inspect any plaintext key.
