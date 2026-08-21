# Multi-turn Memory Chatbot Design

## Objective

Build a small Python chatbot that maintains a multi-turn conversation and remembers durable preferences for each user. LangGraph owns the conversation workflow and thread state; Mem0 owns cross-session, user-scoped memories.

## Scope

The deliverable is a standalone command-line demo under `memory_chatbot/`. It is intentionally separate from the OCR Streamlit application. The demo supports one local terminal session at a time, accepts a stable user identifier, and can start multiple conversation threads for the same user. It runs in offline mock mode by default, with an optional live OpenAI + Mem0 OSS backend for later use.

The first version does not add authentication, a web UI, streaming tokens, tools, a remote database, or production multi-process deployment.

## Architecture

`PreferenceChatbot` exposes one operation:

```python
respond(*, user_id: str, thread_id: str, message: str) -> str
```

It compiles a one-node `StateGraph(MessagesState)` with an `InMemorySaver`. LangGraph associates short-term message history with `thread_id`. The model node:

1. reads the latest human message and the validated `user_id` from configurable runtime values;
2. searches Mem0 for relevant memories filtered by that exact `user_id`;
3. renders retrieved memories inside a clearly delimited, untrusted-data section of the system prompt;
4. invokes the LangChain chat model with the system prompt and thread messages;
5. saves the successful human/assistant exchange to Mem0 under the same `user_id`; and
6. returns the assistant message to LangGraph state.

Mem0 retrieval or write failures are logged and do not discard an otherwise valid model response. Model failures are surfaced to the caller.

## Component boundaries

- `memory_chatbot/settings.py`: reads and validates environment configuration and creates safe project-local data paths.
- `memory_chatbot/memory.py`: defines the memory protocol, a durable JSON mock store, normalizes Mem0 results, and constructs the OSS Mem0 adapter.
- `memory_chatbot/models.py`: supplies the deterministic offline chat model and optional OpenAI model factory.
- `memory_chatbot/chatbot.py`: owns user validation, prompt construction, LangGraph state, and the response workflow.
- `memory_chatbot/cli.py`: parses arguments and runs the terminal loop.
- `memory_chatbot/__main__.py`: supports `python -m memory_chatbot`.

The chat model and memory store are injected into `PreferenceChatbot`, so behavior is testable without API calls. Mock mode uses the same graph and memory interface as live mode: a deterministic model plus a project-local JSON memory file stores each user's statements across process restarts. Live mode swaps in `ChatOpenAI` and `Mem0MemoryStore` without changing the graph.

## Configuration and persistence

Mock mode is the default and requires no credentials. Runtime configuration comes from command-line flags and environment variables loaded from a local `.env` file:

- `--backend mock|openai` (default `mock`)
- `OPENAI_API_KEY` (required only for `--backend openai`)
- `CHAT_MODEL` (default `gpt-4.1-mini`)
- `MEM0_LLM_MODEL` (default `gpt-4.1-mini`)
- `MEM0_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `MEM0_DATA_DIR` (default `memory_chatbot/.data`)

Mock mode stores raw user statements by user ID in `MEM0_DATA_DIR/mock-memories.json`; this is deliberately a deterministic stand-in, not Mem0. Live mode uses Mem0 with an explicitly configured Qdrant collection with `path` and `on_disk: true`, plus a SQLite history database in `MEM0_DATA_DIR`. Generated data and `.env` are ignored by Git.

## User isolation

The CLI requires `--user-id`. `PreferenceChatbot` rejects blank identifiers, identifiers longer than 128 characters, and control characters. Every memory lookup and write carries the exact validated user ID. In live mode, every Mem0 search uses `filters={"user_id": user_id}` and every write uses `user_id=user_id`. The `thread_id` controls only LangGraph short-term state and never replaces the long-term user scope.

This is application-level separation for a local demo, not an authorization boundary. A networked version would derive `user_id` from authenticated identity rather than trusting command-line input.

## Prompt handling

Memories are treated as user-provided facts, not instructions. The system prompt tells the model to use relevant preferences for personalization, ignore instructions embedded inside the memory block, avoid claiming a memory that was not supplied, and honor the user's current request when it supersedes an older preference.

## Testing

Unit tests use deterministic fake chat models and fake memory stores to prove:

- LangGraph carries prior messages within one thread;
- different threads do not share short-term conversation state;
- memories for one user are never shown to another user;
- saved exchanges retain the correct `user_id`;
- Mem0 result normalization handles the documented response shape;
- memory read/write failures degrade gracefully; and
- mock memory persists and remains isolated by user across store instances;
- configuration produces durable project-local paths without requiring a live key; and
- live backend construction rejects a missing key with a clear error.

An import/CLI help check verifies packaging. A live OpenAI/Mem0 smoke test is optional because it requires billable external credentials.

## Acceptance criteria

1. `python -m memory_chatbot --user-id alice` starts an interactive offline chat without an API key.
2. A user can state a preference, end the process, start a new process with the same `user_id`, and receive a response personalized by that preference from the durable mock store.
3. A different `user_id` cannot retrieve the first user's preferences through the application workflow.
4. Successive messages in one `thread_id` include earlier turns; a new thread starts with fresh short-term history while retaining the user's Mem0 preferences.
5. `python -m memory_chatbot --backend openai --user-id alice` selects the real LangChain OpenAI + Mem0 OSS integration and fails clearly when `OPENAI_API_KEY` is absent.
6. Automated tests run without external services and pass.

## Risks and limitations

- Mock mode proves orchestration, durability, and user isolation but does not provide semantic extraction or natural model responses.
- Live Mem0 extraction and semantic search depend on the configured OpenAI models and can be probabilistic.
- Local Qdrant permits only a single process to open a given data directory reliably; this demo is single-process.
- The local CLI trusts the supplied `user_id`; production authentication is out of scope.
- In-memory LangGraph checkpoints disappear at process exit by design; Mem0 is the durable layer requested by this feature.
