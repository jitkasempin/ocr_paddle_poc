# Memory Chatbot

Standalone LangGraph + Mem0 memory chatbot demo under `memory_chatbot/`.

## What it does

- Default `mock` backend runs fully offline with a deterministic chat model.
- Durable user memories are stored in `MEM0_DATA_DIR/mock-memories.json`.
- `user_id` scopes long-term memories across sessions.
- `thread_id` scopes only short-term LangGraph conversation history inside one process.

## Setup

Create an isolated environment and install the standalone demo dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r memory_chatbot/requirements.txt
```

Optional local configuration:

```bash
copy memory_chatbot/.env.example memory_chatbot/.env
```

## Run

Offline mock mode requires no credentials:

```bash
python -m memory_chatbot --user-id alice
```

Optional live backend:

```bash
python -m memory_chatbot --backend openai --user-id alice
```

`OPENAI_API_KEY` is required only for `--backend openai`.

## Demo flow

1. Start `python -m memory_chatbot --user-id alice`
2. Say `Remember that I like jasmine tea.`
3. Type `exit`
4. Start `python -m memory_chatbot --user-id alice` again
5. Ask `What are my preferences?`

Expected mock reply: `You told me these preferences: Remember that I like jasmine tea.`

To verify isolation, repeat the same question with `--user-id bob`; Bob should not receive Alice's memory.

## Notes

- Mock mode is deterministic and local. It is intended to prove orchestration, durability, and per-user isolation, not natural language quality.
- Live mode constructs `ChatOpenAI` and Mem0 lazily, so mock mode stays credential-free.
- The local demo trusts the supplied `user_id`; this is not an auth boundary.
