# LightRAG Legal Chat POC Design

## Objective

Add a self-contained proof of concept that runs LightRAG locally through Docker, ingests legal source documents, and answers questions with file-level citations and supporting excerpts. The POC must be quick to start, must not alter the existing OCR application, and must make unsupported answers visible instead of presenting them as legal fact.

## Scope

The deliverable lives under `legal_lightrag_poc/` and contains:

- a Docker Compose deployment pinned to LightRAG `v1.5.6`;
- an environment template for an OpenAI-compatible LLM and embedding model;
- a reusable Python REST client;
- a command-line ingestion script for files and directories;
- a command-line query/retrieval script that displays citations and source excerpts;
- focused unit tests and a quick-start guide.

This is a local POC, not a production legal-advice service. It does not include a web chat UI, user accounts, multi-tenancy, OCR, document-level access control, evaluation datasets, or a production database.

## Architecture

The official LightRAG API server runs in one Docker container. Its default JSON, NetworkX, and NanoVectorDB stores persist in bind-mounted local directories. The Python scripts communicate with the server over its documented REST endpoints instead of embedding the LightRAG SDK in the existing application.

The server binds to `127.0.0.1` on the host and requires `X-API-Key` for document and query endpoints. The container may contact an OpenAI-compatible provider for entity extraction, query generation, and embeddings. The image is pinned rather than using `latest` so API behavior remains reproducible.

## Components

### Docker deployment

`legal_lightrag_poc/docker-compose.yml` starts `ghcr.io/hkuds/lightrag:v1.5.6`, maps host port `9621`, and persists `/app/data/rag_storage` and `/app/data/inputs`. The `.env.example` uses the lightweight POC storage backends and contains explicit placeholders for provider and LightRAG API keys.

### REST client

`legal_lightrag_poc/legal_rag_client.py` owns authentication, timeouts, error messages, upload requests, track-status polling, and query requests. It calls:

- `GET /health` to verify the server;
- `POST /documents/upload` to enqueue a legal document;
- `GET /documents/track_status/{track_id}` to wait for asynchronous indexing;
- `POST /query` with `include_references=true` and `include_chunk_content=true`.

All endpoint knowledge stays in this module so both command-line scripts remain small.

### Ingestion flow

`ingest_legal_docs.py` accepts a file or directory, discovers supported legal document extensions, uploads each file, and polls every returned tracking ID until all associated documents are `PROCESSED` or `FAILED`. It reports per-file results and exits nonzero when any upload or indexing operation fails.

The original basename becomes LightRAG's `file_path`, which is the stable human-readable citation source. Duplicate filenames remain explicit HTTP 409 failures rather than silently replacing indexed evidence.

### Query and citation flow

`query_legal_rag.py` sends the user's legal question in `hybrid` mode with reranking disabled, avoiding another model service in the minimal deployment. A separate `user_prompt` tells the answer model to:

- use only retrieved evidence;
- cite legal propositions with LightRAG reference IDs such as `[1]`;
- distinguish quoted law from interpretation;
- abstain when the indexed corpus is insufficient;
- include a short POC/not-legal-advice notice.

The script never trusts generated citation labels alone. It prints LightRAG's structured `references` array after the answer, including source paths and bounded chunk excerpts. A `--json` option returns the original structured result for future chatbot integration, and `--context-only` retrieves context without generating a final answer.

## Configuration

The POC reads its client settings from `legal_lightrag_poc/.env`:

- `LIGHTRAG_URL=http://127.0.0.1:9621`
- `LIGHTRAG_API_KEY=<local API key>`
- `LLM_BINDING_API_KEY=<provider key>`
- `EMBEDDING_BINDING_API_KEY=<provider key>`

Changing the embedding model or dimension after ingestion requires deleting the POC's persisted index and re-ingesting the documents. Provider keys must never be committed.

## Error handling

HTTP failures include the LightRAG error detail when available. Uploads have bounded request timeouts, queries have longer bounded timeouts, and indexing polling has a configurable overall deadline. Missing files, empty directories, invalid environment configuration, failed indexing records, and timeouts all produce readable errors and nonzero exit codes.

## Security and legal limitations

- Host exposure is loopback-only by default.
- The LightRAG API requires a user-selected API key.
- Source documents and generated indexes remain in local bind mounts, but document content is sent to the configured model provider.
- The generated answer is informational POC output, not legal advice.
- File-level citations demonstrate retrieval provenance; they do not prove that the model's interpretation is legally correct.
- A production system still needs jurisdiction scoping, source version/effective-date metadata, access control, audit logging, prompt-injection defenses, human review, and evaluation by qualified legal professionals.

## Verification

Unit tests mock the HTTP layer and prove request payloads, citation preservation, polling transitions, and error behavior. Python compilation verifies every script. Docker Compose configuration is validated when Docker is available. A live end-to-end query requires the user's provider key and is documented as an explicit manual verification step rather than simulated.

