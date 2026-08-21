# LightRAG Legal Chat POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a quick-start LightRAG legal-document POC with citation-aware ingestion and retrieval scripts.

**Architecture:** Run the official LightRAG API server in an isolated Docker Compose project and keep all application integration in a small Python REST client. Two thin command-line scripts expose document ingestion and legal Q&A while preserving LightRAG's structured references.

**Tech Stack:** Docker Compose, LightRAG v1.5.6, Python 3.10+, requests, python-dotenv, unittest

**Spec:** `docs/superpowers/specs/2026-08-17-lightrag-legal-poc-design.md`

## Global Constraints

- Keep all POC runtime files under `legal_lightrag_poc/`.
- Do not modify the existing OCR container or application flow.
- Reuse `requests` and `python-dotenv`, which are already repository dependencies.
- Bind the host port to `127.0.0.1` and require a LightRAG API key.
- Pin the LightRAG image to `v1.5.6`.
- Preserve provider credentials only in the ignored `.env` file.
- Always request LightRAG references and chunk contents for user-visible answers.
- Treat the output as informational POC content, not legal advice.

---

### Task 1: REST client behavior

**Files:**
- Create: `legal_lightrag_poc/legal_rag_client.py`
- Create: `legal_lightrag_poc/tests/test_legal_rag_client.py`

**Interfaces:**
- Produces: `LightRAGClient`, `LightRAGError`, `IndexingTimeout`, `QueryResult`, `Reference`, `load_client_from_env()`, and `format_query_result()`.
- Consumes: `requests.Session`, `LIGHTRAG_URL`, and `LIGHTRAG_API_KEY`.

- [x] **Step 1: Write failing client tests**

Cover URL normalization, API-key headers, upload payloads, HTTP error details, pending-to-processed polling, failed indexing, query flags, structured reference preservation, and citation formatting using `unittest.mock`.

- [x] **Step 2: Run the tests and verify the expected import failure**

Run: `python -m unittest discover -s legal_lightrag_poc/tests -v`

Expected: FAIL because `legal_rag_client` does not exist.

- [x] **Step 3: Implement the minimal reusable client**

Implement the endpoint calls behind one `_request()` helper, dataclasses for normalized query output, polling with an injected sleep function, and deterministic console formatting with bounded excerpts.

- [x] **Step 4: Run the client tests**

Run: `python -m unittest discover -s legal_lightrag_poc/tests -v`

Expected: PASS.

### Task 2: Ingestion and query commands

**Files:**
- Create: `legal_lightrag_poc/ingest_legal_docs.py`
- Create: `legal_lightrag_poc/query_legal_rag.py`
- Create: `legal_lightrag_poc/tests/test_cli_helpers.py`

**Interfaces:**
- Consumes: the Task 1 client interfaces.
- Produces: `discover_documents(path, recursive)` and executable `main()` functions.

- [x] **Step 1: Write failing command-helper tests**

Test deterministic extension filtering, non-recursive versus recursive discovery, empty input handling, legal prompt contents, and structured JSON serialization.

- [x] **Step 2: Run the tests and verify the expected import failure**

Run: `python -m unittest discover -s legal_lightrag_poc/tests -v`

Expected: FAIL because the command modules do not exist.

- [x] **Step 3: Implement ingestion**

Discover `.pdf`, `.docx`, `.txt`, `.md`, `.html`, and `.rtf` files, perform a health check, upload each file, wait for all tracks, print a compact summary, and return exit status 1 on any failure.

- [x] **Step 4: Implement retrieval**

Accept a positional question plus `--mode`, `--context-only`, and `--json`; call `query()` with the legal grounding prompt; render the answer and authoritative structured sources.

- [x] **Step 5: Run all tests**

Run: `python -m unittest discover -s legal_lightrag_poc/tests -v`

Expected: PASS.

### Task 3: Deployment and quick start

**Files:**
- Create: `legal_lightrag_poc/docker-compose.yml`
- Create: `legal_lightrag_poc/.env.example`
- Create: `legal_lightrag_poc/requirements.txt`
- Create: `legal_lightrag_poc/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: LightRAG v1.5.6 environment variables and the Task 1 client settings.
- Produces: a locally reachable LightRAG server at `http://127.0.0.1:9621`.

- [x] **Step 1: Add the pinned Compose service and safe environment template**

Use loopback host publishing, persistent POC-local data mounts, `X-API-Key` authentication, OpenAI-compatible LLM/embedding placeholders, JSON/NetworkX/NanoVectorDB POC storage, and no reranker dependency.

- [x] **Step 2: Add dependency and secret hygiene**

List only `requests` and `python-dotenv`; ignore `legal_lightrag_poc/.env` and `legal_lightrag_poc/data/` explicitly.

- [x] **Step 3: Write the quick-start guide**

Document environment creation, server startup, health verification, ingestion, querying, context-only retrieval, JSON integration, reset/re-index behavior, and production/legal caveats.

- [x] **Step 4: Validate configuration and syntax**

Run: `python -m compileall -q legal_lightrag_poc`

Run: `python -m unittest discover -s legal_lightrag_poc/tests -v`

Run when Docker is available: `docker compose --env-file legal_lightrag_poc/.env.example -f legal_lightrag_poc/docker-compose.yml config --quiet`

Expected: all available checks pass.

### Task 4: Completion audit

**Files:**
- Review all files under `legal_lightrag_poc/` and both design documents.

**Interfaces:**
- Consumes: every earlier task deliverable.
- Produces: requirement-by-requirement completion evidence.

- [x] **Step 1: Verify no secrets or placeholders are accidentally executable defaults**

Search for real-looking API keys and confirm `.env` and persisted data are ignored.

- [x] **Step 2: Verify the documented commands match the implemented CLI help**

Run: `python legal_lightrag_poc/ingest_legal_docs.py --help`

Run: `python legal_lightrag_poc/query_legal_rag.py --help`

- [x] **Step 3: Re-run the full test and syntax suite**

Run the Task 3 verification commands and inspect their exit status.

- [x] **Step 4: Report the live-test boundary honestly**

State that a real ingestion/query depends on a provider key and Docker daemon, and provide the exact one-command next steps.
