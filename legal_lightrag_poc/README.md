# LightRAG Legal Chat POC

This is the quickest practical path to a local legal-document chatbot with LightRAG citations:

- one pinned LightRAG Docker container;
- OpenAI-compatible hosted models, so no local GPU is required;
- lightweight local JSON, NetworkX, and NanoVectorDB persistence;
- one script to ingest legal documents;
- one script to answer or retrieve context with structured source references.

LightRAG's REST server is the integration boundary. Your eventual chat UI can call the same `/query` endpoint or reuse `legal_rag_client.py`.

> This is a technical proof of concept, not a legal-advice system. A citation proves which indexed file LightRAG retrieved; it does not prove that the generated interpretation is legally correct.

## 1. Prerequisites

- Docker Desktop with Docker Compose
- Python 3.10 or newer
- An API key for the configured OpenAI-compatible LLM and embedding service

The sample uses `gpt-5.4-mini` and `text-embedding-3-large`. You can replace them with models exposed by your provider, but the embedding model and `EMBEDDING_DIM` must match.

## 2. Configure the POC

In PowerShell:

```powershell
cd legal_lightrag_poc
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Edit `.env` and replace:

1. `LIGHTRAG_API_KEY` with the random value printed above.
2. Both `LLM_BINDING_API_KEY` and `EMBEDDING_BINDING_API_KEY` with your provider key.

The `.env` file and generated `data/` directory are ignored by Git.

Create a small Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Start LightRAG

```powershell
docker compose up -d
docker compose ps
```

Wait until the service is healthy. Then open:

- Web UI: [http://127.0.0.1:9621/webui](http://127.0.0.1:9621/webui)
- API documentation: [http://127.0.0.1:9621/docs](http://127.0.0.1:9621/docs)

If startup fails, inspect the service log:

```powershell
docker compose logs --tail 100 lightrag
```

## 4. Ingest legal documents

Upload one document:

```powershell
python ingest_legal_docs.py C:\legal-corpus\employment-act.pdf
```

Upload every supported document below a directory:

```powershell
python ingest_legal_docs.py C:\legal-corpus --recursive
```

Supported sample formats are PDF, DOCX, TXT, Markdown, HTML, and RTF. The script:

1. verifies that LightRAG is reachable;
2. uploads each document through `/documents/upload`;
3. keeps the original filename as the citation source;
4. polls `/documents/track_status/{track_id}`;
5. exits nonzero if an upload or indexing task fails.

Indexing is the expensive step because LightRAG extracts entities and relationships with the LLM. Start with 3–10 representative legal documents for the POC rather than a full corpus.

## 5. Ask a legal question with citations

```powershell
python query_legal_rag.py "What notice period applies before termination?"
```

The script uses `hybrid` retrieval and asks LightRAG to include:

- a generated, corpus-grounded answer;
- numbered reference IDs;
- the source file for each reference;
- retrieved text excerpts for human checking.

Example shape:

```text
Answer
The agreement requires 30 days' written notice [1].

Retrieved sources
[1] employment-agreement.pdf
  - Either party may terminate this agreement by giving 30 days...
```

If the model omits citation markers or cites an unknown ID, the script prints a warning while still showing LightRAG's authoritative structured references.

### Retrieve context without a generated answer

```powershell
python query_legal_rag.py "termination notice" --mode naive --context-only
```

### Return JSON for a chat application

```powershell
python query_legal_rag.py "What are the tenant's repair duties?" --json
```

The JSON shape is:

```json
{
  "response": "Answer with [1] citations...",
  "references": [
    {
      "reference_id": "1",
      "file_path": "/app/data/inputs/tenancy-act.pdf",
      "content": ["Retrieved supporting chunk..."]
    }
  ]
}
```

Your chat UI should render citation markers from `response`, then resolve them against `references`. Do not derive sources only from model-generated prose.

## Choosing a query mode

- `hybrid` — recommended default here; combines entity and relationship retrieval.
- `naive` — direct vector retrieval, useful for exact clauses and `--context-only` checks.
- `local` — entity-focused questions.
- `global` — broad relationship or theme questions.
- `mix` — combines graph and vector retrieval; most useful after adding a reranker.

Select another mode with `--mode`, for example `--mode naive`.

## Stop, restart, or reset

Stop the container while preserving the index:

```powershell
docker compose down
```

Restart it later with `docker compose up -d`.

Changing the embedding model, dimension, or embedding behavior invalidates the existing vectors. To make that change, stop the service, back up anything important, remove only `legal_lightrag_poc/data/`, and ingest the corpus again. The same re-ingestion rule applies when you want already-indexed files to use a different parser route.

## POC limitations before production

Add these before using the design with real users or confidential legal material:

- jurisdiction and effective-date metadata;
- document-level authorization and tenant isolation;
- encryption, audit logs, retention, and deletion controls;
- malware scanning and prompt-injection defenses for uploaded files;
- a production storage backend and backups;
- retrieval and citation evaluations over lawyer-reviewed questions;
- human escalation and clear legal-information disclaimers;
- provider/privacy review, because document text is sent to the configured model provider.

Official references: [LightRAG Server and WebUI](https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md), [Docker deployment](https://github.com/HKUDS/LightRAG/blob/main/docs/DockerDeployment.md), and [LightRAG repository](https://github.com/HKUDS/LightRAG).
