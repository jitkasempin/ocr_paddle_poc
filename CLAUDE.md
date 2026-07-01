# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit multipage app that OCRs invoices, passports, packing lists, and similar documents, then extracts structured fields with LLMs and stores results in Supabase. Runs in Docker with an NVIDIA GPU. Despite the repo name, PaddleOCR is not the active backend (it's commented out in the Dockerfile).

## Commands

```bash
# Build & run (Docker, GPU required) — the normal way to run this app
docker compose build gradio3 && docker compose up -d gradio3   # serves on port 8506

# Reload after code changes (down → rebuild → up)
bash reload.sh

# Run directly (inside the container / a py3.12 env with deps installed)
streamlit run src/demo_ocr/main.py --server.port 8506 --server.address 0.0.0.0

pip install -r requirements.txt   # install deps
python src/demo_ocr/test_imports.py   # smoke-check that page imports resolve
```

There is no unit-test suite, linter, or build step beyond Docker. `test_imports.py` is an import smoke check, not a test framework.

## Architecture

- **`src/demo_ocr/main.py`** — the only entry point. Sets `st.set_page_config`, loads `style.css`, and renders a sidebar `selectbox` that dispatches to two pages: `ocr_processing_page()` (async, called via `asyncio.run`) and `invoice_viewer_page()`.
- **`processing/`** — the OCR pipeline and its Streamlit page (`ocr_page.py`), plus stamp detection, repetition/quality detection, schema validation (`schematron.py`, `po_schema.py`, `schema_helper.py`), and hybrid search.
- **`core/`** — shared building blocks: field/table extraction templates (`config.py`), extraction logic, prompts, PDF→markdown (`pdf2md/`), file converters, and vLLM/client wrappers.
- **`dots_ocr/`**, **`typhoon_ocr/`** — self-contained OCR backends (layout analysis, model inference, output cleaning). Multiple backends coexist; the active one is chosen in the UI.
- **`doc_classification/`** — zero-shot document typing via OpenCLIP.
- **`invoice_viewer/`** — read-only viewer page backed by Supabase.

## Conventions that will bite you

- **Imports are bare `processing.*` / `core.*`, NOT `demo_ocr.*`.** `main.py` does `sys.path.append(os.path.dirname(__file__))`, so `src/demo_ocr/` is the import root at runtime. The Dockerfile also sets `PYTHONPATH=/app/src`. Match the existing bare-package style when adding imports.
- **`st.set_page_config()` belongs only in `main.py`.** Calling it elsewhere breaks the multipage app.
- **CSS lives in `src/demo_ocr/style.css`**, injected by `main.py`'s `load_css`. `main.py` also contains a large block of the same rules commented out — edit the `.css` file, not the dead block. Theme is light purple (see also `RADIO_BUTTON_STYLING_GUIDE.md` and `.streamlit/config.toml`).
- **Extraction templates are data, in `core/config.py`** (`TEMPLATES_FIELDS`, `TEMPLATES_TABLES`) — add a document type by adding a template entry, not by branching in code.
- **Secrets come from a `.env` file** (git-ignored, loaded via `python-dotenv`) and mounted by `docker-compose.yml`. Never commit it.
- **The `README.md` under `src/demo_ocr/` is stale** — it describes an older `processing/` layout and pages ("INVOICE VIEWER") that no longer match `main.py` ("ITEM VIEWER"). Trust the code and this file over it.

## Docker notes

`docker-compose.yml` defines one service, `gradio3` (the name is legacy — it runs Streamlit). It requires the external `traefik` network, mounts `.env`, and reserves all NVIDIA GPUs. The Dockerfile builds a Python 3.12 venv at `/opt/venv`, installs Tesseract with MRZ data for passport scanning, and `docling-surya` is pip-installed at container start (see the compose `command`).
