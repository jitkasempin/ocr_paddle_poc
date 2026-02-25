# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Invoice OCR system ("IEAT Pre-approve System") — a Streamlit multipage app that processes PDF/image documents through multiple OCR engines, extracts structured data (invoices, passports, bank statements, packing lists), classifies documents, and stores results in Qdrant and Supabase.

## Running the Application

```bash
# Local development (requires PYTHONPATH set to src/)
PYTHONPATH=src streamlit run src/demo_ocr/main.py --server.port 8506

# Docker build & run
docker compose build gradio3
docker compose up -d gradio3

# Full rebuild cycle
./reload.sh
```

The Docker container uses a Python 3.12 venv at `/opt/venv` inside a PyTorch+CUDA base image. The compose command activates the venv before running Streamlit.

## Architecture

### Entry Point & Page Routing
- `src/demo_ocr/main.py` — Streamlit entry point. Sidebar navigation routes to two pages: "OCR PROCESSING" (`processing/ocr_page.py`) and "ITEM VIEWER" (`invoice_viewer/invoice_viewer_page.py`).

### Core OCR Pipeline (`src/demo_ocr/core/`)
Framework-agnostic extraction layer (originally from a Gradio-based tool):
- `client.py` — LLM/VLM API calls via `litellm`. Routes to hosted vLLM (RunPod), Ollama, OpenRouter, or OpenAI based on model name prefix (`hosted_vllm/`, `ollama/`, `openrouter`, `gpt`).
- `extract.py` — `extract_information()` orchestrates parallel field + table extraction from document images. Fields get confidence scores (High/Low).
- `prompts.py` — Builds multimodal messages (text + base64 images) for field/table extraction.
- `config.py` — Template definitions for document types (invoice, passport) with field schemas.
- `vllm.py` — `VLLMServer` class to start/stop a local vLLM server process.
- `pdf2md/pdf2md.py` — Streaming PDF-to-Markdown conversion via VLM with sync fallback.

### OCR Processing Page (`src/demo_ocr/processing/`)
- `ocr_page.py` — Main Streamlit OCR page. Handles PDF upload, page-by-page OCR, document classification, structured data extraction, YOLO-based layout detection, and Supabase writes. ~900 lines, the largest file.
- `ocr.py` — `OCR` class wrapping multiple backends: Typhoon OCR (RunPod), olmOCR (RunPod), Donut (document classification), Gemini (structured output via `outlines`), and Qwen3 (structured extraction).
- `qwen_client.py` — Async OpenAI-compatible client for Qwen3-14B on RunPod.
- `invoice_integration.py` — Repetition detection for OCR output validation. `OCROutputValidator` flags corrupted/looping OCR results.
- `repetition_detector.py` — Low-level repetition detection algorithms.
- `hybrid_search.py` — `HybridSearch` class combining semantic (GritLM-7B embeddings via vLLM) + fuzzy (rapidfuzz) search against Qdrant vector DB.
- `schema_helper.py` — Parsing helpers for Thai dates, decimal values, codes.

### Typhoon OCR (`src/demo_ocr/typhoon_ocr/`)
Adapted from allenai/olmocr (Apache 2.0). Provides `prepare_ocr_messages()` for building OCR prompts with three task types: `default` (markdown tables), `structure` (HTML tables + figure analysis), `v1.5` (clean markdown, Thai figure descriptions). Handles PDF rendering via poppler (`pdftoppm`, `pdfinfo`).

### DotsOCR (`src/demo_ocr/dots_ocr/`)
Document layout parser supporting vLLM and HuggingFace backends. `DotsOCRParser` handles image/PDF parsing with layout detection, grounding OCR, and markdown output. Uses thread pools for multi-page PDF processing.

### Document Classification (`src/demo_ocr/doc_classification/`)
- `zero_shot.py` — `ZeroShotCentroidClassifier` using OpenCLIP embeddings with centroid-based cosine similarity matching. Open-set classification with per-label thresholds.
- `ml_caller/openclip_backend.py` — OpenCLIP model wrapper for multi-scale image embedding.

### Invoice Viewer (`src/demo_ocr/invoice_viewer/`)
- `invoice_viewer_page.py` — Streamlit page to browse Qdrant collection (`ieat_production_embeddings`), display items table, and export CSV.

## Key External Services

- **RunPod** — Hosts VLM models (Typhoon OCR, olmOCR, Qwen2.5-VL, Qwen3-14B)
- **Qdrant** — Vector database for semantic search (collection: `ieat_production_embeddings`)
- **Supabase** — Relational storage for processed invoice data
- **Ollama** — Local/remote LLM inference (Qwen models)
- **Gemini** — Structured output extraction via `outlines`

## Environment & Dependencies

- Python 3.12 with CUDA support (PyTorch 2.9 base image)
- System deps: `poppler-utils` (PDF rendering), `tesseract-ocr` (MRZ passport data), `libvips`
- Key Python packages: `streamlit`, `litellm`, `paddleocr`, `ultralytics` (YOLO), `open-clip-torch`, `outlines`, `olmocr`, `surya-ocr`, `docling`, `qdrant-client`, `rapidfuzz`
- Config via `.env` file (not committed) and `.streamlit/config.toml` (purple theme)

## Important Patterns

- OCR model selection happens at runtime in the UI via radio buttons; the `OCR` class initializes all backends on construction.
- `PYTHONPATH` must include `src/` so that imports like `from core.client import ...` and `from typhoon_ocr import ...` resolve correctly.
- The app processes documents asynchronously — `ocr_processing_page()` is an async function run via `asyncio.run()`.
- Document types and their extraction fields are defined in `core/config.py` (`TEMPLATES_FIELDS`, `TEMPLATES_TABLES`).
