# Project knowledge

## What is this project?
A Streamlit multipage app for OCR processing of invoices, passports, packing lists, and other documents. Uses multiple OCR backends (Typhoon OCR, DOTS OCR, Surya, Docling, Tesseract/MRZ) with LLM-powered structured extraction (OpenAI, Qwen, Google GenAI, LiteLLM, Ollama). Stores results in Supabase. Runs in Docker with NVIDIA GPU support.

## Quickstart
- **Build & run:** `docker compose build gradio3 && docker compose up -d gradio3`
- **Quick reload:** `bash reload.sh`
- **Local dev (inside container):** `streamlit run src/demo_ocr/main.py --server.port 8506`
- **Install deps:** `pip install -r requirements.txt`

## Architecture
- **Entry point:** `src/demo_ocr/main.py` — Streamlit app with sidebar nav (OCR Processing / Item Viewer)
- **`src/demo_ocr/processing/`** — OCR pipeline pages and logic (ocr_page.py, ocr.py, stamp_detection, schema validation)
- **`src/demo_ocr/core/`** — Shared config (field templates), extraction, prompts, PDF-to-markdown, file converters, vLLM/client wrappers
- **`src/demo_ocr/dots_ocr/`** — DOTS OCR model inference, layout analysis, output cleaning
- **`src/demo_ocr/typhoon_ocr/`** — Typhoon OCR utilities and PDF helpers
- **`src/demo_ocr/doc_classification/`** — Zero-shot document classification via OpenCLIP
- **`src/demo_ocr/invoice_viewer/`** — Invoice/item viewer page (reads from Supabase)
- **Styling:** `src/demo_ocr/style.css` + `.streamlit/config.toml` (light purple theme)
- **Docker:** Single service `gradio3`, PyTorch base image, Python 3.12 venv, CUDA/GPU, Traefik labels
- **PYTHONPATH:** `/app/src` (set in Dockerfile) — imports use `demo_ocr.*` paths

## Key conventions
- Python 3.12, async where needed (asyncio.run in main.py)
- Streamlit for UI — `st.set_page_config` only in main.py
- Environment variables via `.env` file (not committed)
- Field/table templates defined in `src/demo_ocr/core/config.py`
- Custom CSS loaded from `src/demo_ocr/style.css`

## Things to avoid
- Don't commit `.env` or any secrets
- Don't call `st.set_page_config()` outside of `main.py`
- Don't install packages globally — use the project's `requirements.txt`
