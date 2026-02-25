# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Invoice OCR system using multiple OCR engines (PaddleOCR, Surya, Tesseract) combined with LLM-powered field extraction. Built with Streamlit frontend, Docker containerization with NVIDIA GPU support, Qdrant for vector search, and Supabase for data storage.

## Commands

### Running the Application

**Local development:**
```bash
cd src/demo_ocr
streamlit run main.py
```

**Docker (recommended for full GPU support):**
```bash
docker-compose down && docker-compose build gradio3 && docker-compose up -d gradio3
```
App runs on port 8506.

**Quick reload scripts:**
```bash
./reload.sh              # Full Docker reload
./reload_streamlit.sh    # Streamlit-only reload
```

### Dependencies

```bash
pip install -r requirements.txt
```

Special packages installed separately in Docker:
- `paddlepaddle-gpu` and `paddleocr[all]` for PaddleOCR
- `docling-surya` for document parsing

## Architecture

### Core Pipeline Flow

```
PDF/Image Upload → Document Classification (OpenCLIP) → Multi-Engine OCR
    → LLM Field Extraction → Repetition Detection → Confidence Scoring
    → Human Review (Streamlit UI) → Save to Supabase + Qdrant
```

### Key Directories

- `src/demo_ocr/` - Main application
  - `main.py` - Streamlit multipage app entry point
  - `processing/` - OCR pipeline and orchestration
    - `ocr_no_flash.py` - Main OCR orchestration engine
    - `ocr_page.py` - OCR processing UI page
    - `invoice_integration.py` - Validation and deduplication
    - `repetition_detector.py` - Duplicate detection
    - `hybrid_search.py` - Vector + keyword search via Qdrant
    - `qwen_client.py` - Async Qwen3 LLM client
  - `core/` - Core utilities
    - `client.py` - VLM/LLM API client (LiteLLM wrapper)
    - `extract.py` - Field and table extraction logic
    - `config.py` - Template field definitions
    - `file_converters/pdf_converter.py` - PDF to image conversion
  - `invoice_viewer/` - Invoice data viewer page
  - `doc_classification/` - Document type classification with OpenCLIP
  - `dots_ocr/` - DOTS OCR integration for layout-aware processing
  - `typhoon_ocr/` - Typhoon OCR integration

### Data Flow

1. **File conversion** (`core/file_converters/`) - PDFs converted to images via PyMuPDF
2. **Classification** (`doc_classification/zero_shot.py`) - OpenCLIP zero-shot classification determines document type
3. **OCR** - Multiple engines (Surya primary, Paddle fallback) extract raw text
4. **Extraction** (`core/extract.py`) - LLM extracts structured fields based on document type templates
5. **Validation** (`processing/invoice_integration.py`) - Repetition detection, field validation, confidence scoring
6. **Storage** - Validated data to Supabase, embeddings to Qdrant

### LLM Integration

Multiple LLM backends supported via `core/client.py`:
- Qwen3-14B via RunPod (`VLM_MODEL_URL` env var)
- OpenAI GPT models
- Local models via Ollama
- vLLM endpoints

### Vector Search

`processing/hybrid_search.py` implements hybrid search:
- Semantic search via Qdrant embeddings
- Keyword matching via RapidFuzz
- Combined ranking for invoice retrieval

## Configuration

### Environment Variables

Required in `.env`:
- Database and API credentials
- `VLM_MODEL_URL` - Vision LLM endpoint (defaults to RunPod)
- `AGENT_RUN_ID` - Debug identifier

### Streamlit Theme

Configured in `.streamlit/config.toml` with purple theme (#9B7EBD primary color).

## Document Types

Supported document types with corresponding extraction templates:
- Invoice
- Packing List
- Passport
- Certificate
- Bank Statement

Each type has specific field templates defined in `core/config.py`.

## GPU Requirements

Docker container uses NVIDIA runtime with CUDA. The `docker-compose.yml` reserves all available GPUs for accelerated OCR and model inference.
