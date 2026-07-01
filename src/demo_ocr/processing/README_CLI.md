# OCR CLI

Headless port of the Streamlit OCR-processing page (`ocr_page.py :: ocr_processing_page`).
Same two-stage pipeline, no UI / stamp detection / DELTA item-parsing / database writes.

1. **OCR** — document file → markdown, via a remote VLM (`typhoon` / `dots_ocr` / `olmocr` / `lighton`). Needs the heavy stack (torch, docling, typhoon_ocr, …) + GPU, imported lazily and only for image/PDF input.
2. **Extract** — markdown → JSON, via `Qwen3VLLMClient` structured output with a doc-type Pydantic schema. Light deps only.

Passing `--markdown FILE` (or a `.md` input) skips stage 1.

## Setup

```bash
cd /root/ocr_paddle_poc
python3 -m venv .venv
.venv/bin/pip install pydantic openai loguru python-dateutil   # extraction-only deps
```

## Run

Use the wrapper from the repo root (sets `PYTHONPATH` + venv automatically):

```bash
./run_cli.sh --markdown sample.md --doc-type invoice      # markdown in -> JSON out (skips OCR)
./run_cli.sh notes.md --doc-type passport                 # .md auto-detected, skips OCR
./run_cli.sh invoice.pdf --doc-type invoice --vlm typhoon # full pipeline (GPU + heavy deps)
./run_cli.sh --markdown notes.md --doc-type invoice -o out.json   # write to file
./run_cli.sh --markdown notes.md --doc-type invoice --old-qwen    # legacy Qwen3-14B
./run_cli.sh --help
```

Or invoke the module directly:

```bash
cd src/demo_ocr && PYTHONPATH=. python -m processing.cli --markdown sample.md
```

## Options

| Flag | Values | Default |
|------|--------|---------|
| `input` (positional) | PDF / PNG / JPG / MD | — |
| `--markdown FILE` | skip OCR, read text from FILE | — |
| `--doc-type` | `invoice`, `markdown`, `passport`, `packing_list`, `stock_boj5` | `invoice` |
| `--vlm` | `typhoon`, `dots_ocr`, `olmocr`, `lighton` | `typhoon` |
| `--page` | PDF page number for OCR | `0` |
| `--old-qwen` | use legacy Qwen3-14B extractor | off |
| `-o, --output` | write JSON to file | stdout |

## Test (offline, no GPU / no network)

```bash
cd src/demo_ocr && PYTHONPATH=. python -m processing.test_cli
```

## Notes

- The remote endpoints (Qwen in `qwen_client.py`, VLMs in `ocr_no_flash.py`) are hardcoded RunPod URLs — the same ones the Streamlit app uses. If extraction hangs, the serverless worker is likely scaled-to-zero/offline; that's infra, not the CLI.
- Full PDF/image OCR only works inside the Docker/GPU container where the heavy deps are installed.
