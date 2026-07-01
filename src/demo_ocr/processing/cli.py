"""Command-line OCR + structured-extraction tool.

The headless equivalent of the ``ocr_processing_page`` Streamlit page, minus the
UI, stamp detection, DELTA-specific item parsing, and database writes.

Two stages, mirroring the app:

  1. OCR    : document file -> markdown, via a remote VLM (typhoon / dots_ocr /
              lighton / olmocr). Needs the heavy ``OCR`` class (torch, docling,
              typhoon_ocr, ...), so it is imported lazily and only when an
              image/PDF is given.
  2. EXTRACT: markdown -> JSON, via ``Qwen3VLLMClient`` structured output with a
              doc-type Pydantic schema. Light deps only (pydantic + openai).

Pass ``--markdown FILE`` (or a ``.md`` INPUT) to skip stage 1 entirely.

Usage:
    python -m processing.cli invoice.pdf --doc-type invoice --vlm typhoon
    python -m processing.cli notes.md --doc-type invoice
    python -m processing.cli --markdown notes.md --doc-type passport -o out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .qwen_client import Qwen3VLLMClient
from .schemas import DOC_TYPE_SCHEMAS


# --- stage 2: extraction (no heavy deps) ---------------------------------
def _normalize_to_json_payload(raw_output):
    """Coerce Qwen output into a JSON-serialisable payload (from ocr_page.py)."""
    if isinstance(raw_output, (dict, list)):
        return raw_output
    if isinstance(raw_output, BaseModel):
        return raw_output.model_dump(mode="json")
    if isinstance(raw_output, str):
        cleaned = raw_output.strip()
        fenced = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw_output": raw_output}
    return {"raw_output": str(raw_output)}


async def extract_structured(markdown: str, schema: type[BaseModel], use_new_qwen: bool = True):
    """Markdown -> JSON dict, replicating OCR.structured_output()."""
    prompt = f"""
        Convert the given text into JSON Object that conforms to the following schema: {schema.model_json_schema()}.
        Do not hallucinate and do not use any information that is not in the text.
        If the information is not available, use empty string.

        Given text: {markdown}
        /no_think
        """
    client = Qwen3VLLMClient()
    messages = [{"role": "user", "content": prompt}]
    if use_new_qwen:
        raw = await client.chat_completion(messages, schema)
    else:
        raw = await client.chat_completion_old(messages, schema)
    return _normalize_to_json_payload(raw)


# --- stage 1: OCR (heavy; lazy) ------------------------------------------
async def ocr_to_markdown(file_path: str, vlm: str, page_number: int) -> str:
    """Document file -> markdown, via the remote VLM. Imports OCR lazily."""
    from .ocr_no_flash import OCR  # heavy: torch/docling/typhoon_ocr/fastmrz

    model = OCR()
    if vlm == "typhoon":
        return await model.typhoon_runpod_predict(file_path, "default", page_number or 1)
    if vlm == "dots_ocr":
        return await model.dotsocr_runpod_predict(file_path)
    if vlm == "olmocr":
        return await model.olmocr_runpod_predict(file_path, page_number)
    if vlm == "lighton":
        return model.call_runpod_serverless_sync(file_path, page_number)
    raise ValueError(f"unknown VLM backend: {vlm!r}")


# --- driver ---------------------------------------------------------------
async def run(
    input_path: Optional[str],
    doc_type: str,
    vlm: str,
    page_number: int,
    markdown_path: Optional[str],
    use_new_qwen: bool,
) -> dict:
    schema = DOC_TYPE_SCHEMAS[doc_type]

    # Resolve markdown: explicit --markdown, a .md input, or OCR the file.
    if markdown_path:
        markdown = Path(markdown_path).read_text(encoding="utf-8")
    elif input_path and input_path.lower().endswith(".md"):
        markdown = Path(input_path).read_text(encoding="utf-8")
    elif input_path:
        print(f"[ocr] {input_path} via {vlm} (page {page_number}) ...", file=sys.stderr)
        markdown = await ocr_to_markdown(input_path, vlm, page_number)
    else:
        raise SystemExit("error: provide an INPUT file or --markdown FILE")

    if not markdown or not markdown.strip():
        raise SystemExit("error: OCR produced no text")

    print(f"[extract] {doc_type} -> schema {schema.__name__} ...", file=sys.stderr)
    return await extract_structured(markdown, schema, use_new_qwen)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="processing.cli",
        description="OCR a document and extract structured JSON (headless port of the Streamlit OCR page).",
    )
    p.add_argument("input", nargs="?", help="PDF / PNG / JPG / MD to process")
    p.add_argument("--markdown", help="Skip OCR; read markdown/text from this file")
    p.add_argument("--doc-type", default="invoice", choices=sorted(DOC_TYPE_SCHEMAS),
                   help="Extraction schema to use (default: invoice)")
    p.add_argument("--vlm", default="typhoon", choices=["typhoon", "dots_ocr", "olmocr", "lighton"],
                   help="OCR backend for image/PDF input (default: typhoon)")
    p.add_argument("--page", type=int, default=0, help="Page number for PDF OCR (default: 0)")
    p.add_argument("--old-qwen", action="store_true", help="Use the legacy Qwen3-14B extractor")
    p.add_argument("-o", "--output", help="Write JSON here (default: stdout)")
    args = p.parse_args(argv)

    result = asyncio.run(run(
        input_path=args.input,
        doc_type=args.doc_type,
        vlm=args.vlm,
        page_number=args.page,
        markdown_path=args.markdown,
        use_new_qwen=not args.old_qwen,
    ))

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[done] wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
