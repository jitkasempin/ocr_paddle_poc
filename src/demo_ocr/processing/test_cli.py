"""Offline self-checks for the OCR CLI (no GPU / no network).

Run: python -m processing.test_cli   (from src/demo_ocr, or with PYTHONPATH=src/demo_ocr)
"""
from __future__ import annotations

from . import cli
from .schemas import DOC_TYPE_SCHEMAS, Invoice


def test_normalize_payload():
    # plain dict passes through
    assert cli._normalize_to_json_payload({"a": 1}) == {"a": 1}
    # fenced json block is unwrapped
    assert cli._normalize_to_json_payload('```json\n{"x": 2}\n```') == {"x": 2}
    # bare json string parses
    assert cli._normalize_to_json_payload('{"y": 3}') == {"y": 3}
    # pydantic model -> dict (json mode so Decimal is serialisable)
    payload = cli._normalize_to_json_payload(Invoice())
    assert isinstance(payload, dict) and "summary" in payload
    # non-json string is preserved, not lost
    assert cli._normalize_to_json_payload("not json") == {"raw_output": "not json"}


def test_doc_type_dispatch():
    assert set(DOC_TYPE_SCHEMAS) == {"invoice", "markdown", "passport", "packing_list", "stock_boj5"}
    for schema in DOC_TYPE_SCHEMAS.values():
        # every schema must expose a JSON schema (used in the extraction prompt)
        assert "properties" in schema.model_json_schema()


def test_invoice_schema_healing():
    # Thai-date healing + decimal healing run through the extracted validators
    inv = Invoice.model_validate({
        "document": {"date": "5 มกราคม 2567"},
        "items": [{"description": "widget", "amount": "1,234.50", "quantity": 2}],
        "summary": {"total_amount": "1,234.50"},
    })
    assert inv.document.date == "05.01.2024"
    assert str(inv.items[0].amount) == "1234.50"
    assert inv.summary.total_amount == inv.items[0].amount


def test_end_to_end_markdown(monkeypatch=None):
    # Full driver path with OCR + Qwen stubbed: markdown -> extract -> dict.
    import asyncio
    from pathlib import Path
    import tempfile

    async def fake_extract(markdown, schema, use_new_qwen=True):
        assert markdown.strip() == "# hello"
        assert schema is Invoice
        return {"ok": True}

    orig = cli.extract_structured
    cli.extract_structured = fake_extract
    try:
        with tempfile.TemporaryDirectory() as d:
            md = Path(d) / "in.md"
            md.write_text("# hello", encoding="utf-8")
            out = asyncio.run(cli.run(
                input_path=str(md), doc_type="invoice", vlm="typhoon",
                page_number=0, markdown_path=None, use_new_qwen=True,
            ))
            assert out == {"ok": True}
    finally:
        cli.extract_structured = orig


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL OK")


if __name__ == "__main__":
    _run()
