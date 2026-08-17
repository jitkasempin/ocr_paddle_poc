#!/usr/bin/env python3
"""Upload legal source documents to a running LightRAG server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from legal_rag_client import (
    IndexingTimeout,
    LightRAGClient,
    LightRAGError,
    UploadReceipt,
    load_client_from_env,
)


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md", ".html", ".rtf"})


def discover_documents(source: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return supported documents in deterministic path order."""

    path = Path(source)
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {path}")

    candidates = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.is_file()
            and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda candidate: str(candidate).casefold(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload PDF, DOCX, TXT, Markdown, HTML, or RTF legal documents "
            "and wait for LightRAG indexing to finish."
        )
    )
    parser.add_argument("path", help="Document file or directory to ingest")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include supported files in nested directories",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2,
        help="Seconds between indexing-status checks (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Maximum seconds to wait for each indexing track (default: 1800)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], LightRAGClient] = load_client_from_env,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        documents = discover_documents(args.path, recursive=args.recursive)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    if not documents:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(
            f"No supported legal documents found. Supported extensions: {extensions}",
            file=sys.stderr,
        )
        return 2

    try:
        client = client_factory()
        health = client.health()
    except (LightRAGError, ValueError) as exc:
        print(f"LightRAG connection failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"LightRAG is reachable (status={health.get('status', 'unknown')}). "
        f"Uploading {len(documents)} document(s)."
    )
    receipts: list[UploadReceipt] = []
    failures = 0

    for document in documents:
        try:
            receipt = client.upload_document(document)
        except LightRAGError as exc:
            failures += 1
            print(f"UPLOAD FAILED  {document}: {exc}", file=sys.stderr)
            continue
        receipts.append(receipt)
        print(f"QUEUED         {document}  track={receipt.track_id}")

    for receipt in receipts:
        try:
            result = client.wait_for_track(
                receipt.track_id,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
        except (LightRAGError, IndexingTimeout, ValueError) as exc:
            failures += 1
            print(f"INDEX FAILED   {receipt.source}: {exc}", file=sys.stderr)
            continue

        if result.successful:
            print(f"INDEXED        {receipt.source}")
            continue

        failures += 1
        for document in result.failed_documents:
            source = document.get("file_path") or receipt.source.name
            error = document.get("error_msg") or "Unknown indexing error"
            print(f"INDEX FAILED   {source}: {error}", file=sys.stderr)

    succeeded = len(documents) - failures
    print(f"Finished: {succeeded} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
