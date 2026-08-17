#!/usr/bin/env python3
"""Ask a legal question and display LightRAG's structured citations."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Sequence

from legal_rag_client import (
    LightRAGClient,
    LightRAGError,
    format_query_result,
    load_client_from_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query indexed legal documents and return citations."
    )
    parser.add_argument("question", nargs="+", help="Legal question to ask")
    parser.add_argument(
        "--mode",
        choices=("local", "global", "hybrid", "naive", "mix"),
        default="hybrid",
        help="LightRAG retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Retrieve supporting context without generating a final answer",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON for chatbot or API integration",
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=500,
        help="Maximum displayed characters per retrieved chunk (default: 500)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], LightRAGClient] = load_client_from_env,
) -> int:
    args = build_parser().parse_args(argv)
    question = " ".join(args.question).strip()

    try:
        client = client_factory()
        client.health()
        result = client.query(
            question,
            mode=args.mode,
            context_only=args.context_only,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_query_result(result, excerpt_chars=args.excerpt_chars))
    except (LightRAGError, ValueError) as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
