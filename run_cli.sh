#!/bin/bash
# Wrapper for the OCR CLI: sets PYTHONPATH and uses the project venv so you can
# run it from the repo root without the ../../.venv prefix.
#
#   ./run_cli.sh --markdown sample.md --doc-type invoice
#   ./run_cli.sh invoice.pdf --doc-type invoice --vlm typhoon
#   ./run_cli.sh --help
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"   # ponytail: fall back to system python if venv missing

PYTHONPATH="$REPO_DIR/src/demo_ocr" exec "$PY" -m processing.cli "$@"
