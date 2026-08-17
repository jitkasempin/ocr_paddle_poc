from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest_legal_docs import discover_documents, main as ingest_main  # noqa: E402
from legal_rag_client import QueryResult, Reference  # noqa: E402
from query_legal_rag import main as query_main  # noqa: E402


class FakeQueryClient:
    def __init__(self):
        self.arguments = None

    def health(self):
        return {"status": "healthy"}

    def query(self, question, *, mode, context_only):
        self.arguments = {
            "question": question,
            "mode": mode,
            "context_only": context_only,
        }
        return QueryResult(
            response="The limitation period is two years [2].",
            references=(
                Reference(
                    reference_id="2",
                    file_path="/documents/limitations-act.pdf",
                    content=("Claims must be filed within two years.",),
                ),
            ),
            raw={
                "response": "The limitation period is two years [2].",
                "references": [
                    {
                        "reference_id": "2",
                        "file_path": "/documents/limitations-act.pdf",
                        "content": ["Claims must be filed within two years."],
                    }
                ],
                "response_time": 1.25,
            },
        )


class CliHelperTests(unittest.TestCase):
    def test_discover_documents_filters_extensions_and_sorts_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "z-contract.PDF").write_bytes(b"pdf")
            (root / "a-statute.md").write_text("law", encoding="utf-8")
            (root / "notes.csv").write_text("ignore", encoding="utf-8")

            documents = discover_documents(root, recursive=False)

        self.assertEqual(
            ["a-statute.md", "z-contract.PDF"],
            [document.name for document in documents],
        )

    def test_discover_documents_only_descends_when_recursive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "jurisdiction"
            nested.mkdir()
            (root / "act.txt").write_text("act", encoding="utf-8")
            (nested / "case.docx").write_bytes(b"docx")

            shallow = discover_documents(root, recursive=False)
            recursive = discover_documents(root, recursive=True)

        self.assertEqual(["act.txt"], [document.name for document in shallow])
        self.assertEqual(
            ["act.txt", "case.docx"], [document.name for document in recursive]
        )

    def test_ingest_main_rejects_directory_without_supported_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "spreadsheet.csv").write_text("not supported")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                status = ingest_main([temp_dir], client_factory=lambda: None)

        self.assertEqual(2, status)
        self.assertIn("No supported legal documents", stderr.getvalue())

    def test_query_main_json_output_preserves_structured_references(self):
        client = FakeQueryClient()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = query_main(
                ["How long is the limitation period?", "--json"],
                client_factory=lambda: client,
            )

        output = json.loads(stdout.getvalue())
        self.assertEqual(0, status)
        self.assertEqual("limitations-act.pdf", Path(output["references"][0]["file_path"]).name)
        self.assertEqual(
            ["Claims must be filed within two years."],
            output["references"][0]["content"],
        )
        self.assertEqual(1.25, output["response_time"])

    def test_query_main_forwards_mode_and_context_only(self):
        client = FakeQueryClient()

        with contextlib.redirect_stdout(io.StringIO()):
            status = query_main(
                ["Show the supporting context", "--mode", "naive", "--context-only"],
                client_factory=lambda: client,
            )

        self.assertEqual(0, status)
        self.assertEqual(
            {
                "question": "Show the supporting context",
                "mode": "naive",
                "context_only": True,
            },
            client.arguments,
        )


if __name__ == "__main__":
    unittest.main()
