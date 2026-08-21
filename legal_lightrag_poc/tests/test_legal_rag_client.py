from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag_client import (  # noqa: E402
    IndexingTimeout,
    LightRAGClient,
    LightRAGError,
    format_query_result,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise json.JSONDecodeError("invalid", "", 0)
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.headers: dict[str, str] = {}
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs):
        files = kwargs.get("files")
        uploaded = None
        if files:
            filename, stream = files["file"]
            uploaded = (filename, stream.read())

        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": kwargs.get("json"),
                "uploaded": uploaded,
                "timeout": kwargs.get("timeout"),
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        return self.responses.pop(0)


class LightRAGClientTests(unittest.TestCase):
    def test_health_uses_normalized_url_and_api_key(self):
        session = FakeSession([FakeResponse(200, {"status": "healthy"})])
        client = LightRAGClient(
            "http://127.0.0.1:9621/", "poc-secret", session=session
        )

        health = client.health()

        self.assertEqual({"status": "healthy"}, health)
        self.assertEqual("poc-secret", session.headers["X-API-Key"])
        self.assertEqual(
            "http://127.0.0.1:9621/health", session.calls[0]["url"]
        )

    def test_upload_document_sends_the_original_basename_and_bytes(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "status": "success",
                        "message": "queued",
                        "track_id": "upload-123",
                    },
                )
            ]
        )
        client = LightRAGClient("http://lightrag", "key", session=session)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "employment-act.pdf"
            source.write_bytes(b"legal document bytes")

            receipt = client.upload_document(source)

        self.assertEqual("upload-123", receipt.track_id)
        self.assertEqual("employment-act.pdf", receipt.source.name)
        self.assertEqual(
            ("employment-act.pdf", b"legal document bytes"),
            session.calls[0]["uploaded"],
        )
        self.assertEqual(
            "http://lightrag/documents/upload", session.calls[0]["url"]
        )

    def test_http_error_includes_lightrag_detail(self):
        session = FakeSession(
            [FakeResponse(409, {"detail": "Document storage already contains this file"})]
        )
        client = LightRAGClient("http://lightrag", "key", session=session)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "duplicate.pdf"
            source.write_bytes(b"duplicate")

            with self.assertRaisesRegex(
                LightRAGError, "409.*Document storage already contains this file"
            ):
                client.upload_document(source)

    def test_wait_for_track_observes_pending_then_processed(self):
        pending = {
            "track_id": "track-1",
            "documents": [
                {
                    "id": "doc-1",
                    "status": "PENDING",
                    "file_path": "statute.pdf",
                    "error_msg": None,
                }
            ],
            "total_count": 1,
            "status_summary": {"PENDING": 1},
        }
        processed = {
            **pending,
            "documents": [
                {
                    "id": "doc-1",
                    "status": "PROCESSED",
                    "file_path": "statute.pdf",
                    "error_msg": None,
                }
            ],
            "status_summary": {"PROCESSED": 1},
        }
        session = FakeSession(
            [FakeResponse(200, pending), FakeResponse(200, processed)]
        )
        client = LightRAGClient("http://lightrag", "key", session=session)

        result = client.wait_for_track("track-1", poll_interval=0, timeout=2)

        self.assertTrue(result.successful)
        self.assertEqual("PROCESSED", result.documents[0]["status"])
        self.assertEqual(2, len(session.calls))

    def test_wait_for_track_returns_failed_document_details(self):
        failed = {
            "track_id": "track-2",
            "documents": [
                {
                    "id": "doc-2",
                    "status": "FAILED",
                    "file_path": "contract.pdf",
                    "error_msg": "Parser failed",
                }
            ],
            "total_count": 1,
            "status_summary": {"FAILED": 1},
        }
        session = FakeSession([FakeResponse(200, failed)])
        client = LightRAGClient("http://lightrag", "key", session=session)

        result = client.wait_for_track("track-2", poll_interval=0, timeout=2)

        self.assertFalse(result.successful)
        self.assertEqual("Parser failed", result.failed_documents[0]["error_msg"])

    def test_wait_for_track_times_out_while_document_is_pending(self):
        pending = {
            "track_id": "track-3",
            "documents": [
                {
                    "id": "doc-3",
                    "status": "PROCESSING",
                    "file_path": "case.pdf",
                    "error_msg": None,
                }
            ],
            "total_count": 1,
            "status_summary": {"PROCESSING": 1},
        }
        session = FakeSession([FakeResponse(200, pending)])
        client = LightRAGClient("http://lightrag", "key", session=session)

        with self.assertRaisesRegex(IndexingTimeout, "track-3"):
            client.wait_for_track("track-3", poll_interval=0, timeout=0)

    def test_track_status_rejects_a_malformed_documents_field(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "track_id": "track-malformed",
                        "documents": {"status": "PROCESSED"},
                        "total_count": 1,
                        "status_summary": {"PROCESSED": 1},
                    },
                )
            ]
        )
        client = LightRAGClient("http://lightrag", "key", session=session)

        with self.assertRaisesRegex(
            LightRAGError, "track-status response.*documents.*list"
        ):
            client.track_status("track-malformed")

    def test_query_requests_references_and_preserves_chunk_boundaries(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "response": "The notice period is 30 days [1].",
                        "references": [
                            {
                                "reference_id": "1",
                                "file_path": "/documents/employment-act.pdf",
                                "content": ["Section 10 requires 30 days.", "Exceptions apply."],
                            }
                        ],
                    },
                )
            ]
        )
        client = LightRAGClient("http://lightrag", "key", session=session)

        result = client.query("What notice period applies?")

        payload = session.calls[0]["json"]
        self.assertEqual("hybrid", payload["mode"])
        self.assertTrue(payload["include_references"])
        self.assertTrue(payload["include_chunk_content"])
        self.assertFalse(payload["enable_rerank"])
        self.assertIn("cite", payload["user_prompt"].lower())
        self.assertEqual("1", result.references[0].reference_id)
        self.assertEqual(
            ("Section 10 requires 30 days.", "Exceptions apply."),
            result.references[0].content,
        )

    def test_format_query_result_shows_authoritative_source_and_excerpt(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "response": "A claim must be filed within two years [7].",
                        "references": [
                            {
                                "reference_id": "7",
                                "file_path": "/documents/limitations-act.pdf",
                                "content": ["An action shall be brought within two years."],
                            }
                        ],
                    },
                )
            ]
        )
        result = LightRAGClient(
            "http://lightrag", "key", session=session
        ).query("How long do I have to file?")

        rendered = format_query_result(result, excerpt_chars=200)

        self.assertIn("A claim must be filed within two years [7].", rendered)
        self.assertIn("[7] limitations-act.pdf", rendered)
        self.assertIn("An action shall be brought within two years.", rendered)


if __name__ == "__main__":
    unittest.main()
