"""Small REST client for the LightRAG legal-chat proof of concept."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import requests


LEGAL_GROUNDING_PROMPT = """
Answer as a legal-information assistant using only the retrieved document context.
For every legal rule, obligation, deadline, exception, or conclusion, cite one or
more supporting LightRAG reference IDs in square brackets, for example [1] or
[1][2]. Never invent a citation. Distinguish document text from your
interpretation and identify material conflicts between sources. If the indexed
documents do not contain enough evidence, say that the available corpus is
insufficient and do not fill the gap from general knowledge. End with a brief
notice that this proof-of-concept output is legal information, not legal advice.
""".strip()

_TERMINAL_STATUSES = {"processed", "failed"}
_CITATION_PATTERN = re.compile(r"\[(\d+)]")


class LightRAGError(RuntimeError):
    """Raised when the LightRAG server or client configuration fails."""


class IndexingTimeout(LightRAGError):
    """Raised when asynchronous document indexing exceeds its deadline."""


@dataclass(frozen=True)
class UploadReceipt:
    source: Path
    status: str
    message: str
    track_id: str


@dataclass(frozen=True)
class TrackResult:
    track_id: str
    documents: tuple[dict[str, Any], ...]
    status_summary: dict[str, int]

    @property
    def successful(self) -> bool:
        return bool(self.documents) and all(
            str(document.get("status", "")).lower() == "processed"
            for document in self.documents
        )

    @property
    def failed_documents(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            document
            for document in self.documents
            if str(document.get("status", "")).lower() == "failed"
        )


@dataclass(frozen=True)
class Reference:
    reference_id: str
    file_path: str
    content: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "file_path": self.file_path,
            "content": list(self.content),
        }


@dataclass(frozen=True)
class QueryResult:
    response: str
    references: tuple[Reference, ...]
    raw: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "references": [reference.as_dict() for reference in self.references],
        }


class LightRAGClient:
    """Synchronous client for the small subset of LightRAG used by the POC."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        request_timeout: float = 60,
        query_timeout: float = 300,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("LightRAG base URL cannot be empty")
        if not api_key.strip():
            raise ValueError("LightRAG API key cannot be empty")
        if request_timeout <= 0 or query_timeout <= 0:
            raise ValueError("HTTP timeouts must be greater than zero")

        self.base_url = normalized_url
        self.request_timeout = request_timeout
        self.query_timeout = query_timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "X-API-Key": api_key.strip(),
            }
        )
        self._sleep = sleep
        self._clock = clock

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                timeout=timeout or self.request_timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise LightRAGError(f"Cannot reach LightRAG at {url}: {exc}") from exc

        if response.status_code >= 400:
            detail = _response_detail(response)
            raise LightRAGError(
                f"{method.upper()} {path} failed with HTTP "
                f"{response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise LightRAGError(
                f"{method.upper()} {path} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise LightRAGError(
                f"{method.upper()} {path} returned JSON that is not an object"
            )
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def upload_document(self, source: str | Path) -> UploadReceipt:
        path = Path(source)
        if not path.is_file():
            raise LightRAGError(f"Document does not exist or is not a file: {path}")

        with path.open("rb") as stream:
            payload = self._request(
                "POST",
                "/documents/upload",
                files={"file": (path.name, stream)},
            )

        track_id = str(payload.get("track_id", "")).strip()
        if not track_id:
            raise LightRAGError(
                f"LightRAG accepted {path.name} but returned no tracking ID"
            )
        return UploadReceipt(
            source=path,
            status=str(payload.get("status", "")),
            message=str(payload.get("message", "")),
            track_id=track_id,
        )

    def track_status(self, track_id: str) -> TrackResult:
        normalized_track_id = track_id.strip()
        if not normalized_track_id:
            raise ValueError("Tracking ID cannot be empty")
        payload = self._request(
            "GET", f"/documents/track_status/{normalized_track_id}"
        )
        documents_value = payload.get("documents", [])
        documents = (
            tuple(dict(document) for document in documents_value)
            if isinstance(documents_value, list)
            else ()
        )
        summary_value = payload.get("status_summary", {})
        status_summary = (
            {str(key): int(value) for key, value in summary_value.items()}
            if isinstance(summary_value, Mapping)
            else {}
        )
        return TrackResult(
            track_id=str(payload.get("track_id", normalized_track_id)),
            documents=documents,
            status_summary=status_summary,
        )

    def wait_for_track(
        self,
        track_id: str,
        *,
        poll_interval: float = 2,
        timeout: float = 900,
    ) -> TrackResult:
        if poll_interval < 0 or timeout < 0:
            raise ValueError("Polling interval and timeout cannot be negative")
        deadline = self._clock() + timeout

        while True:
            result = self.track_status(track_id)
            statuses = {
                str(document.get("status", "")).lower()
                for document in result.documents
            }
            if result.documents and statuses.issubset(_TERMINAL_STATUSES):
                return result
            if self._clock() >= deadline:
                raise IndexingTimeout(
                    f"Indexing track {track_id} did not finish within {timeout:g} seconds"
                )
            self._sleep(poll_interval)

    def query(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        context_only: bool = False,
        user_prompt: str = LEGAL_GROUNDING_PROMPT,
    ) -> QueryResult:
        normalized_question = question.strip()
        if len(normalized_question) < 3:
            raise ValueError("Question must contain at least three characters")
        allowed_modes = {"local", "global", "hybrid", "naive", "mix"}
        if mode not in allowed_modes:
            raise ValueError(f"Unsupported LightRAG query mode: {mode}")

        payload = self._request(
            "POST",
            "/query",
            timeout=self.query_timeout,
            json={
                "query": normalized_question,
                "mode": mode,
                "only_need_context": context_only,
                "response_type": "Multiple Paragraphs",
                "enable_rerank": False,
                "include_references": True,
                "include_chunk_content": True,
                "stream": False,
                "user_prompt": user_prompt,
            },
        )

        references_value = payload.get("references", [])
        references: list[Reference] = []
        if isinstance(references_value, list):
            for item in references_value:
                if not isinstance(item, Mapping):
                    continue
                content_value = item.get("content", [])
                if isinstance(content_value, str):
                    content = (content_value,)
                elif isinstance(content_value, list):
                    content = tuple(str(chunk) for chunk in content_value)
                else:
                    content = ()
                references.append(
                    Reference(
                        reference_id=str(item.get("reference_id", "")),
                        file_path=str(item.get("file_path", "unknown_source")),
                        content=content,
                    )
                )

        return QueryResult(
            response=str(payload.get("response", "")),
            references=tuple(references),
            raw=payload,
        )


def load_client_from_env(env_path: str | Path | None = None) -> LightRAGClient:
    """Load the POC's `.env` file and construct a validated client."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise LightRAGError(
            "python-dotenv is required; run: pip install -r requirements.txt"
        ) from exc

    target = Path(env_path) if env_path else Path(__file__).with_name(".env")
    load_dotenv(target, override=False)
    base_url = os.getenv("LIGHTRAG_URL", "http://127.0.0.1:9621")
    api_key = os.getenv("LIGHTRAG_API_KEY", "")
    if not api_key.strip():
        raise LightRAGError(
            f"LIGHTRAG_API_KEY is missing. Copy .env.example to {target.name} "
            "and set a private value."
        )
    return LightRAGClient(
        base_url,
        api_key,
        request_timeout=_float_env("LIGHTRAG_REQUEST_TIMEOUT", 60),
        query_timeout=_float_env("LIGHTRAG_QUERY_TIMEOUT", 300),
    )


def format_query_result(result: QueryResult, *, excerpt_chars: int = 500) -> str:
    """Render generated prose plus the server's authoritative reference records."""

    if excerpt_chars < 1:
        raise ValueError("Excerpt length must be greater than zero")

    sections = ["Answer", result.response.strip() or "(No answer returned)"]
    known_ids = {reference.reference_id for reference in result.references}
    cited_ids = set(_CITATION_PATTERN.findall(result.response))
    if result.references and not cited_ids:
        sections.extend(
            [
                "",
                "Citation warning: the generated answer did not contain reference IDs; "
                "review the retrieved sources below before relying on it.",
            ]
        )
    unknown_ids = cited_ids - known_ids
    if unknown_ids:
        sections.extend(
            [
                "",
                "Citation warning: the answer used unknown reference IDs: "
                + ", ".join(sorted(unknown_ids)),
            ]
        )

    sections.extend(["", "Retrieved sources"])
    if not result.references:
        sections.append("No structured references were returned by LightRAG.")
        return "\n".join(sections)

    for reference in result.references:
        source_name = _source_name(reference.file_path)
        sections.append(f"[{reference.reference_id}] {source_name}")
        for chunk in reference.content:
            excerpt = " ".join(chunk.split())
            if len(excerpt) > excerpt_chars:
                excerpt = excerpt[: excerpt_chars - 1].rstrip() + "…"
            if excerpt:
                sections.append(f"  - {excerpt}")
    return "\n".join(sections)


def _source_name(file_path: str) -> str:
    normalized = file_path.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or "unknown_source"


def _response_detail(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        detail = payload.get("detail") or payload.get("message")
        if detail:
            return str(detail)
    text = str(getattr(response, "text", "")).strip()
    return text or "No error detail returned"


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LightRAGError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise LightRAGError(f"{name} must be greater than zero")
    return parsed
