import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rag_gateway import main
from rag_gateway.config import Settings, get_settings
from rag_gateway.document_loader import (
    OLLAMA_OCR_PROMPT,
    IngestDocument,
    LoaderOptions,
    OllamaDocumentParser,
    load_directory,
    scan_directory,
)
from rag_gateway.main import _resolve_loader_directory


def test_startup_logger_emits_info_with_uvicorn_default_root_level() -> None:
    assert logging.getLogger().level >= logging.WARNING
    assert main.logger.getEffectiveLevel() == logging.INFO
    assert main.logger.name == "uvicorn.error"


def test_scan_directory_applies_recursive_include_and_exclude(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "guide.MD").write_text("guide", encoding="utf-8")
    (nested / "scan.pdf").write_bytes(b"pdf")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")

    scan = scan_directory(
        LoaderOptions(
            directory=tmp_path,
            include_extensions=("txt", ".md", ".pdf"),
            exclude_extensions=("md", ".exe"),
        )
    )

    assert [path.relative_to(tmp_path).as_posix() for path in scan.files] == [
        "nested/scan.pdf",
        "root.txt",
    ]
    assert scan.discovered == 4
    assert scan.skipped == 2


def test_scan_directory_can_disable_recursion(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.txt").write_text("child", encoding="utf-8")

    scan = scan_directory(LoaderOptions(directory=tmp_path, recursive=False))

    assert [path.name for path in scan.files] == ["root.txt"]
    assert scan.discovered == 1


def test_web_loader_directory_must_be_inside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    child = allowed_root / "child"
    child.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    assert _resolve_loader_directory(child, [allowed_root]) == child.resolve()
    with pytest.raises(HTTPException) as error:
        _resolve_loader_directory(outside, [allowed_root])
    assert error.value.status_code == 403


def test_document_loader_ui_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)

    with TestClient(main.app) as client:
        assert client.get("/tools/document-loader").status_code == 404
        settings.document_loader_enabled = True
        response = client.get("/tools/document-loader")

    assert response.status_code == 200
    assert "Local document loader" in response.text
    assert "/v1/document-loader/jobs" in response.text


async def test_lifespan_logs_ollama_models_and_atlas_connection(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured_requests: list[httpx.Request] = []
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb://atlas.test",
        mongodb_api_key="startup-test-key",
        document_loader_enabled=True,
        document_parser_model="glm-ocr:latest",
    )
    original_async_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.0] * settings.embedding_dimensions}]},
        )

    def mock_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(respond)
        return original_async_client(*args, **kwargs)

    class FakeStore:
        def __init__(self, _settings: Settings) -> None:
            pass

        def ping(self) -> bool:
            return True

        def close(self) -> None:
            pass

    class FakeParser:
        model = "glm-ocr:latest"

        def __init__(self, *_args: Any) -> None:
            pass

        def list_models(self) -> list[str]:
            return ["glm-ocr:latest", "llama3.2-vision:latest", settings.ollama_model]

        def close(self) -> None:
            pass

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "AtlasStore", FakeStore)
    monkeypatch.setattr(main, "OllamaDocumentParser", FakeParser)
    monkeypatch.setattr(main.httpx, "AsyncClient", mock_async_client)

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        async with main.lifespan(main.app):
            pass

    assert "Ollama connection check succeeded; discovered models:" in caplog.text
    assert "Configured Ollama document parser model 'glm-ocr:latest' is installed" in caplog.text
    assert (
        "MongoDB Atlas connection check succeeded: host=atlas.test "
        "database=rag_gateway collection=chunks"
    ) in caplog.text
    assert "Configured Ollama chat model 'qwen2.5:7b' is installed" in caplog.text
    assert "MongoDB AI API model check succeeded: model=voyage-3.5 dimensions=1024" in caplog.text
    assert len(captured_requests) == 1
    assert captured_requests[0].url == "https://ai.mongodb.com/v1/embeddings"
    assert captured_requests[0].headers["Authorization"] == "Bearer " + settings.mongodb_api_key
    assert "startup-test-key" not in caplog.text


def test_ollama_parser_sends_image_and_ocr_prompt() -> None:
    captured: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "parse:latest"}]})
        return httpx.Response(200, json={"message": {"content": "## Page title\nExtracted text"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        parser = OllamaDocumentParser("http://ollama.test/v1", "glm-ocr:latest", client=client)
        assert parser.list_models() == ["parse:latest"]
        result = parser.parse_image(b"test image bytes")

    assert result == "## Page title\nExtracted text"
    chat_body = captured[1]
    body = json.loads(chat_body.content)
    assert chat_body.url.path == "/api/chat"
    assert body["model"] == "glm-ocr:latest"
    assert body["messages"][0]["content"] == OLLAMA_OCR_PROMPT
    assert base64.b64decode(body["messages"][0]["images"][0]) == b"test image bytes"


def test_extracts_supported_text_formats_without_ollama(tmp_path: Path) -> None:
    text = tmp_path / "notes.md"
    text.write_text("# Notes\n\nLocal content", encoding="utf-8")
    html = tmp_path / "page.html"
    html.write_text("<main>Visible <b>text</b></main><script>hidden()</script>", encoding="utf-8")
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,value\nAlpha,42\n", encoding="utf-8")
    parser = OllamaDocumentParser("http://ollama.test", "glm-ocr:latest")
    try:
        assert parser.extract(text) == "# Notes\n\nLocal content"
        assert parser.extract(html) == "Visible\ntext"
        assert parser.extract(csv_file) == "name | value\nAlpha | 42"
    finally:
        parser.close()


async def test_load_directory_reports_per_file_and_overall_status(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("good source", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("bad source", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("a,b", encoding="utf-8")

    class FakeParser:
        model = "parse:latest"

        def extract(self, path: Path) -> str:
            if path.name == "bad.txt":
                raise ValueError("simulated parser failure")
            return path.read_text(encoding="utf-8")

    ingested: list[IngestDocument] = []
    events: list[dict[str, Any]] = []

    async def ingest(document: IngestDocument) -> dict[str, int]:
        ingested.append(document)
        return {"chunks_upserted": 1}

    summary = await load_directory(
        LoaderOptions(directory=tmp_path, tenant_id="tenant-a", include_extensions=("txt",)),
        FakeParser(),
        ingest,
        progress=events.append,
    )

    assert summary.status == "completed_with_errors"
    assert (summary.discovered, summary.eligible, summary.skipped) == (3, 2, 1)
    assert (summary.processed, summary.succeeded, summary.failed) == (2, 1, 1)
    assert ingested[0].document_id == "good.txt"
    assert ingested[0].tenant_id == "tenant-a"
    assert events[-1]["stage"] == "completed_with_errors"
    assert any(event["stage"] == "file_failed" for event in events)
