import base64
import json
import logging
import time
from hashlib import sha256
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
    assert "Document loader" in response.text
    assert "/v1/document-loader/jobs" in response.text


def test_document_preview_extracts_without_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        gateway_api_key="test-key",
        document_loader_enabled=True,
    )
    model_checks = 0

    class FakeParser:
        def __init__(self, _url: str, model: str, *_args: Any) -> None:
            self.model = model

        def list_models(self) -> list[str]:
            nonlocal model_checks
            model_checks += 1
            return ["glm-ocr:latest"]

        def extract(self, path: Path) -> str:
            return path.read_text(encoding="utf-8")

        def close(self) -> None:
            pass

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "OllamaDocumentParser", FakeParser)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    try:
        with TestClient(main.app) as client:
            page = client.get("/tools/document-loader")
            denied = client.post(
                "/v1/document-loader/preview",
                files={"file": ("notes.txt", b"Preview this text")},
            )
            preview = client.post(
                "/v1/document-loader/preview",
                headers={"Authorization": "Bearer test-key"},
                files={"file": ("notes.txt", b"Preview this text")},
            )
            unsupported = client.post(
                "/v1/document-loader/preview",
                headers={"Authorization": "Bearer test-key"},
                files={"file": ("notes.exe", b"not supported")},
            )

        assert page.status_code == 200
        assert "Preview extracted text before upload" in page.text
        assert denied.status_code == 401
        assert preview.status_code == 200
        assert preview.json() == {
            "filename": "notes.txt",
            "extension": ".txt",
            "characters": len("Preview this text"),
            "text": "Preview this text",
        }
        assert unsupported.status_code == 415
        assert model_checks == 1
    finally:
        main.app.dependency_overrides.pop(get_settings, None)


def test_gateway_admin_key_and_registered_keys_can_access_loader_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_key = "gateway-admin-key"
    tenant_key = "tenant-loader-key"
    settings = Settings(
        _env_file=None,
        gateway_api_key=admin_key,
        api_key_registry={sha256(tenant_key.encode()).hexdigest(): ["tenant-a"]},
        document_loader_enabled=True,
    )

    class FakeParser:
        def __init__(self, *_args: Any) -> None:
            pass

        def list_models(self) -> list[str]:
            return ["glm-ocr:latest"]

        def close(self) -> None:
            pass

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "OllamaDocumentParser", FakeParser)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    try:
        with TestClient(main.app) as client:
            admin_response = client.get(
                "/v1/document-loader/models",
                headers={"Authorization": f"Bearer {admin_key}"},
            )
            tenant_response = client.get(
                "/v1/document-loader/models",
                headers={"Authorization": f"Bearer {tenant_key}"},
            )
            denied_response = client.get(
                "/v1/document-loader/models",
                headers={"Authorization": "Bearer invalid-key"},
            )

        assert admin_response.status_code == 200
        assert tenant_response.status_code == 200
        assert denied_response.status_code == 401
    finally:
        main.app.dependency_overrides.pop(get_settings, None)


def test_document_loader_accepts_browser_uploads_and_ingests_relative_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        gateway_api_key="test-key",
        document_loader_enabled=True,
    )
    ingested_documents: list[tuple[str, str, str]] = []

    class FakeStore:
        def replace_document(
            self,
            tenant_id: str,
            document_id: str,
            chunks: list[str],
            _embeddings: list[list[float]],
            _metadata: dict[str, Any],
        ) -> int:
            ingested_documents.append((tenant_id, document_id, "\n".join(chunks)))
            return len(chunks)

    class FakeEmbedder:
        async def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
            assert input_type == "document"
            return [[0.1, 0.2] for _ in texts]

    class FakeParser:
        model = "glm-ocr:latest"

        def __init__(self, *_args: Any) -> None:
            pass

        def list_models(self) -> list[str]:
            return [self.model]

        def extract(self, path: Path) -> str:
            return path.read_text(encoding="utf-8")

        def close(self) -> None:
            pass

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "OllamaDocumentParser", FakeParser)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, main.get_store, lambda: FakeStore())
    monkeypatch.setitem(main.app.dependency_overrides, main.get_embedder, lambda: FakeEmbedder())
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/v1/document-loader/uploads",
                headers={"Authorization": "Bearer test-key"},
                data={
                    "relative_paths": json.dumps(
                        ["lab/architecture.md", "diagram.mmd", "ignored.exe"]
                    ),
                    "tenant_id": "default",
                },
                files=[
                    ("files", ("architecture.md", b"VM inventory", "text/markdown")),
                    ("files", ("diagram.mmd", b"flowchart TD\n  A --> B", "text/plain")),
                    ("files", ("ignored.exe", b"binary", "application/octet-stream")),
                ],
            )
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
            for _ in range(50):
                job_response = client.get(
                    f"/v1/document-loader/jobs/{job_id}",
                    headers={"Authorization": "Bearer test-key"},
                )
                job = job_response.json()
                if job["status"] in {"completed", "completed_with_errors", "failed"}:
                    break
                time.sleep(0.01)

            traversal_response = client.post(
                "/v1/document-loader/uploads",
                headers={"Authorization": "Bearer test-key"},
                data={"relative_paths": json.dumps(["../outside.md"]), "tenant_id": "default"},
                files=[("files", ("outside.md", b"must not escape the temporary directory"))],
            )

        assert job["status"] == "completed"
        assert job["succeeded"] == 2
        assert job["skipped"] == 1
        assert traversal_response.status_code == 422
        assert {document_id for _, document_id, _ in ingested_documents} == {
            "lab/architecture.md",
            "diagram.mmd",
        }
        assert all(tenant_id == "default" for tenant_id, _, _ in ingested_documents)
    finally:
        main.app.dependency_overrides.pop(get_settings, None)
        main.app.dependency_overrides.pop(main.get_store, None)
        main.app.dependency_overrides.pop(main.get_embedder, None)


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

        def vector_index_status(self) -> tuple[bool, bool]:
            return True, True

        def list_tenant_ids(self) -> list[str]:
            return ["default", "lab"]

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
    assert "Atlas Vector Search index is ready: name=rag_chunks_vector" in caplog.text
    assert "Atlas collection tenants: default, lab" in caplog.text
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


@pytest.mark.parametrize("extension", [".png", ".jpg", ".jpeg"])
def test_extracts_ollama_supported_images_without_pillow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
) -> None:
    image = tmp_path / f"diagram{extension}"
    image.write_bytes(b"encoded image bytes")
    parser = OllamaDocumentParser("http://ollama.test", "glm-ocr:latest")
    parsed_images: list[bytes] = []
    monkeypatch.setattr(parser, "parse_image", lambda value: parsed_images.append(value) or "OCR text")

    assert parser.extract(image) == "OCR text"
    assert parsed_images == [b"encoded image bytes"]


def test_extracts_supported_text_formats_without_ollama(tmp_path: Path) -> None:
    text = tmp_path / "notes.md"
    text.write_text("# Notes\n\nLocal content", encoding="utf-8")
    mermaid = tmp_path / "diagram.mmd"
    mermaid.write_text("flowchart TD\n  A --> B", encoding="utf-8")
    html = tmp_path / "page.html"
    html.write_text("<main>Visible <b>text</b></main><script>hidden()</script>", encoding="utf-8")
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,value\nAlpha,42\n", encoding="utf-8")
    parser = OllamaDocumentParser("http://ollama.test", "glm-ocr:latest")
    try:
        assert parser.extract(text) == "# Notes\n\nLocal content"
        assert parser.extract(mermaid) == "flowchart TD\n  A --> B"
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
