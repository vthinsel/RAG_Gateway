import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_gateway import main
from rag_gateway.config import Settings, get_settings
from rag_gateway.observability import ObservabilityMiddleware, RequestMetrics, request_metrics


def test_monitoring_dashboard_shows_requests_and_requires_admin(
    monkeypatch,
) -> None:
    admin_key = "monitoring-admin-key"
    tenant_key = "monitoring-tenant-key"
    settings = Settings(
        _env_file=None,
        gateway_api_key=admin_key,
        api_key_registry={sha256(tenant_key.encode()).hexdigest(): ["tenant-a"]},
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    try:
        with TestClient(main.app) as client:
            page = client.get("/tools/monitoring")
            denied = client.get("/v1/admin/monitoring")
            tenant_denied = client.get(
                "/v1/admin/monitoring",
                headers={"Authorization": f"Bearer {tenant_key}"},
            )
            health = client.get("/healthz?private-search=must-not-be-recorded")
            response = client.get(
                "/v1/admin/monitoring",
                headers={"Authorization": f"Bearer {admin_key}"},
            )

        assert page.status_code == 200
        assert "Live API monitoring" in page.text
        assert "/v1/admin/monitoring" in page.text
        assert denied.status_code == 401
        assert tenant_denied.status_code == 401
        assert response.status_code == 200
        snapshot = response.json()
        observed = next(
            item
            for item in snapshot["recent_requests"]
            if item["request_id"] == health.headers["x-request-id"]
        )
        assert observed["route"] == "/healthz"
        assert observed["status"] == 200
        assert observed["duration_ms"] >= 0
        assert snapshot["requests_total"] >= 1
        assert snapshot["retained_request_limit"] == 500
        assert response.headers["cache-control"] == "no-store"
        assert "must-not-be-recorded" not in response.text
    finally:
        main.app.dependency_overrides.pop(get_settings, None)


def test_handled_502_logs_upstream_cause_without_secret_content(caplog) -> None:
    application = FastAPI()
    application.add_middleware(ObservabilityMiddleware)
    application.add_exception_handler(StarletteHTTPException, main.observed_http_exception)

    @application.get("/failure")
    async def failure() -> None:
        upstream_request = httpx.Request(
            "POST", "http://user:private-password@ollama.test/chat?key=private-query"
        )
        upstream_response = httpx.Response(503, request=upstream_request, text="private-body")
        try:
            upstream_response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise HTTPException(502, "Gateway upstream unavailable") from error

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        TestClient(application) as client,
    ):
        response = client.get("/failure", headers={"X-Request-ID": "diagnostic-test"})
    assert response.status_code == 502
    event = next(item for item in request_metrics.snapshot()["recent_requests"]
                 if item["request_id"] == "diagnostic-test")
    assert event["outcome"] == "error"
    assert event["details"]["error"]["type"] == "HTTPStatusError"
    assert event["details"]["error"]["upstream_status"] == 503
    assert event["details"]["error"]["upstream_host"] == "ollama.test"
    assert '"request_id":"diagnostic-test"' in caplog.text
    combined = caplog.text + json.dumps(event)
    assert "private-password" not in combined
    assert "private-query" not in combined
    assert "private-body" not in combined


@pytest.mark.parametrize("kind", ["success", "http_error", "timeout", "invalid_json", "stream_error", "mongodb_error"])
async def test_chat_monitoring_records_processing_and_errors_without_content(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        _env_file=None, gateway_api_key="private-admin-token", mongodb_uri="", mongodb_api_key="",
        ollama_base_url="http://upstream.test/v1", ollama_model="test:latest",
        operations_db_path=tmp_path / "operations.sqlite3",
    )

    class FakeParser:
        def __init__(self, *_args: Any) -> None:
            pass

        def list_models(self) -> list[str]:
            return [settings.ollama_model]

        def close(self) -> None:
            pass

    async def retrieve(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        if kind == "mongodb_error":
            raise PyMongoError("private-mongodb-uri")
        return [{"document_id": "doc", "chunk_index": 0, "text": "private-evidence", "score": 0.9}]

    def upstream(request: httpx.Request) -> httpx.Response:
        if kind in {"http_error", "stream_error"}:
            return httpx.Response(503, text="private-upstream-body", request=request)
        if kind == "timeout":
            raise httpx.ReadTimeout("private-timeout-message", request=request)
        if kind == "invalid_json":
            return httpx.Response(200, text="private-invalid-json", request=request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "private-answer [S1]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }, request=request)

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, main.get_store, lambda: object())
    monkeypatch.setitem(main.app.dependency_overrides, main.get_embedder, lambda: object())
    monkeypatch.setattr(main, "OllamaDocumentParser", FakeParser)
    monkeypatch.setattr(main, "retrieve_context", retrieve)
    request_id = f"processing-{kind}"
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        async with main.lifespan(main.app):
            await main.app.state.integration_task
            previous_http = main.app.state.http
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
                main.app.state.http = upstream_client
                try:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=main.app), base_url="http://gateway.test"
                    ) as client:
                        response = await client.post("/v1/chat/completions", headers={
                            "Authorization": "Bearer private-admin-token", "X-Request-ID": request_id,
                        }, json={
                            "messages": [{"role": "user", "content": "private-prompt"}],
                            "temperature": 0.1, "top_k": 3, "stream": kind == "stream_error",
                        })
                finally:
                    main.app.state.http = previous_http
    event = next(item for item in request_metrics.snapshot()["recent_requests"]
                 if item["request_id"] == request_id)
    details = event["details"]
    assert details["model"] == "test:latest"
    if kind != "mongodb_error":
        assert details["source_count"] == 1
        assert details["generation_options"]["temperature"] == 0.1
    assert details["retrieval_options"]["top_k"] == 3
    assert details["stage_durations_ms"]["retrieval"] >= 0
    assert details["response_bytes"] > 0
    assert response.headers["x-request-id"] == request_id
    if kind == "success":
        assert event["outcome"] == "success"
        assert details["tokens"]["total_tokens"] == 30
        assert details["finish_reason"] == "stop"
        assert details["stage"] == "completed"
    else:
        assert event["outcome"] == "error"
        assert details["error"]["status"] == 502
        assert response.status_code == (200 if kind == "stream_error" else 502)
        if kind in {"http_error", "stream_error"}:
            assert details["error"]["upstream_status"] == 503
            assert details["error"]["upstream_host"] == "upstream.test"
        if kind == "timeout":
            assert details["error"]["category"] == "upstream_timeout"
        if kind == "mongodb_error":
            assert details["error"]["category"] == "mongodb_error"
            assert details["stage"] == "retrieval"
        assert any(record.levelno == logging.ERROR and request_id in record.message
                   for record in caplog.records)
    serialized = json.dumps(event) + caplog.text
    for secret in ("private-admin-token", "private-evidence", "private-upstream-body",
                   "private-timeout-message", "private-invalid-json", "private-answer", "private-prompt",
                   "private-mongodb-uri"):
        assert secret not in serialized


def test_http_200_stream_failure_counts_as_monitoring_error() -> None:
    metrics = RequestMetrics()
    metrics.observe("POST", "/v1/chat/completions", 200, 0.1, details={
        "error": {"status": 502, "category": "upstream_timeout"},
    })
    snapshot = metrics.snapshot()
    assert snapshot["requests_failed"] == 1
    assert snapshot["routes"][0]["errors"] == 1
    assert snapshot["recent_requests"][0]["status"] == 200
    assert snapshot["recent_requests"][0]["outcome"] == "error"
