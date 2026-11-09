import asyncio
import json
import logging
import threading
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_gateway import main
from rag_gateway.config import Settings, get_settings
from rag_gateway.models import RetrievalEvaluationRequest
from rag_gateway.observability import (
    ObservabilityMiddleware,
    RequestMetrics,
    processing_stage,
    request_metrics,
    request_stage,
)
from rag_gateway.operations import OperationsStore
from rag_gateway.rag import retrieve_context
from rag_gateway.store import AtlasStore


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


@pytest.mark.parametrize("rerank,hybrid", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("fail_vector", [False, True])
async def test_query_stage_timings_are_measured_without_double_counting(
    rerank: bool, hybrid: bool, fail_vector: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("rag_gateway.observability.time.perf_counter", lambda: clock["now"])
    settings = Settings(_env_file=None, retrieval_rerank_enabled=rerank,
                        retrieval_hybrid_enabled=hybrid)
    event_loop_thread = threading.get_ident()
    worker_threads = []
    calls = []

    class Embedder:
        async def embed(self, _texts: list[str], input_type: str) -> list[list[float]]:
            assert input_type == "query"
            clock["now"] += 0.01
            return [[0.1]]

    class Collection:
        def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
            worker_threads.append(threading.get_ident())
            calls.append(pipeline)
            if "$vectorSearch" in pipeline[0]:
                clock["now"] += 0.035 if rerank else 0.02
                if fail_vector:
                    raise PyMongoError("private-mongodb-error")
                return [{"_id": "doc", "document_id": "doc", "chunk_index": 0,
                         "text": "private-evidence", "score": 0.9}]
            clock["now"] += 0.03
            return [{"_id": "doc", "document_id": "doc", "chunk_index": 0,
                     "text": "private-evidence", "score": 3.0}]

    store = AtlasStore.__new__(AtlasStore)
    store.settings = settings
    store.collection = Collection()
    application = FastAPI()
    application.add_middleware(ObservabilityMiddleware)
    application.add_exception_handler(StarletteHTTPException, main.observed_http_exception)

    @application.post("/timed-query")
    async def query(request: Request) -> dict[str, int]:
        try:
            async with request_stage(request.scope, "retrieval"):
                sources = await retrieve_context("private-query", "default", settings, Embedder(), store)
        except PyMongoError as error:
            raise HTTPException(502, "Retrieval failed") from error
        async with request_stage(request.scope, "ollama"):
            clock["now"] += 0.04
        return {"source_count": len(sources)}

    request_id = f"stages-{rerank}-{hybrid}-{fail_vector}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.post("/timed-query", headers={"X-Request-ID": request_id})
    assert response.status_code == (502 if fail_vector else 200)
    event = next(item for item in request_metrics.snapshot()["recent_requests"]
                 if item["request_id"] == request_id)
    timings = event["details"]["stage_durations_ms"]
    assert timings["embedding"] == 10
    if fail_vector:
        assert "ollama" not in timings
    else:
        assert timings["ollama"] == 40
    vector_key = "atlas_vector_rerank" if rerank else "atlas_vector"
    assert timings[vector_key] == (35 if rerank else 20)
    assert ("atlas_lexical" in timings) is (hybrid and not fail_vector)
    if hybrid and not fail_vector:
        assert timings["atlas_lexical"] == 30
    leaf_total = sum(value for key, value in timings.items() if key != "retrieval")
    assert leaf_total == pytest.approx(event["duration_ms"])
    assert timings["retrieval"] == leaf_total - (0 if fail_vector else 40)
    assert len(calls) == (2 if hybrid and not fail_vector else 1)
    assert all(worker != event_loop_thread for worker in worker_threads)
    assert "private-query" not in json.dumps(event)
    assert "private-evidence" not in json.dumps(event)


async def test_active_query_exposes_current_stage_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("rag_gateway.observability.time.perf_counter", lambda: clock["now"])
    entered = asyncio.Event()
    release = asyncio.Event()
    application = FastAPI()
    application.add_middleware(ObservabilityMiddleware)

    @application.post("/active-query")
    async def query() -> dict[str, bool]:
        with processing_stage("embedding"):
            clock["now"] += 0.015
            entered.set()
            await release.wait()
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        task = asyncio.create_task(client.post("/active-query", headers={"X-Request-ID": "active-stage"}))
        await entered.wait()
        try:
            event = next(item for item in request_metrics.snapshot()["recent_requests"]
                         if item["request_id"] == "active-stage")
            assert event["status"] == "in progress"
            assert event["details"]["active_stage"] == {"name": "embedding", "elapsed_ms": 15}
        finally:
            release.set()
            await task


@pytest.mark.parametrize("fail_second", [False, True])
async def test_batch_evaluation_retains_individual_query_timings(tmp_path: Path, fail_second: bool) -> None:
    settings = Settings(_env_file=None)
    application = FastAPI()
    application.add_middleware(ObservabilityMiddleware)
    application.add_exception_handler(StarletteHTTPException, main.observed_http_exception)
    application.state.operations = OperationsStore(tmp_path / "operations.sqlite3")

    class Embedder:
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, _texts: list[str], input_type: str) -> list[list[float]]:
            self.calls += 1
            if fail_second and self.calls == 2:
                raise RuntimeError("private-embedding-error")
            return [[0.1]]

    class Store:
        def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            with processing_stage("atlas_vector"):
                return [{"document_id": "doc", "chunk_index": 0, "score": 0.9}]

    @application.post("/v1/evaluations/retrieval")
    async def evaluate(request: Request) -> dict[str, Any]:
        return await main.evaluate_retrieval(
            RetrievalEvaluationRequest(cases=[
                {"query": "private-question-one", "expected_document_ids": ["doc"]},
                {"query": "private-question-two", "expected_document_ids": ["doc"]},
            ]), request, None, settings, Store(), Embedder(),
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.post("/v1/evaluations/retrieval", headers={"X-Request-ID": "batch-stages"})
    assert response.status_code == (502 if fail_second else 200)
    event = next(item for item in request_metrics.snapshot()["recent_requests"]
                 if item["request_id"] == "batch-stages")
    cases = event["details"]["case_timings"]
    assert len(cases) == 2
    for index, case in enumerate(cases, start=1):
        assert case["case_index"] == index
        failed = fail_second and index == 2
        assert case["status"] == ("error" if failed else "success")
        assert case["duration_ms"] >= 0
        assert case["stage_durations_ms"]["embedding"] >= 0
        if failed:
            assert "atlas_vector" not in case["stage_durations_ms"]
        else:
            assert case["stage_durations_ms"]["atlas_vector"] >= 0
    assert "private-question" not in json.dumps(event)
