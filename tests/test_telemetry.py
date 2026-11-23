import asyncio
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any

import anyio.lowlevel
import httpx
import pytest
from anyio import CancelScope
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from rag_gateway import main
from rag_gateway.config import Settings, get_settings
from rag_gateway.document_loader import OllamaDocumentParser
from rag_gateway.operations import OperationsStore
from rag_gateway.store import AtlasStore
from rag_gateway.telemetry import (
    Telemetry,
    persist_model_usage,
    record_model_usage,
    request_telemetry,
    usage_prometheus,
    usage_tenant,
)
from rag_gateway.voyage import VoyageEmbedder


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/api/chat"])
async def test_chat_traces_and_persists_each_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, endpoint: str, hybrid: bool,
) -> None:
    settings = Settings(
        _env_file=None, gateway_api_key="test-admin", mongodb_api_key="private-api-key",
        api_key_registry={sha256(b"tenant-key").hexdigest(): ["tenant-a"]},
        voyage_embedding_model="embedding-model", embedding_dimensions=2,
        ollama_model="chat-model", ollama_base_url="http://upstream.test/v1",
        retrieval_rerank_enabled=True, retrieval_rerank_model="rerank-model",
        retrieval_hybrid_enabled=hybrid,
        operations_db_path=tmp_path / "ops.sqlite3",
        model_pricing={
            "mongodb_ai:embedding:embedding-model": {"input_per_million_usd": 0.1},
            "ollama:generation:chat-model": {
                "input_per_million_usd": 1.0, "output_per_million_usd": 2.0,
            },
            "atlas:rerank:rerank-model": {"per_request_usd": 0.001},
        },
    )
    operations = OperationsStore(settings.operations_db_path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"service.name": "test-gateway"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime = Telemetry(settings, operations, provider)
    requests: list[httpx.Request] = []
    upstream_usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"total_tokens": 12},
            })
        if stream:
            assert json.loads(request.content)["stream_options"] == {"include_usage": True}
            events = [
                {"choices": [{"delta": {"content": "private-answer"}, "finish_reason": None}]},
                {"choices": [], "usage": upstream_usage},
                {"choices": [], "usage": upstream_usage},
            ]
            return httpx.Response(200, text="".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ) + "data: [DONE]\n\n")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "private-answer"}, "finish_reason": "stop"}],
            "usage": upstream_usage,
        })

    class Database:
        def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{**item, "rerank_score": 0.8}
                    for item in pipeline[0]["$documents"]["$literal"]]

    class Collection:
        database = Database()

        def aggregate(self, _pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{
                "document_id": "doc", "chunk_index": 0, "text": "private-evidence",
                "score": 0.9, "rerank_score": 0.8,
            }]

    store = AtlasStore.__new__(AtlasStore)
    store.settings = settings
    store.collection = Collection()
    monkeypatch.setattr(main.app.state, "telemetry", runtime, raising=False)
    monkeypatch.setattr(main.app.state, "operations", operations, raising=False)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, main.get_store, lambda: store)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        monkeypatch.setattr(main.app.state, "http", upstream_client, raising=False)
        monkeypatch.setitem(
            main.app.dependency_overrides, main.get_embedder,
            lambda: VoyageEmbedder(settings, upstream_client),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://gateway.test",
        ) as client:
            response = await client.post(f"{endpoint}?secret=private-query", headers={
                "Authorization": "Bearer test-admin", "X-Request-ID": "usage-request",
                "X-Tenant-ID": "tenant-a",
                "traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
                "baggage": "secret=private-baggage",
            }, json={
                "messages": [{"role": "user", "content": "private-prompt"}],
                "stream": stream,
            })
            assert response.status_code == 200
            usage_response = await client.get(
                "/v1/admin/usage?tenant_id=tenant-a",
                headers={"Authorization": "Bearer test-admin"},
            )
            metrics = await client.get("/metrics", headers={"Authorization": "Bearer test-admin"})
            assert (await client.get("/metrics")).status_code == 401
            assert (await client.get("/v1/admin/usage")).status_code == 401
            assert (await client.get(
                "/metrics", headers={"Authorization": "Bearer tenant-key"},
            )).status_code == 401

    groups = {item["operation"]: item for item in usage_response.json()["groups"]}
    assert set(groups) == {"embedding", "rerank", "generation"}
    assert all(item["calls"] == 1 for item in groups.values())
    assert groups["embedding"]["input_tokens"] == 12
    assert groups["embedding"]["estimated_cost_usd"] == pytest.approx(0.0000012)
    assert groups["generation"]["total_tokens"] == 30
    assert groups["generation"]["estimated_cost_usd"] == pytest.approx(0.00004)
    assert groups["rerank"]["total_tokens"] is None
    assert groups["rerank"]["missing_usage_calls"] == 1
    assert groups["rerank"]["estimated_cost_usd"] == 0.001
    assert "rag_gateway_model_tokens_total" in metrics.text
    assert 'tenant_id="tenant-a"' in metrics.text
    assert groups == {
        row["operation"]: row for row in OperationsStore(settings.operations_db_path).model_usage_summary()
    }

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    root = by_name[f"POST {endpoint}"]
    assert root.parent is not None
    assert root.parent.span_id == int("2222222222222222", 16)
    assert root.context.trace_id == int("1" * 32, 16)
    assert by_name["embedding"].parent.span_id == by_name["retrieval"].context.span_id
    rerank_stage = "atlas_hybrid_rerank" if hybrid else "atlas_vector_rerank"
    assert by_name[rerank_stage].parent.span_id == by_name["retrieval"].context.span_id
    assert by_name[rerank_stage].attributes["gen_ai.request.model"] == "rerank-model"
    assert by_name["ollama_stream" if stream else "ollama"].parent.span_id == root.context.span_id
    assert root.attributes["rag.request_id"] == "usage-request"
    assert all("traceparent" in request.headers for request in requests)
    assert all("baggage" not in request.headers for request in requests)
    with sqlite3.connect(settings.operations_db_path) as connection:
        records = connection.execute(
            "SELECT request_id, trace_id, pricing_json FROM model_usage"
        ).fetchall()
    assert all(row[0] == "usage-request" and row[1] == "1" * 32 for row in records)
    serialized = repr([(span.attributes, span.events, span.status.description) for span in spans])
    serialized += repr(records) + metrics.text + usage_response.text
    for private in (
        "private-prompt", "private-answer", "private-evidence", "private-api-key",
        "private-query", "private-baggage",
    ):
        assert private not in serialized
    runtime.close()


def test_missing_usage_prices_and_invalid_counts_remain_unknown(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, model_pricing={
        "ollama:generation:input-only": {"input_per_million_usd": 1},
        "ollama:generation:free": {
            "input_per_million_usd": 0, "output_per_million_usd": 0,
        },
    })
    operations = OperationsStore(tmp_path / "usage.sqlite3")
    runtime = Telemetry(settings, operations)
    with request_telemetry(runtime, {}), usage_tenant("tenant-a"):
        record_model_usage("ollama", "generation", "no-price", {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        })
        record_model_usage("ollama", "generation", "input-only", {
            "prompt_tokens": 10, "completion_tokens": 5,
        })
        record_model_usage("ollama", "generation", "free", {
            "prompt_tokens": 10, "completion_tokens": 5,
        })
        record_model_usage("atlas", "rerank", "native", None)
        record_model_usage("mongodb_ai", "embedding", "unknown", {
            "total_tokens": True,
        }, "error")
    groups = {item["model"]: item for item in operations.model_usage_summary()}
    assert groups["no-price"]["estimated_cost_usd"] is None
    assert groups["input-only"]["estimated_cost_usd"] is None
    assert groups["free"]["estimated_cost_usd"] == 0
    assert groups["native"]["total_tokens"] is None
    assert groups["unknown"]["total_tokens"] is None
    assert groups["unknown"]["failed_calls"] == 1
    assert operations.model_usage_summary("other-tenant") == []
    assert "rag_gateway_model_unpriced_calls_total" in usage_prometheus(list(groups.values()))


@pytest.mark.parametrize(("rates", "usage", "expected"), [
    ({"per_request_usd": 0.003}, None, 0.003),
    ({"per_request_usd": 0.003, "input_per_million_usd": 1}, None, None),
    (
        {"per_request_usd": 0.003, "input_per_million_usd": 1},
        {"prompt_tokens": 10}, 0.00301,
    ),
    (
        {"input_per_million_usd": 1, "output_per_million_usd": 2},
        {"prompt_tokens": 1000, "completion_tokens": 2000}, 0.005,
    ),
])
def test_cost_estimates_apply_configured_rates_without_guessing_usage(
    tmp_path: Path, rates: dict[str, float], usage: Any, expected: float | None,
) -> None:
    settings = Settings(_env_file=None, model_pricing={"ollama:generation:chat": rates})
    operations = OperationsStore(tmp_path / "usage.sqlite3")
    with request_telemetry(Telemetry(settings, operations), {}):
        record_model_usage("ollama", "generation", "chat", usage)
    cost = operations.model_usage_summary()[0]["estimated_cost_usd"]
    assert cost is None if expected is None else cost == pytest.approx(expected)


@pytest.mark.parametrize("failure", ["timeout", "error_event", "incomplete"])
async def test_stream_failure_is_recorded_once_and_spans_do_not_leak_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    settings = Settings(
        _env_file=None, gateway_api_key="admin", ollama_model="chat",
        operations_db_path=tmp_path / "usage.sqlite3",
    )
    operations = OperationsStore(settings.operations_db_path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime = Telemetry(settings, operations, provider)
    monkeypatch.setattr(main.app.state, "telemetry", runtime, raising=False)
    monkeypatch.setattr(main.app.state, "operations", operations, raising=False)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, main.get_store, lambda: object())
    monkeypatch.setitem(main.app.dependency_overrides, main.get_embedder, lambda: object())

    def upstream(_request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("private-error-url-and-prompt")
        event = (
            {"error": {"message": "private-error-url-and-prompt"}}
            if failure == "error_event" else {"choices": []}
        )
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://gateway.test",
        ) as client,
    ):
        monkeypatch.setattr(main.app.state, "http", upstream_client, raising=False)
        response = await client.post("/v1/chat/completions", headers={
            "Authorization": "Bearer admin", "X-Tenant-ID": "tenant-error",
        }, json={"messages": [{"role": "user", "content": ""}], "stream": True})
    assert response.status_code == 200
    if failure != "incomplete":
        assert '"error"' in response.text
    group = operations.model_usage_summary()[0]
    assert group["calls"] == 1
    assert group["failed_calls"] == 1
    assert group["tenant_id"] == "tenant-error"
    assert group["total_tokens"] is None
    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "POST /v1/chat/completions")
    assert root.status.status_code == trace.StatusCode.ERROR
    generation = next(span for span in spans if span.name == "ollama_stream")
    assert generation.status.status_code == trace.StatusCode.ERROR
    assert "private-error-url-and-prompt" not in repr([
        (span.attributes, span.events, span.status.description) for span in spans
    ])
    runtime.close()


async def test_cancelled_request_persists_usage_and_marks_trace_failed(tmp_path: Path) -> None:
    operations = OperationsStore(tmp_path / "usage.sqlite3")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime = Telemetry(Settings(_env_file=None), operations, provider)
    with CancelScope() as cancellation:
        with request_telemetry(runtime, {}), usage_tenant("cancelled-tenant"):
            try:
                cancellation.cancel()
                await anyio.lowlevel.checkpoint()
            finally:
                await persist_model_usage(
                    "ollama", "generation", "chat",
                    {"prompt_tokens": 10, "completion_tokens": 2}, "incomplete",
                )
    group = operations.model_usage_summary()[0]
    assert group["tenant_id"] == "cancelled-tenant"
    assert group["calls"] == group["failed_calls"] == 1
    assert group["total_tokens"] == 12
    assert exporter.get_finished_spans()[0].status.status_code == trace.StatusCode.ERROR
    runtime.close()


@pytest.mark.parametrize("pricing", [
    {"input_per_million_usd": -1},
    {"input_per_million_usd": float("inf")},
    {"per_request_usd": float("nan")},
    {"typo_rate": 1},
])
def test_invalid_prices_are_rejected(pricing: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_pricing={"provider:operation:model": pricing})


async def test_embedding_batches_retries_and_concurrent_tenants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None, mongodb_api_key="test-key", embedding_dimensions=1,
        embedding_batch_size=1,
    )
    operations = OperationsStore(tmp_path / "usage.sqlite3")
    runtime = Telemetry(settings, operations)
    calls = 0

    def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={
            "usage": {"total_tokens": 10},
            "data": [{"index": 0, "embedding": [0.1]}],
        })

    async def no_wait(_delay: float) -> None:
        pass

    monkeypatch.setattr("rag_gateway.voyage.asyncio.sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        embedder = VoyageEmbedder(settings, client)

        async def embed(tenant: str) -> None:
            with request_telemetry(runtime, {}), usage_tenant(tenant):
                result = await embedder.embed(["one", "two"], "document")
                assert result == [[0.1], [0.1]]

        await asyncio.gather(embed("a"), embed("b"))
    groups = {item["tenant_id"]: item for item in operations.model_usage_summary()}
    assert groups["a"]["calls"] == 3
    assert groups["a"]["failed_calls"] == 1
    assert groups["a"]["missing_usage_calls"] == 1
    assert groups["b"]["calls"] == 2
    assert groups["a"]["input_tokens"] == groups["b"]["input_tokens"] == 20
    assert all(item["estimated_cost_usd"] is None for item in groups.values())


def test_ocr_usage_uses_native_token_counts(tmp_path: Path) -> None:
    operations = OperationsStore(tmp_path / "usage.sqlite3")
    runtime = Telemetry(Settings(_env_file=None), operations)
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(
            200, json={
                "message": {"content": "extracted text"},
                "prompt_eval_count": 15, "eval_count": 5,
            },
        ))) as client,
        request_telemetry(runtime, {}),
        usage_tenant("ocr-tenant"),
    ):
        parser = OllamaDocumentParser("http://ollama.test", "ocr-model", client=client)
        assert parser.parse_image(b"image") == "extracted text"
    group = operations.model_usage_summary()[0]
    assert group["operation"] == "ocr"
    assert group["tenant_id"] == "ocr-tenant"
    assert group["total_tokens"] == 20
    assert group["input_tokens"] == 15
    assert group["output_tokens"] == 5
