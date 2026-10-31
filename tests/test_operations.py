from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_gateway.observability import ObservabilityMiddleware, RequestMetrics, request_metrics
from rag_gateway.operations import OperationsStore


def test_operations_store_persists_jobs_and_marks_active_jobs_interrupted(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operations.sqlite3"
    store = OperationsStore(database)
    job = {
        "job_id": "job-a",
        "tenant_id": "tenant-a",
        "status": "running",
        "events": [{"stage": "running", "message": "Started"}],
    }

    store.save_job(job)
    restarted_store = OperationsStore(database)

    restored = restarted_store.get_job("job-a")
    assert restored is not None
    assert restored["status"] == "interrupted"
    assert restored["events"][-1]["stage"] == "interrupted"
    assert restarted_store.get_job("job-a", ["tenant-b"]) is None
    assert restarted_store.list_jobs(["tenant-b"]) == []


def test_operations_store_clears_only_terminal_jobs_for_authorized_tenants(
    tmp_path: Path,
) -> None:
    store = OperationsStore(tmp_path / "operations.sqlite3")
    store.save_job(
        {"job_id": "done-a", "tenant_id": "tenant-a", "status": "completed", "events": []}
    )
    store.save_job(
        {"job_id": "running-a", "tenant_id": "tenant-a", "status": "running", "events": []}
    )
    store.save_job(
        {"job_id": "done-b", "tenant_id": "tenant-b", "status": "failed", "events": []}
    )

    assert store.clear_completed_jobs(["tenant-a"]) == 1
    assert [job["job_id"] for job in store.list_jobs()] == ["done-b", "running-a"]


def test_operations_store_persists_and_scopes_evaluation_runs(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite3"
    store = OperationsStore(database)
    evaluation = {
        "tenant_id": "tenant-a",
        "top_k": 5,
        "metrics": {
            "case_count": 1,
            "hit_rate_at_k": 1.0,
            "recall_at_k": 1.0,
            "mean_reciprocal_rank": 1.0,
            "mean_latency_ms": 12.5,
        },
        "cases": [
            {
                "query": "example",
                "expected_document_ids": ["doc-a"],
                "hit": True,
            }
        ],
    }

    run_id = store.save_evaluation(evaluation)
    restored = OperationsStore(database).get_evaluation(run_id)

    assert restored is not None
    assert restored["run_id"] == run_id
    assert restored["created_at"]
    assert restored["dataset_id"]
    assert restored["cases"] == evaluation["cases"]
    assert store.get_evaluation(run_id, ["tenant-b"]) is None
    assert store.list_evaluations(["tenant-b"]) == []
    summary = store.list_evaluations(["tenant-a"])[0]
    assert summary["case_count"] == 1
    assert "cases" not in summary


def test_observability_middleware_adds_request_id_and_exports_metrics() -> None:
    test_app = FastAPI()
    test_app.add_middleware(ObservabilityMiddleware)

    @test_app.get("/probe")
    async def probe() -> dict[str, str]:
        return {"status": "ok"}

    before = request_metrics.prometheus()
    with TestClient(test_app) as client:
        response = client.get(
            "/probe?private-search=must-not-be-recorded",
            headers={"X-Request-ID": "client.trace-1"},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "client.trace-1"
    assert 'route="/probe"' in request_metrics.prometheus()
    assert "rag_gateway_http_request_duration_seconds_count" in request_metrics.prometheus()
    assert request_metrics.prometheus() != before
    snapshot = request_metrics.snapshot()
    observed = next(
        item for item in snapshot["recent_requests"] if item["request_id"] == "client.trace-1"
    )
    assert observed["route"] == "/probe"
    assert observed["status"] == 200
    assert observed["duration_ms"] >= 0
    assert "must-not-be-recorded" not in repr(snapshot)


def test_request_metrics_exposes_in_flight_requests() -> None:
    metrics = RequestMetrics()
    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "route": SimpleNamespace(path="/v1/chat/completions"),
    }

    metrics.begin("chat-request", "POST", scope)
    active = metrics.snapshot()

    assert active["requests_in_flight"] == 1
    assert active["recent_requests"][0]["status"] == "in progress"
    assert active["recent_requests"][0]["route"] == "/v1/chat/completions"

    metrics.observe("POST", "/v1/chat/completions", 200, 0.025, request_id="chat-request")
    completed = metrics.snapshot()

    assert completed["requests_in_flight"] == 0
    assert completed["recent_requests"][0]["status"] == 200
    assert completed["recent_requests"][0]["duration_ms"] == 25
