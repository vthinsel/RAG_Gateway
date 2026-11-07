import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from run_benchmark import (
    build_parser,
    describe_conditions,
    execute_profiles,
    load_profiles,
    load_questions,
    make_payload,
    output_path,
    perform_request,
    redact,
    validate_options,
)


def test_blind_input_and_runtime_options() -> None:
    metadata, questions = load_questions(
        Path(__file__).parent / "homelab_benchmark_v16_questions_only.json"
    )
    assert len(questions) == metadata["question_count"] == 33
    options = validate_options({"temperature": 0, "retrieval_min_score": 0})
    payload = make_payload(questions[0]["question"], options, "chat")
    assert payload["messages"] == [{"role": "user", "content": questions[0]["question"]}]
    assert payload["reasoning_effort"] == "none"
    assert payload["hybrid_enabled"] is False
    assert payload["rerank_enabled"] is False
    assert payload["temperature"] == payload["retrieval_min_score"] == 0
    assert "expected_answer" not in json.dumps(payload)
    with pytest.raises(ValueError, match="blind"):
        load_questions(Path(__file__).parent / "homelab_benchmark_v16.json")


@pytest.mark.parametrize("options", [
    {"ollama_model": "other"}, {"retrieval_rerank_candidates": 20},
    {"temperature": True}, {"temperature": float("nan")}, {"retrieval_min_score": 1.1},
    {"retrieval_num_candidates": 1.5}, {"reasoning_effort": "high\n"},
    {"hybrid_enabled": "false"},
])
def test_invalid_options_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        validate_options(options)


def test_capture_preserves_all_gateway_fields() -> None:
    raw = {
        "choices": [{"message": {"content": "Answer [S1] [S9]", "reasoning_content": "Trace"},
                     "finish_reason": "stop"}],
        "sources": [{"citation": "S1", "document_id": "real-atlas-id", "chunk_index": 2,
                     "score": 0.03, "vector_score": 0.8, "lexical_score": 4,
                     "rerank_score": 0.9, "metadata": {"file": "guide.md"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "extra": "retained"},
        "model": "public-alias", "unknown_vendor_field": {"keep": True},
    }

    async def exercise() -> dict:
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json=raw)
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await perform_request(client, "/v1/chat/completions", {}, "chat")

    result = asyncio.run(exercise())
    assert result["status"] == "ok"
    assert result["raw_response"] == raw
    assert result["sources"] == raw["sources"]
    assert result["tokens"] == {"prompt": 10, "completion": 5, "total": 15}
    assert result["invalid_citation_labels"] == ["S9"]
    assert result["timings_ms"]["client_total"] >= 0
    assert result["timings_ms"]["ttft"] is None


def test_conditions_expose_hybrid_noop_and_candidate_floor() -> None:
    options = validate_options({"hybrid_enabled": True, "retrieval_num_candidates": 10})
    conditions = describe_conditions(options, {"retrieval_hybrid_candidates": 100}, "chat")
    assert conditions["score_cutoff_applied"] is False
    assert conditions["num_candidates_from_config"] == 100
    assert conditions["score_kind"] == "reciprocal_rank_fusion"
    assert conditions["configured_actual_model"] is None
    assert "seed" in conditions["unforced_generation_fields"]
    assert describe_conditions(options, {}, "chat")["num_candidates_from_config"] is None


def test_output_cannot_escape_benchmark_and_secrets_are_redacted() -> None:
    with pytest.raises(ValueError, match="benchmark"):
        output_path("../outside", "run.json")
    value = {"raw_response": ["Bearer private-test-token"], "answer": "private-test-token"}
    assert "private-test-token" not in json.dumps(redact(value, "private-test-token"))


def test_profile_overrides_are_explicit() -> None:
    args = build_parser().parse_args([
        "--tenant", "test", "--hybrid", "--no-rerank", "--num-candidates", "200",
        "--reasoning", "high", "--temperature", "0", "--max-tokens", "400",
    ])
    profile = load_profiles(args)[0]
    assert profile["options"]["hybrid_enabled"] is True
    assert profile["options"]["rerank_enabled"] is False
    assert profile["options"]["max_tokens"] == 400
    payload = make_payload("question", profile["options"], "retrieve")
    assert "reasoning_effort" not in payload
    assert "temperature" not in payload


def test_full_run_checkpoints_warmup_errors_probes_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    args = build_parser().parse_args([
        "--tenant", "tenant-a", "--base-url", "http://test", "--repetitions", "2",
        "--warmup", "1", "--capture-retrieval", "--shuffle", "--order-seed", "12",
    ])
    metadata, questions = load_questions(
        Path(__file__).parent / "homelab_benchmark_v16_questions_only.json"
    )
    profiles = [
        {"name": "vector", "options": validate_options({})},
        {"name": "hybrid", "options": validate_options({"hybrid_enabled": True})},
    ]
    saved = []
    sent = []

    def save(path: Path, result: dict, secret: str = "") -> None:
        assert path.is_relative_to(Path(__file__).parent)
        saved.append(copy.deepcopy(redact(result, secret)))

    monkeypatch.setattr("run_benchmark.save_result", save)

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        assert request.headers["x-tenant-id"] == "tenant-a"
        assert "expected_answer" not in request.content.decode()
        source = {"citation": "S1", "document_id": "atlas-doc", "chunk_index": 1, "score": 0.8}
        if request.url.path == "/v1/retrieve":
            return httpx.Response(200, json={"data": [{**source, "text": "evidence"}]})
        if payload["messages"][0]["content"] == questions[1]["question"]:
            return httpx.Response(502, json={"detail": "upstream error"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Answer [S1]"}, "finish_reason": "stop"}],
            "sources": [source], "usage": {"total_tokens": 12},
            "vendor_echo": "private-test-token",
        })

    async def exercise() -> list:
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(respond),
            headers={"X-Tenant-ID": "tenant-a"},
        ) as client:
            return await execute_profiles(
                args, profiles, metadata, questions[:2], client, "private-test-token"
            )

    results = asyncio.run(exercise())
    assert len(results) == 2
    for result in results:
        assert result["state"] == "completed_with_errors"
        assert result["summary"]["completed"] == 4
        assert result["summary"]["warmup_completed"] == 1
        assert result["summary"]["ok"] == result["summary"]["errors"] == 2
        assert result["summary"]["reported_total_tokens"] == 24
        assert result["records"][0]["retrieval_probe"]["is_exact_chat_context"] is False
        assert result["records"][0]["retrieval_probe"]["same_document_chunk_order_as_chat"] is True
        assert result["finished_at"]
    assert len(saved) > 10
    assert "private-test-token" not in json.dumps(saved)
    assert {body["hybrid_enabled"] for body in sent} == {True, False}


@pytest.mark.parametrize("kind", ["timeout", "invalid_json", "missing_choices"])
def test_transport_and_malformed_response_failures_are_recorded(kind: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if kind == "timeout":
            raise httpx.ReadTimeout("test timeout", request=request)
        if kind == "invalid_json":
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json={"choices": []})

    async def exercise() -> dict:
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await perform_request(client, "/v1/chat/completions", {}, "chat")

    result = asyncio.run(exercise())
    assert result["status"] == "error"
    assert result["timings_ms"]["client_total"] >= 0
    assert "error" in result


def test_cli_over_real_http_writes_valid_json_and_live_progress() -> None:
    token = "mock-cli-token-not-a-secret"
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.headers["Authorization"] == f"Bearer {token}"
            assert self.headers["X-Tenant-ID"] == "test-tenant"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            finished = threading.Event()
            finished.wait(0.03)
            reply = json.dumps({
                "choices": [{"message": {"content": "Mock answer [S1]"}, "finish_reason": "stop"}],
                "sources": [{"citation": "S1", "document_id": "mock-atlas-id", "chunk_index": 0,
                             "score": 0.8, "metadata": {}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                "extra": token,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_args: object) -> None:
            pass

    folder = Path(__file__).parent / "results" / f"mock_cli_{uuid.uuid4().hex[:8]}"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = subprocess.run([
            sys.executable, str(Path(__file__).parent / "run_benchmark.py"),
            "--base-url", f"http://127.0.0.1:{server.server_port}", "--tenant", "test-tenant",
            "--ids", "HL-01,HL-02", "--temperature", "0", "--reasoning", "none",
            "--progress-interval", "0.01", "--output-dir", str(folder),
        ], env={**os.environ, "GATEWAY_API_KEY": token}, capture_output=True,
            text=True, timeout=20, check=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert completed.returncode == 0, completed.stderr
    assert "waiting single" in completed.stdout
    assert "sources=1 | tokens=12" in completed.stdout
    assert token not in completed.stdout + completed.stderr
    assert len(bodies) == 2
    files = list(folder.glob("*.json"))
    assert len(files) == 1
    raw = files[0].read_text(encoding="utf-8")
    assert token not in raw
    result = json.loads(raw)
    assert result["state"] == "completed"
    assert result["summary"]["reported_total_tokens"] == 24
    assert result["records"][0]["raw_response"]["extra"] == "[REDACTED]"
    assert result["conditions"]["requested_options"]["temperature"] == 0
    assert result["conditions"]["requested_options"]["reasoning_effort"] == "none"
    assert all("expected_answer" not in json.dumps(body) for body in bodies)