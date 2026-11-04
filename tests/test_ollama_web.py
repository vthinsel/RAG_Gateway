from __future__ import annotations

from hashlib import sha256
from typing import Any

import httpx
from fastapi.testclient import TestClient

from rag_gateway import main
from rag_gateway.config import Settings, get_settings


def test_ollama_management_page_and_admin_api(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        gateway_api_key="admin-key",
        api_key_registry={sha256(b"tenant-key").hexdigest(): ["tenant-a"]},
        ollama_base_url="http://ollama.test:11434/v1",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    upstream_requests: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        upstream_requests.append((method, url, kwargs.get("json", {})))
        path = url.removeprefix("http://ollama.test:11434")
        if path == "/api/tags":
            result = {"models": [{"name": "qwen:latest", "size": 1}]}
        elif path == "/api/ps":
            result = {"models": [{"name": "qwen:latest", "context_length": 4096}]}
        elif path == "/api/version":
            result = {"version": "0.1.0"}
        elif path == "/api/show":
            result = {
                "modelfile": "FROM qwen:latest",
                "details": {
                    "family": "qwen",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_K_M",
                },
                "thinking": {"values": ["low", "medium", "high"], "default": "medium"},
            }
        elif path == "/api/generate":
            result = {"done": True}
        else:
            result = {"status": "success"}
        return httpx.Response(200, json=result, request=httpx.Request(method, url))

    try:
        with TestClient(main.app) as client:
            monkeypatch.setattr(main.app.state.http, "request", fake_request)

            page = client.get("/tools/ollama")
            assert page.status_code == 200
            assert "Ollama management" in page.text
            assert "/v1/admin/ollama/context" in page.text
            assert "Reasoning options" in page.text
            assert "renderModelDetails" in page.text
            assert client.get("/").status_code == 200
            assert 'href="/tools/ollama"' in client.get("/").text

            assert client.get("/v1/admin/ollama/models").status_code == 401
            tenant_headers = {"Authorization": "Bearer tenant-key"}
            assert client.get("/v1/admin/ollama/models", headers=tenant_headers).status_code == 401

            headers = {"Authorization": "Bearer admin-key"}
            assert client.get("/v1/admin/ollama/models", headers=headers).json() == {
                "models": [{"name": "qwen:latest", "size": 1}]
            }
            assert client.get("/v1/admin/ollama/running", headers=headers).json()["models"]
            assert client.get("/v1/admin/ollama/version", headers=headers).json() == {
                "version": "0.1.0"
            }
            assert (
                client.post(
                    "/v1/admin/ollama/pull", headers=headers, json={"model": "qwen:latest"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/admin/ollama/delete", headers=headers, json={"model": "qwen:old"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/admin/ollama/copy",
                    headers=headers,
                    json={"source": "qwen:latest", "destination": "qwen:copy"},
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/admin/ollama/context",
                    headers=headers,
                    json={"model": "qwen:latest", "context_size": 8192},
                ).status_code
                == 200
            )
            show_response = client.post(
                "/v1/admin/ollama/show", headers=headers, json={"model": "qwen:latest"}
            )
            assert show_response.status_code == 200
            assert show_response.json()["thinking"] == {
                "values": ["low", "medium", "high"],
                "default": "medium",
            }
            assert show_response.json()["details"]["family"] == "qwen"
            assert (
                client.post(
                    "/v1/admin/ollama/stop", headers=headers, json={"model": "qwen:latest"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/admin/ollama/context",
                    headers=headers,
                    json={"model": "qwen:latest", "context_size": 0},
                ).status_code
                == 422
            )
            assert (
                client.post(
                    "/v1/admin/ollama/context",
                    headers=headers,
                    json={"model": "qwen:latest\nSYSTEM malicious", "context_size": 8192},
                ).status_code
                == 422
            )

        assert all("/v1/api/" not in url for _, url, _ in upstream_requests)
        assert (
            "POST",
            "http://ollama.test:11434/api/pull",
            {
                "name": "qwen:latest",
                "stream": False,
            },
        ) in upstream_requests
        assert (
            "POST",
            "http://ollama.test:11434/api/create",
            {
                "name": "qwen:latest-ctx8192",
                "modelfile": "FROM qwen:latest\nPARAMETER num_ctx 8192\n",
                "stream": False,
            },
        ) in upstream_requests
    finally:
        main.app.dependency_overrides.pop(get_settings, None)


def test_ollama_upstream_errors_are_reported(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        gateway_api_key="admin-key",
        ollama_base_url="http://ollama.test:11434/v1",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)

    async def failed_request(method: str, url: str, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            404,
            text="model not found",
            request=httpx.Request(method, url),
        )

    try:
        with TestClient(main.app) as client:
            monkeypatch.setattr(main.app.state.http, "request", failed_request)
            response = client.get(
                "/v1/admin/ollama/models",
                headers={"Authorization": "Bearer admin-key"},
            )
        assert response.status_code == 502
        assert "model not found" in response.json()["detail"]
    finally:
        main.app.dependency_overrides.pop(get_settings, None)
