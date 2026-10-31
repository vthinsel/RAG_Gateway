from hashlib import sha256

from fastapi.testclient import TestClient

from rag_gateway import main
from rag_gateway.config import Settings, get_settings


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
