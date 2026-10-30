from hashlib import sha256
from pathlib import Path

from dotenv import dotenv_values
from fastapi.testclient import TestClient

from rag_gateway import main
from rag_gateway.config import Settings, get_settings
from rag_gateway.environment import save_settings, settings_response


def test_settings_response_redacts_secrets() -> None:
    settings = Settings(
        _env_file=None,
        gateway_api_key="gateway-secret",
        mongodb_uri="mongodb+srv://user:password@cluster.example",
        mongodb_api_key="mongodb-secret",
    )

    response = settings_response(settings)

    assert response["settings"]["mongodb_database"] == "rag_gateway"
    assert response["secrets_configured"] == {
        "gateway_api_key": True,
        "mongodb_uri": True,
        "mongodb_api_key": True,
    }
    assert set(response["settings"]) | set(response["secrets_configured"]) == set(Settings.model_fields)
    assert "gateway-secret" not in repr(response)
    assert "mongodb-secret" not in repr(response)
    assert "password" not in repr(response)


def test_save_settings_persists_values_and_preserves_unmanaged_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Keep this comment\nGATEWAY_BIND_ADDRESS=127.0.0.1\nMONGODB_DATABASE=old\n",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        gateway_api_key="a-new-secret",
        api_key_registry={"hash": ["tenant-a"]},
        mongodb_uri="mongodb+srv://user:password@cluster.example",
        document_loader_roots=[tmp_path],
        retrieval_min_score=0.63,
    )

    saved_path = save_settings(settings, env_file)

    assert saved_path == env_file.absolute()
    values = dotenv_values(env_file)
    assert values["GATEWAY_BIND_ADDRESS"] == "127.0.0.1"
    assert values["MONGODB_DATABASE"] == "rag_gateway"
    assert values["GATEWAY_API_KEY"] == "a-new-secret"
    assert values["API_KEY_REGISTRY"] == '{"hash":["tenant-a"]}'
    assert values["MONGODB_URI"] == "mongodb+srv://user:password@cluster.example"
    assert values["DOCUMENT_LOADER_ROOTS"] == f'["{tmp_path}"]'
    assert values["RETRIEVAL_MIN_SCORE"] == "0.63"
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_save_settings_refuses_symlinked_env_file(tmp_path: Path) -> None:
    target = tmp_path / "outside.env"
    target.write_text("KEEP=value\n", encoding="utf-8")
    link = tmp_path / ".env"
    link.symlink_to(target)

    try:
        save_settings(Settings(_env_file=None), link)
    except OSError as error:
        assert "symbolic-link" in str(error)
    else:
        raise AssertionError("Expected symlinked .env to be rejected")
    assert target.read_text(encoding="utf-8") == "KEEP=value\n"


def test_dotenv_overrides_environment_when_precedence_is_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MONGODB_DATABASE=persisted_value\n", encoding="utf-8")
    monkeypatch.setenv("MONGODB_DATABASE", "compose_value")
    monkeypatch.setenv("RAG_GATEWAY_DOTENV_PRECEDENCE", "true")

    settings = Settings(_env_file=env_file)

    assert settings.mongodb_database == "persisted_value"


def test_environment_overrides_dotenv_by_default(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MONGODB_DATABASE=dotenv_value\n", encoding="utf-8")
    monkeypatch.setenv("MONGODB_DATABASE", "environment_value")
    monkeypatch.delenv("RAG_GATEWAY_DOTENV_PRECEDENCE", raising=False)

    settings = Settings(_env_file=env_file)

    assert settings.mongodb_database == "environment_value"


def test_environment_api_requires_admin_and_persists_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        gateway_api_key="old-admin-key",
        api_key_registry={sha256(b"tenant-key").hexdigest(): ["tenant-a"]},
    )

    class FakeParser:
        def __init__(self, *_args) -> None:
            pass

        def list_models(self) -> list[str]:
            return []

        def close(self) -> None:
            pass

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "OllamaDocumentParser", FakeParser)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    try:
        with TestClient(main.app) as client:
            assert client.get("/tools/environment").status_code == 200
            assert client.get("/v1/admin/environment").status_code == 401
            read_response = client.get(
                "/v1/admin/environment",
                headers={"Authorization": "Bearer old-admin-key"},
            )
            tenant_key_response = client.get(
                "/v1/admin/environment",
                headers={"Authorization": "Bearer tenant-key"},
            )
            assert read_response.status_code == 200
            assert tenant_key_response.status_code == 401
            assert "old-admin-key" not in read_response.text

            save_response = client.put(
                "/v1/admin/environment",
                headers={"Authorization": "Bearer old-admin-key"},
                json={
                    "settings": {
                        "mongodb_database": "lab_rag",
                        "retrieval_min_score": 0.61,
                    },
                    "secrets": {"gateway_api_key": "new-admin-key"},
                },
            )

        assert save_response.status_code == 200
        assert save_response.json()["restart_required"] is True
        values = dotenv_values(tmp_path / ".env")
        assert values["MONGODB_DATABASE"] == "lab_rag"
        assert values["RETRIEVAL_MIN_SCORE"] == "0.61"
        assert values["GATEWAY_API_KEY"] == "new-admin-key"
        restarted_settings = Settings(_env_file=tmp_path / ".env")
        assert restarted_settings.mongodb_database == "lab_rag"
        assert restarted_settings.gateway_api_key == "new-admin-key"
    finally:
        main.app.dependency_overrides.pop(get_settings, None)


def test_gateway_key_generation_requires_admin_and_does_not_persist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, gateway_api_key="admin-key")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    try:
        with TestClient(main.app) as client:
            denied_response = client.post("/v1/admin/environment/gateway-api-key")
            generated_response = client.post(
                "/v1/admin/environment/gateway-api-key",
                headers={"Authorization": "Bearer admin-key"},
            )

        assert denied_response.status_code == 401
        assert generated_response.status_code == 200
        generated_key = generated_response.json()["gateway_api_key"]
        assert len(generated_key) >= 64
        assert not (tmp_path / ".env").exists()
    finally:
        main.app.dependency_overrides.pop(get_settings, None)
