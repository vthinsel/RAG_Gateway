from pathlib import Path

from rag_gateway import runner
from rag_gateway.config import Settings


def test_native_runner_uses_gateway_bind_settings(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        gateway_bind_address="192.0.2.164",
        gateway_port=8123,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(runner.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    runner.run()

    assert captured == {
        "host": "192.0.2.164",
        "port": 8123,
    }


def test_gateway_bind_settings_load_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GATEWAY_BIND_ADDRESS=192.0.2.164\nGATEWAY_PORT=8123\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.gateway_bind_address == "192.0.2.164"
    assert settings.gateway_port == 8123
