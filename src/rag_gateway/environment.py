import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rag_gateway.config import Settings

SECRET_FIELDS = {"gateway_api_key", "mongodb_uri", "mongodb_api_key"}


def settings_response(settings: Settings) -> dict[str, Any]:
    values = settings.model_dump(mode="json")
    secrets = {
        field: bool(values.pop(field))
        for field in SECRET_FIELDS
    }
    return {"settings": values, "secrets_configured": secrets}


def _dotenv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(",", ":"))
    return json.dumps(str(value), ensure_ascii=False)


def save_settings(settings: Settings, env_path: Path | None = None) -> Path:
    path = (env_path or Path(".env")).absolute()
    if path.is_symlink():
        raise OSError("Refusing to overwrite a symbolic-link environment file")
    known_keys = {name.upper() for name in Settings.model_fields}
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    preserved_lines = []
    for line in existing_lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            preserved_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key not in known_keys:
            preserved_lines.append(line)

    values = settings.model_dump(mode="json")
    setting_lines = [f"{name.upper()}={_dotenv_value(value)}" for name, value in values.items()]
    content = "\n".join([*preserved_lines, *setting_lines, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(content)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path
