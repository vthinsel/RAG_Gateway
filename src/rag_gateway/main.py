import asyncio
import json
import logging
import os
import secrets
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo.errors import PyMongoError
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse, StreamingResponse

from rag_gateway.chunking import chunk_text
from rag_gateway.config import Settings, get_settings
from rag_gateway.document_loader import (
    MAX_SOURCE_FILE_BYTES,
    LoaderOptions,
    OllamaDocumentParser,
    load_directory,
    normalize_extensions,
)
from rag_gateway.document_loader import (
    IngestDocument as ParsedDocument,
)
from rag_gateway.environment import SECRET_FIELDS, save_settings, settings_response
from rag_gateway.models import DocumentLoaderRequest, IngestDocument, IngestResult
from rag_gateway.rag import add_rag_system_message, extract_user_query, retrieve_context
from rag_gateway.store import AtlasStore
from rag_gateway.voyage import VoyageEmbedder

bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
MAX_UPLOAD_FILES = 500
MAX_UPLOAD_TOTAL_BYTES = 200 * 1024 * 1024


def _host_label(url: str) -> str:
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    return parsed.hostname or "unknown host"


def _mongodb_host_label(uri: str) -> str:
    return urlsplit(uri).hostname or "configured MongoDB URI"


async def _check_mongodb_ai_model_api(settings: Settings, client: httpx.AsyncClient) -> None:
    api_base = settings.mongodb_api_model_base_url
    model = settings.voyage_embedding_model
    host = _host_label(settings.mongodb_api_model_endpoint)
    if not settings.mongodb_api_key:
        logger.info(
            "MongoDB AI API model check skipped: endpoint=%s model=%s; "
            "MONGODB_API_KEY is not configured",
            host,
            model,
        )
        return

    logger.info("Checking MongoDB AI API model: endpoint=%s model=%s", host, model)
    try:
        response = await client.post(
            f"{api_base}/embeddings",
            headers={"Authorization": "Bearer " + settings.mongodb_api_key},
            json={"model": model, "input": ["MongoDB AI API startup connectivity check"]},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError("unexpected embeddings response shape")
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("response did not contain a non-empty embedding")
    except httpx.HTTPStatusError as error:
        logger.warning(
            "MongoDB AI API model check failed: endpoint=%s model=%s status=%s",
            host,
            model,
            error.response.status_code,
        )
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
        logger.warning(
            "MongoDB AI API model check failed: endpoint=%s model=%s error=%s",
            host,
            model,
            type(error).__name__,
        )
    else:
        actual_dimensions = len(embedding)
        if actual_dimensions != settings.embedding_dimensions:
            logger.warning(
                "MongoDB AI API model responded but returned %s dimensions; "
                "EMBEDDING_DIMENSIONS is configured as %s",
                actual_dimensions,
                settings.embedding_dimensions,
            )
        logger.info(
            "MongoDB AI API model check succeeded: model=%s dimensions=%s",
            model,
            actual_dimensions,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "Gateway authentication configuration: cwd=%s dotenv=%s dotenv_exists=%s "
        "gateway_key_configured=%s process_env_override=%s registry_entries=%s",
        Path.cwd(),
        Path.cwd() / ".env",
        (Path.cwd() / ".env").is_file(),
        bool(settings.gateway_api_key),
        "GATEWAY_API_KEY" in os.environ,
        len(settings.api_key_registry),
    )
    app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout_seconds)
    app.state.store = None
    app.state.document_loader_jobs = {}
    app.state.document_loader_tasks = set()
    atlas_error: PyMongoError | None = None
    if settings.mongodb_uri:
        logger.info(
            "Starting MongoDB Atlas connection check: host=%s database=%s collection=%s",
            _mongodb_host_label(settings.mongodb_uri),
            settings.mongodb_database,
            settings.mongodb_collection,
        )
        try:
            app.state.store = await run_in_threadpool(AtlasStore, settings)
            logger.info("MongoDB Atlas client initialized; sending ping")
            await run_in_threadpool(app.state.store.ping)
            logger.info(
                "MongoDB Atlas connection check succeeded: host=%s database=%s collection=%s",
                _mongodb_host_label(settings.mongodb_uri),
                settings.mongodb_database,
                settings.mongodb_collection,
            )
            try:
                index_exists, index_queryable = await run_in_threadpool(
                    app.state.store.vector_index_status
                )
                if index_exists and index_queryable:
                    logger.info(
                        "Atlas Vector Search index is ready: name=%s",
                        settings.atlas_vector_index,
                    )
                elif index_exists:
                    logger.warning(
                        "Atlas Vector Search index exists but is not queryable yet: name=%s",
                        settings.atlas_vector_index,
                    )
                else:
                    logger.warning(
                        "Atlas Vector Search index was not found: name=%s",
                        settings.atlas_vector_index,
                    )
            except PyMongoError as error:
                logger.warning(
                    "Atlas Vector Search index check failed: name=%s error=%s",
                    settings.atlas_vector_index,
                    type(error).__name__,
                )
            try:
                tenant_ids = await run_in_threadpool(app.state.store.list_tenant_ids)
                if tenant_ids:
                    logger.info(
                        "Atlas collection tenants: %s",
                        ", ".join(tenant_ids),
                    )
                else:
                    logger.warning("Atlas collection has no tenant_id values")
            except PyMongoError as error:
                logger.warning(
                    "Atlas tenant listing failed: collection=%s error=%s",
                    settings.mongodb_collection,
                    type(error).__name__,
                )
        except PyMongoError as error:
            logger.error("Atlas store initialization failed (%s): %s", type(error).__name__, error)
            logger.debug("Atlas store initialization details", exc_info=error)
            if app.state.store is not None:
                await run_in_threadpool(app.state.store.close)
                app.state.store = None
            atlas_error = error
    else:
        logger.warning("MongoDB Atlas connection check skipped: MONGODB_URI is not configured")

    await _check_mongodb_ai_model_api(settings, app.state.http)

    parser = OllamaDocumentParser(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.document_parser_timeout_seconds,
    )
    logger.info(
        "Checking Ollama model service: endpoint=%s chat_model=%s document_loader=%s",
        _host_label(settings.ollama_base_url),
        settings.ollama_model,
        "enabled" if settings.document_loader_enabled else "disabled",
    )
    try:
        models = await asyncio.to_thread(parser.list_models)
        logger.info(
            "Ollama connection check succeeded; discovered models: %s",
            ", ".join(models) if models else "(none)",
        )
        if not models:
            logger.warning("Ollama responded successfully but has no models installed")
        if not any(model.casefold() == settings.ollama_model.casefold() for model in models):
            logger.warning("Configured Ollama chat model %r is not installed", settings.ollama_model)
        else:
            logger.info("Configured Ollama chat model %r is installed", settings.ollama_model)
        if settings.document_loader_enabled and not any(
            model.casefold() == settings.document_parser_model.casefold() for model in models
        ):
            logger.warning(
                "Configured document parser model %r is not installed in Ollama",
                settings.document_parser_model,
            )
        elif settings.document_loader_enabled:
            logger.info("Configured Ollama document parser model %r is installed", settings.document_parser_model)
    except (httpx.HTTPError, ValueError) as error:
        logger.warning(
            "Ollama model discovery failed: endpoint=%s error=%s",
            _host_label(settings.ollama_base_url),
            type(error).__name__,
        )
    finally:
        parser.close()
    if atlas_error is not None:
        await app.state.http.aclose()
        raise atlas_error
    yield
    if app.state.store is not None:
        await run_in_threadpool(app.state.store.close)
    await app.state.http.aclose()


app = FastAPI(title="Atlas RAG Gateway", version="0.1.0", lifespan=lifespan)


def _require_document_loader(settings: Settings) -> None:
    if not settings.document_loader_enabled:
        raise HTTPException(status_code=404, detail="Document loader is disabled")


def _resolve_loader_directory(directory: Path, roots: list[Path]) -> Path:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail="Directory does not exist or is not accessible")
    allowed_roots = [root.expanduser().resolve() for root in roots if root.expanduser().exists()]
    if not allowed_roots:
        raise HTTPException(status_code=503, detail="Configure DOCUMENT_LOADER_ROOTS before enabling the loader")
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Directory is outside DOCUMENT_LOADER_ROOTS")
    return resolved


def _job_for_tenant(
    job_id: str,
    jobs: dict[str, dict[str, Any]],
    allowed_tenants: list[str] | None,
) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Loader job not found")
    if allowed_tenants is not None and job["tenant_id"] not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    return job


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> list[str] | None:
    if not settings.gateway_api_key and not settings.api_key_registry:
        raise HTTPException(status_code=503, detail="Gateway API key is not configured")
    if credentials is None:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
    if settings.api_key_registry:
        credential_hash = sha256(credentials.credentials.encode()).hexdigest()
        for configured_hash, tenant_ids in settings.api_key_registry.items():
            if compare_digest(credential_hash, configured_hash):
                return tenant_ids
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
    if credentials is None or not compare_digest(
        sha256(credentials.credentials.encode()).digest(),
        sha256(settings.gateway_api_key.encode()).digest(),
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
    return None


def require_tool_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> list[str] | None:
    if (
        credentials is not None
        and settings.gateway_api_key
        and compare_digest(
            sha256(credentials.credentials.encode()).digest(),
            sha256(settings.gateway_api_key.encode()).digest(),
        )
    ):
        return None
    return require_api_key(credentials, settings)


def require_environment_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.gateway_api_key:
        raise HTTPException(
            status_code=503,
            detail="GATEWAY_API_KEY must be configured to administer environment settings",
        )
    if credentials is None or not compare_digest(
        sha256(credentials.credentials.encode()).digest(),
        sha256(settings.gateway_api_key.encode()).digest(),
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing admin bearer token")


def get_store(request: Request) -> AtlasStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MongoDB Atlas is not configured")
    return store


def get_embedder(request: Request, settings: Settings = Depends(get_settings)) -> VoyageEmbedder:
    return VoyageEmbedder(settings, request.app.state.http)


def get_tenant_id(
    x_tenant_id: str | None,
    settings: Settings,
    allowed_tenants: list[str] | None = None,
) -> str:
    tenant_id = x_tenant_id or settings.default_tenant_id
    if not tenant_id or len(tenant_id) > 256:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")
    if allowed_tenants is not None and tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    return tenant_id


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def welcome_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "welcome.html")


@app.get("/v1/status", include_in_schema=False)
async def gateway_status(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "ready",
        "application": {"status": "ok", "version": app.version},
        "atlas": {"status": "not_configured"},
        "vector_index": {"status": "unknown", "name": settings.atlas_vector_index},
        "ollama": {
            "status": "unknown",
            "endpoint": _host_label(settings.ollama_base_url),
            "model": settings.ollama_model,
        },
        "mongodb_ai_api": {
            "status": "configured" if settings.mongodb_api_key else "not_configured",
            "endpoint": _host_label(settings.mongodb_api_model_endpoint),
            "model": settings.voyage_embedding_model,
        },
        "authentication": {
            "status": "configured"
            if settings.gateway_api_key or settings.api_key_registry
            else "not_configured"
        },
    }

    store = getattr(request.app.state, "store", None)
    if store is not None:
        try:
            await run_in_threadpool(store.ping)
            status["atlas"] = {"status": "connected"}
            index_exists, index_queryable = await run_in_threadpool(store.vector_index_status)
            status["vector_index"]["status"] = (
                "ready" if index_exists and index_queryable else
                "building" if index_exists else "missing"
            )
        except PyMongoError as error:
            status["atlas"] = {"status": "error", "error": type(error).__name__}
            status["vector_index"]["status"] = "unavailable"
            logger.warning("Gateway status Atlas check failed (%s)", type(error).__name__)
    else:
        status["atlas"] = {"status": "not_configured"}
        status["vector_index"]["status"] = "unavailable"

    parser = OllamaDocumentParser(
        settings.ollama_base_url,
        settings.ollama_model,
        min(settings.document_parser_timeout_seconds, settings.upstream_timeout_seconds),
    )
    try:
        models = await asyncio.to_thread(parser.list_models)
        matching_model = any(
            model.casefold() == settings.ollama_model.casefold() for model in models
        )
        status["ollama"]["status"] = "connected" if matching_model else "model_missing"
        status["ollama"]["models_discovered"] = len(models)
    except (httpx.HTTPError, ValueError) as error:
        status["ollama"].update({"status": "error", "error": type(error).__name__})
        logger.warning("Gateway status Ollama check failed (%s)", type(error).__name__)
    finally:
        parser.close()

    components = (
        status["atlas"]["status"] == "connected",
        status["vector_index"]["status"] == "ready",
        status["ollama"]["status"] == "connected",
        status["mongodb_ai_api"]["status"] == "configured",
        bool(settings.gateway_api_key or settings.api_key_registry),
    )
    if not all(components):
        status["status"] = (
            "not_ready"
            if status["atlas"]["status"] in {"not_configured", "error"}
            or status["vector_index"]["status"] in {"missing", "unavailable"}
            else "degraded"
        )
    return status


@app.get("/readyz")
async def readiness(request: Request) -> JSONResponse:
    store = getattr(request.app.state, "store", None)
    settings = get_settings()
    missing_configuration = []
    if store is None:
        missing_configuration.append("MONGODB_URI")
    if not (settings.gateway_api_key or settings.api_key_registry):
        missing_configuration.append("GATEWAY_API_KEY or API_KEY_REGISTRY")
    if not settings.mongodb_api_key:
        missing_configuration.append("MONGODB_API_KEY")
    if missing_configuration:
        logger.warning("Readiness check missing configuration: %s", ", ".join(missing_configuration))
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    try:
        await run_in_threadpool(store.ping)
    except PyMongoError as error:
        logger.warning("Atlas readiness ping failed (%s): %s", type(error).__name__, error)
        logger.debug("Atlas readiness ping details", exc_info=error)
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(content={"status": "ready"})


@app.get("/tools/document-loader", include_in_schema=False)
async def document_loader_ui(settings: Settings = Depends(get_settings)) -> FileResponse:
    _require_document_loader(settings)
    return FileResponse(Path(__file__).parent / "static" / "document_loader.html")


@app.get("/tools/environment", include_in_schema=False)
async def environment_ui() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "environment.html")


@app.get("/tools/chunks", include_in_schema=False)
async def chunks_ui() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "chunks.html")


@app.get("/v1/admin/environment")
async def read_environment(
    _admin: None = Depends(require_environment_admin),
) -> dict[str, Any]:
    try:
        return settings_response(Settings())
    except (OSError, ValueError) as error:
        logger.exception("Could not read environment configuration")
        raise HTTPException(status_code=500, detail="Could not read environment configuration") from error


@app.post("/v1/admin/environment/gateway-api-key")
async def generate_gateway_api_key(
    _admin: None = Depends(require_environment_admin),
) -> dict[str, str]:
    return {"gateway_api_key": secrets.token_urlsafe(48)}


@app.put("/v1/admin/environment")
async def update_environment(
    body: dict[str, Any],
    _admin: None = Depends(require_environment_admin),
) -> dict[str, Any]:
    settings_updates = body.get("settings", {})
    secrets = body.get("secrets", {})
    clear_secrets = body.get("clear_secrets", [])
    if not isinstance(settings_updates, dict) or not isinstance(secrets, dict):
        raise HTTPException(status_code=422, detail="'settings' and 'secrets' must be objects")
    if not isinstance(clear_secrets, list) or not all(
        isinstance(field, str) for field in clear_secrets
    ):
        raise HTTPException(status_code=422, detail="'clear_secrets' must be an array of names")

    allowed_settings = set(Settings.model_fields) - SECRET_FIELDS
    unknown_settings = set(settings_updates) - allowed_settings
    unknown_secrets = set(secrets) - SECRET_FIELDS
    invalid_clear_secrets = set(clear_secrets) - SECRET_FIELDS
    if unknown_settings or unknown_secrets or invalid_clear_secrets:
        raise HTTPException(
            status_code=422,
            detail={
                "unknown_settings": sorted(unknown_settings),
                "unknown_secrets": sorted(unknown_secrets),
                "invalid_clear_secrets": sorted(invalid_clear_secrets),
            },
        )
    if any(not isinstance(value, str) for value in secrets.values()):
        raise HTTPException(status_code=422, detail="Secret values must be strings")

    candidate_values = Settings().model_dump()
    candidate_values.update(settings_updates)
    for field, value in secrets.items():
        if value:
            candidate_values[field] = value
    for field in clear_secrets:
        candidate_values[field] = ""

    try:
        updated_settings = Settings.model_validate(candidate_values)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not updated_settings.gateway_api_key:
        raise HTTPException(
            status_code=422,
            detail="GATEWAY_API_KEY cannot be cleared while environment administration is enabled",
        )

    try:
        saved_path = save_settings(updated_settings)
    except OSError as error:
        logger.exception("Could not persist environment configuration")
        raise HTTPException(status_code=500, detail="Could not persist environment configuration") from error
    logger.info("Environment configuration saved to %s; restart required to apply changes", saved_path)
    return {"saved": True, "restart_required": True, **settings_response(updated_settings)}


@app.get("/v1/document-loader/models")
async def list_document_parser_models(
    settings: Settings = Depends(get_settings),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    require_tool_api_key(credentials, settings)
    _require_document_loader(settings)
    parser = OllamaDocumentParser(
        settings.ollama_base_url,
        settings.document_parser_model,
        settings.document_parser_timeout_seconds,
    )
    try:
        models = await asyncio.to_thread(parser.list_models)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Could not query Ollama: {error}") from error
    finally:
        parser.close()
    return {"configured_model": settings.document_parser_model, "models": models}


async def _queue_document_loader_job(
    *,
    directory: Path,
    tenant_id: str,
    recursive: bool,
    include_extensions: frozenset[str] | None,
    exclude_extensions: frozenset[str],
    model: str | None,
    request: Request,
    settings: Settings,
    store: AtlasStore,
    embedder: VoyageEmbedder,
    cleanup_directory: bool = False,
) -> dict[str, Any]:
    tasks: set[asyncio.Task[Any]] = request.app.state.document_loader_tasks
    if sum(not task.done() for task in tasks) >= settings.document_loader_max_active_jobs:
        raise HTTPException(status_code=429, detail="Maximum active document-loader jobs reached")

    parser = OllamaDocumentParser(
        settings.ollama_base_url,
        model or settings.document_parser_model,
        settings.document_parser_timeout_seconds,
    )
    try:
        available_models = await asyncio.to_thread(parser.list_models)
    except httpx.HTTPError as error:
        parser.close()
        raise HTTPException(status_code=502, detail=f"Could not query Ollama: {error}") from error
    matching_model = next(
        (available for available in available_models if available.casefold() == parser.model.casefold()),
        None,
    )
    if matching_model is None:
        parser.close()
        raise HTTPException(
            status_code=422,
            detail=f"Parser model {parser.model!r} is not available in Ollama",
        )
    parser.model = matching_model

    job_id = uuid4().hex
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "directory": "browser upload" if cleanup_directory else str(directory),
        "tenant_id": tenant_id,
        "model": parser.model,
        "discovered": 0,
        "eligible": 0,
        "skipped": 0,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "current_file": None,
        "errors": [],
        "events": [{"stage": "queued", "message": "Job queued"}],
    }
    jobs: dict[str, dict[str, Any]] = request.app.state.document_loader_jobs
    jobs[job_id] = job

    async def ingest_parsed_document(document: ParsedDocument) -> dict[str, Any]:
        chunks = chunk_text(document.text, settings.chunk_size, settings.chunk_overlap)
        embeddings = await embedder.embed(chunks, input_type="document")
        count = await run_in_threadpool(
            store.replace_document,
            document.tenant_id,
            document.document_id,
            chunks,
            embeddings,
            document.metadata,
        )
        return {"chunks_upserted": count}

    def update_progress(event: dict[str, Any]) -> None:
        job.update({key: value for key, value in event.items() if key != "message"})
        job["status"] = event["stage"]
        job["events"].append(event)
        if len(job["events"]) > 250:
            del job["events"][:-250]

    async def run_job() -> None:
        job["status"] = "running"
        try:
            summary = await load_directory(
                LoaderOptions(
                    directory=directory,
                    tenant_id=tenant_id,
                    recursive=recursive,
                    include_extensions=include_extensions,
                    exclude_extensions=exclude_extensions,
                ),
                parser,
                ingest_parsed_document,
                update_progress,
                settings.max_document_chars,
            )
            job.update(asdict(summary))
            job["directory"] = "browser upload" if cleanup_directory else str(directory)
            job["tenant_id"] = tenant_id
            job["model"] = parser.model
            job["events"] = job["events"][-250:]
        except Exception as error:  # noqa: BLE001
            job["status"] = "failed"
            job["failed"] += 1
            job["errors"].append({"file": "", "error": str(error)})
            job["events"].append({"stage": "failed", "message": str(error)})
        finally:
            parser.close()
            if cleanup_directory:
                await asyncio.to_thread(shutil.rmtree, directory, True)

    task = asyncio.create_task(run_job())
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"job_id": job_id, "status": job["status"], "status_url": f"/v1/document-loader/jobs/{job_id}"}


@app.post("/v1/document-loader/jobs", status_code=202)
async def start_document_loader_job(
    body: DocumentLoaderRequest,
    request: Request,
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
    embedder: VoyageEmbedder = Depends(get_embedder),
) -> dict[str, Any]:
    _require_document_loader(settings)
    if allowed_tenants is not None and body.tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    directory = _resolve_loader_directory(body.directory, settings.document_loader_roots)
    try:
        include_extensions = normalize_extensions(body.include_extensions)
        exclude_extensions = (
            normalize_extensions(body.exclude_extensions, validate_supported=False)
            if body.exclude_extensions
            else ()
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return await _queue_document_loader_job(
        directory=directory,
        tenant_id=body.tenant_id,
        recursive=body.recursive,
        include_extensions=include_extensions,
        exclude_extensions=exclude_extensions,
        model=body.model,
        request=request,
        settings=settings,
        store=store,
        embedder=embedder,
    )


@app.post("/v1/document-loader/uploads", status_code=202)
async def upload_document_loader_files(
    request: Request,
    files: list[UploadFile] = File(...),
    relative_paths: str = Form(default="[]"),
    tenant_id: str = Form(default="default", min_length=1, max_length=256),
    model: str = Form(default=""),
    include_extensions: str = Form(default=""),
    exclude_extensions: str = Form(default=""),
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
    embedder: VoyageEmbedder = Depends(get_embedder),
) -> dict[str, Any]:
    _require_document_loader(settings)
    if allowed_tenants is not None and tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    tasks: set[asyncio.Task[Any]] = request.app.state.document_loader_tasks
    if sum(not task.done() for task in tasks) >= settings.document_loader_max_active_jobs:
        raise HTTPException(status_code=429, detail="Maximum active document-loader jobs reached")
    if not files:
        raise HTTPException(status_code=422, detail="Select at least one file")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"Upload is limited to {MAX_UPLOAD_FILES} files")
    try:
        submitted_paths = json.loads(relative_paths)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="relative_paths must be a JSON array") from error
    if (
        not isinstance(submitted_paths, list)
        or len(submitted_paths) != len(files)
        or not all(isinstance(path, str) for path in submitted_paths)
    ):
        raise HTTPException(
            status_code=422,
            detail="relative_paths must contain one path for each uploaded file",
        )
    try:
        include_filter = normalize_extensions(
            [extension for extension in include_extensions.split(",") if extension.strip()]
            if include_extensions.strip()
            else None
        )
        exclude_filter = (
            normalize_extensions(
                [extension for extension in exclude_extensions.split(",") if extension.strip()],
                validate_supported=False,
            )
            if exclude_extensions.strip()
            else frozenset()
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    temporary_directory = Path(tempfile.mkdtemp(prefix="rag-gateway-upload-"))
    total_bytes = 0
    seen_paths: set[str] = set()
    try:
        for upload, submitted_path in zip(files, submitted_paths, strict=True):
            normalized_path = submitted_path.replace("\\", "/")
            relative_path = PurePosixPath(normalized_path)
            if (
                not normalized_path
                or relative_path.is_absolute()
                or any(part in {"", ".", ".."} for part in normalized_path.split("/"))
                or "\x00" in normalized_path
                or len(normalized_path) > 512
            ):
                raise HTTPException(status_code=422, detail="Upload contains an invalid relative path")
            normalized_key = relative_path.as_posix()
            if normalized_key in seen_paths:
                raise HTTPException(status_code=422, detail="Upload contains duplicate relative paths")
            seen_paths.add(normalized_key)
            destination = temporary_directory.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            file_bytes = 0
            with destination.open("wb") as destination_file:
                while chunk := await upload.read(1024 * 1024):
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > MAX_SOURCE_FILE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{normalized_key} exceeds the 50 MiB per-file limit",
                        )
                    if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Upload exceeds the 200 MiB total limit",
                        )
                    await asyncio.to_thread(destination_file.write, chunk)
            await upload.close()
    except BaseException:
        await asyncio.to_thread(shutil.rmtree, temporary_directory, True)
        raise

    try:
        return await _queue_document_loader_job(
            directory=temporary_directory,
            tenant_id=tenant_id,
            recursive=True,
            include_extensions=include_filter,
            exclude_extensions=exclude_filter,
            model=model or None,
            request=request,
            settings=settings,
            store=store,
            embedder=embedder,
            cleanup_directory=True,
        )
    except BaseException:
        await asyncio.to_thread(shutil.rmtree, temporary_directory, True)
        raise


@app.get("/v1/document-loader/jobs/{job_id}")
async def get_document_loader_job(
    job_id: str,
    request: Request,
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
) -> dict[str, Any]:
    job = _job_for_tenant(job_id, request.app.state.document_loader_jobs, allowed_tenants)
    return dict(job)


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.public_model_name,
                "object": "model",
                "created": 0,
                "owned_by": "local-rag-gateway",
            }
        ],
    }


@app.get("/v1", dependencies=[Depends(require_api_key)])
async def api_root(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "object": "gateway",
        "name": app.title,
        "model": settings.public_model_name,
        "models_url": "/v1/models",
        "ollama_models_url": "/v1/api/tags",
        "chat_completions_url": "/v1/chat/completions",
    }


@app.get("/v1/api/tags", dependencies=[Depends(require_api_key)])
async def list_ollama_compatible_models(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    model_name = settings.public_model_name
    return {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "modified_at": "1970-01-01T00:00:00Z",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "api",
                    "family": "rag-gateway",
                    "families": [],
                    "parameter_size": "remote",
                    "quantization_level": "n/a",
                },
            }
        ]
    }


@app.get("/v1/api/ps", dependencies=[Depends(require_api_key)])
async def list_ollama_running_models() -> dict[str, list[Any]]:
    return {"models": []}


@app.get("/v1/api/version", dependencies=[Depends(require_api_key)])
async def ollama_compatible_version() -> dict[str, str]:
    return {"version": app.version}


@app.post("/v1/ingest", response_model=IngestResult)
async def ingest_document(
    body: IngestDocument,
    allowed_tenants: list[str] | None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
    embedder: VoyageEmbedder = Depends(get_embedder),
) -> IngestResult:
    if allowed_tenants is not None and body.tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    if len(body.text) > settings.max_document_chars:
        raise HTTPException(status_code=413, detail="Document exceeds MAX_DOCUMENT_CHARS")
    chunks = chunk_text(body.text, settings.chunk_size, settings.chunk_overlap)
    try:
        embeddings = await embedder.embed(chunks, input_type="document")
        count = await run_in_threadpool(
            store.replace_document,
            body.tenant_id,
            body.document_id,
            chunks,
            embeddings,
            body.metadata,
        )
    except (httpx.HTTPError, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {error}") from error
    return IngestResult(
        document_id=body.document_id,
        tenant_id=body.tenant_id,
        chunks_upserted=count,
    )


@app.delete(
    "/v1/documents/{document_id:path}",
)
async def delete_document(
    document_id: str,
    x_tenant_id: str | None = Header(default=None),
    allowed_tenants: list[str] | None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
) -> dict[str, int]:
    tenant_id = get_tenant_id(x_tenant_id, settings, allowed_tenants)
    deleted = await run_in_threadpool(store.delete_document, tenant_id, document_id)
    return {"deleted_chunks": deleted}


@app.get("/v1/chunks/tenants")
async def list_chunk_tenants(
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
    store: AtlasStore = Depends(get_store),
) -> dict[str, Any]:
    statistics = await run_in_threadpool(store.list_tenant_statistics, allowed_tenants)
    return {"tenants": statistics}


@app.get("/v1/chunks/documents")
async def list_chunk_documents(
    tenant_id: str,
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
    store: AtlasStore = Depends(get_store),
) -> dict[str, Any]:
    if not tenant_id or len(tenant_id) > 256:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")
    if allowed_tenants is not None and tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    documents = await run_in_threadpool(store.list_documents, tenant_id)
    return {"tenant_id": tenant_id, "documents": documents}


@app.delete("/v1/chunks/documents/{document_id:path}")
async def delete_chunk_document(
    document_id: str,
    tenant_id: str,
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
    store: AtlasStore = Depends(get_store),
) -> dict[str, int]:
    if not tenant_id or len(tenant_id) > 256:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")
    if allowed_tenants is not None and tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    deleted = await run_in_threadpool(store.delete_document, tenant_id, document_id)
    return {"deleted_chunks": deleted}


@app.delete("/v1/chunks/tenant")
async def delete_chunk_tenant(
    tenant_id: str,
    allowed_tenants: list[str] | None = Depends(require_tool_api_key),
    store: AtlasStore = Depends(get_store),
) -> dict[str, int]:
    if not tenant_id or len(tenant_id) > 256:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")
    if allowed_tenants is not None and tenant_id not in allowed_tenants:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    deleted = await run_in_threadpool(store.delete_tenant, tenant_id)
    return {"deleted_chunks": deleted}


@app.post("/v1/retrieve")
async def retrieve(
    body: dict[str, Any],
    x_tenant_id: str | None = Header(default=None),
    allowed_tenants: list[str] | None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
    embedder: VoyageEmbedder = Depends(get_embedder),
) -> dict[str, Any]:
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=422, detail="query must be a non-empty string")
    tenant_id = get_tenant_id(x_tenant_id, settings, allowed_tenants)
    try:
        sources = await retrieve_context(query[: settings.max_query_chars], tenant_id, settings, embedder, store)
    except (httpx.HTTPError, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {error}") from error
    return {"data": sources}


def _chat_payload(body: dict[str, Any], messages: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    forwarded_keys = (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
    )
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    payload.update({key: body[key] for key in forwarded_keys if key in body})
    return payload


def _ollama_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options")
    options = options if isinstance(options, dict) else {}
    payload: dict[str, Any] = {
        "messages": body.get("messages"),
        "stream": body.get("stream", True),
    }
    option_mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "num_predict": "max_tokens",
        "stop": "stop",
        "seed": "seed",
    }
    payload.update(
        {
            target: options[source]
            for source, target in option_mapping.items()
            if source in options
        }
    )
    if body.get("format") == "json":
        payload["response_format"] = {"type": "json_object"}
    return payload


def _ollama_chat_result(result: dict[str, Any], model: str) -> dict[str, Any]:
    choices = result.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    response: dict[str, Any] = {
        "model": model,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "message": {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        },
        "done": True,
        "done_reason": choice.get("finish_reason"),
    }
    if "sources" in result:
        response["sources"] = result["sources"]
    return response


async def _ollama_chat_stream(response: StreamingResponse, model: str) -> AsyncIterator[bytes]:
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    buffer = b""
    async for chunk in response.body_iterator:
        buffer += chunk.encode() if isinstance(chunk, str) else chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.startswith(b"data: "):
                event = line[6:]
                if event == b"[DONE]":
                    yield (
                        json.dumps(
                            {
                                "model": model,
                                "created_at": created_at,
                                "message": {"role": "assistant", "content": ""},
                                "done": True,
                            }
                        )
                        + "\n"
                    ).encode()
                    return
                chunk_data = json.loads(event)
                if "error" in chunk_data:
                    yield (json.dumps(chunk_data) + "\n").encode()
                    continue
                choice = (chunk_data.get("choices") or [{}])[0]
                delta = choice.get("delta", {})
                ollama_chunk: dict[str, Any] = {
                    "model": model,
                    "created_at": created_at,
                    "message": {
                        "role": delta.get("role", "assistant"),
                        "content": delta.get("content", ""),
                    },
                    "done": choice.get("finish_reason") is not None,
                }
                if "sources" in chunk_data:
                    ollama_chunk["sources"] = chunk_data["sources"]
                yield (json.dumps(ollama_chunk) + "\n").encode()


@app.post("/v1/chat/completions")
async def chat_completions(
    body: dict[str, Any],
    request: Request,
    x_tenant_id: str | None = Header(default=None),
    allowed_tenants: list[str] | None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
    embedder: VoyageEmbedder = Depends(get_embedder),
) -> Response:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=422, detail="messages must be a non-empty array")
    query = extract_user_query(messages, settings.max_query_chars)
    sources: list[dict[str, Any]] = []
    if query:
        tenant_id = get_tenant_id(x_tenant_id, settings, allowed_tenants)
        try:
            sources = await retrieve_context(query, tenant_id, settings, embedder, store)
        except (httpx.HTTPError, RuntimeError) as error:
            raise HTTPException(status_code=502, detail=f"Retrieval failed: {error}") from error

    augmented_messages = add_rag_system_message(messages, sources)
    payload = _chat_payload(body, augmented_messages, settings)
    endpoint = f"{settings.ollama_base_url.rstrip('/')}/chat/completions"
    if payload["stream"]:
        return StreamingResponse(
            _stream_completion(request.app.state.http, endpoint, payload, sources),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        upstream = await request.app.state.http.post(endpoint, json=payload)
        upstream.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Local Ollama chat request failed") from error
    result = upstream.json()
    result["model"] = body.get("model", settings.public_model_name)
    result["sources"] = [
        {
            "citation": f"S{index}",
            "document_id": source["document_id"],
            "chunk_index": source["chunk_index"],
            "score": source.get("score"),
            "metadata": source.get("metadata", {}),
        }
        for index, source in enumerate(sources, start=1)
    ]
    return JSONResponse(content=result)


@app.post("/v1/api/chat")
async def ollama_compatible_chat(
    body: dict[str, Any],
    request: Request,
    x_tenant_id: str | None = Header(default=None),
    allowed_tenants: list[str] | None = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    store: AtlasStore = Depends(get_store),
    embedder: VoyageEmbedder = Depends(get_embedder),
) -> Response:
    chat_body = _ollama_chat_body(body)
    if not isinstance(chat_body["messages"], list) or not chat_body["messages"]:
        raise HTTPException(status_code=422, detail="messages must be a non-empty array")
    response = await chat_completions(
        chat_body,
        request,
        x_tenant_id,
        allowed_tenants,
        settings,
        store,
        embedder,
    )
    if isinstance(response, StreamingResponse):
        return StreamingResponse(
            _ollama_chat_stream(response, settings.public_model_name),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = json.loads(response.body)
    return JSONResponse(content=_ollama_chat_result(result, settings.public_model_name))


async def _stream_completion(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    sources: list[dict[str, Any]],
) -> AsyncIterator[bytes]:
    source_event = {
        "id": "rag-sources",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
        "sources": [
            {
                "citation": f"S{index}",
                "document_id": source["document_id"],
                "chunk_index": source["chunk_index"],
                "score": source.get("score"),
                "metadata": source.get("metadata", {}),
            }
            for index, source in enumerate(sources, start=1)
        ],
    }
    yield f"data: {json.dumps(source_event)}\n\n".encode()
    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            if response.is_error:
                yield b'data: {"error":{"message":"Local Ollama chat request failed"}}\n\n'
                yield b"data: [DONE]\n\n"
                return
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n".encode()
    except httpx.HTTPError:
        yield b'data: {"error":{"message":"Local Ollama chat request failed"}}\n\n'
        yield b"data: [DONE]\n\n"
