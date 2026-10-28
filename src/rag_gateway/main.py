import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo.errors import PyMongoError
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse, StreamingResponse

from rag_gateway.chunking import chunk_text
from rag_gateway.config import Settings, get_settings
from rag_gateway.document_loader import (
    IngestDocument as ParsedDocument,
)
from rag_gateway.document_loader import (
    LoaderOptions,
    OllamaDocumentParser,
    load_directory,
    normalize_extensions,
)
from rag_gateway.models import DocumentLoaderRequest, IngestDocument, IngestResult
from rag_gateway.rag import add_rag_system_message, extract_user_query, retrieve_context
from rag_gateway.store import AtlasStore
from rag_gateway.voyage import VoyageEmbedder

bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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


@app.get("/v1/document-loader/models")
async def list_document_parser_models(
    settings: Settings = Depends(get_settings),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    require_api_key(credentials, settings)
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


@app.post("/v1/document-loader/jobs", status_code=202)
async def start_document_loader_job(
    body: DocumentLoaderRequest,
    request: Request,
    allowed_tenants: list[str] | None = Depends(require_api_key),
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

    tasks: set[asyncio.Task[Any]] = request.app.state.document_loader_tasks
    if sum(not task.done() for task in tasks) >= settings.document_loader_max_active_jobs:
        raise HTTPException(status_code=429, detail="Maximum active document-loader jobs reached")

    parser = OllamaDocumentParser(
        settings.ollama_base_url,
        body.model or settings.document_parser_model,
        settings.document_parser_timeout_seconds,
    )
    try:
        available_models = await asyncio.to_thread(parser.list_models)
    except httpx.HTTPError as error:
        parser.close()
        raise HTTPException(status_code=502, detail=f"Could not query Ollama: {error}") from error
    matching_model = next(
        (model for model in available_models if model.casefold() == parser.model.casefold()),
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
        "directory": str(directory),
        "tenant_id": body.tenant_id,
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
                    tenant_id=body.tenant_id,
                    recursive=body.recursive,
                    include_extensions=include_extensions,
                    exclude_extensions=exclude_extensions,
                ),
                parser,
                ingest_parsed_document,
                update_progress,
                settings.max_document_chars,
            )
            job.update(asdict(summary))
            job["directory"] = str(directory)
            job["tenant_id"] = body.tenant_id
            job["model"] = parser.model
            job["events"] = job["events"][-250:]
        except Exception as error:  # noqa: BLE001
            job["status"] = "failed"
            job["failed"] += 1
            job["errors"].append({"file": "", "error": str(error)})
            job["events"].append({"stage": "failed", "message": str(error)})
        finally:
            parser.close()

    task = asyncio.create_task(run_job())
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"job_id": job_id, "status": job["status"], "status_url": f"/v1/document-loader/jobs/{job_id}"}


@app.get("/v1/document-loader/jobs/{job_id}")
async def get_document_loader_job(
    job_id: str,
    request: Request,
    allowed_tenants: list[str] | None = Depends(require_api_key),
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
