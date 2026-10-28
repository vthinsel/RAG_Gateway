from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from rag_gateway.document_loader import (
    DEFAULT_INCLUDE_EXTENSIONS,
    IngestDocument,
    LoaderOptions,
    OllamaDocumentParser,
    load_directory,
)


def _extensions(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [extension.strip() for extension in value.split(",") if extension.strip()]


def _report(event: dict[str, Any]) -> None:
    stage = event["stage"].upper().replace("_", " ")
    current = event.get("current")
    total = event.get("total")
    position = f" [{current}/{total}]" if current is not None else ""
    print(f"{stage}{position}: {event['message']}", flush=True)


async def _run(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("GATEWAY_API_KEY")
    if not api_key:
        print("Provide --api-key or set GATEWAY_API_KEY.", file=sys.stderr)
        return 2

    parser = OllamaDocumentParser(args.ollama_url, args.model, args.timeout)
    try:
        models = await asyncio.to_thread(parser.list_models)
        matching_model = next((name for name in models if name.casefold() == args.model.casefold()), None)
        if matching_model is None:
            print(f"Model {args.model!r} is not available in Ollama. Installed: {', '.join(models) or 'none'}", file=sys.stderr)
            return 2
        parser.model = matching_model

        async with httpx.AsyncClient(timeout=args.timeout) as client:
            async def ingest(document: IngestDocument) -> dict[str, Any]:
                response = await client.post(
                    f"{args.gateway_url.rstrip('/')}/v1/ingest",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "tenant_id": document.tenant_id,
                        "document_id": document.document_id,
                        "text": document.text,
                        "metadata": document.metadata,
                    },
                )
                response.raise_for_status()
                return response.json()

            summary = await load_directory(
                LoaderOptions(
                    directory=args.directory,
                    tenant_id=args.tenant,
                    recursive=not args.no_recursive,
                    include_extensions=_extensions(args.include),
                    exclude_extensions=_extensions(args.exclude) or (),
                ),
                parser,
                ingest,
                progress=_report,
            )
    except (OSError, ValueError, httpx.HTTPError, RuntimeError) as error:
        print(f"Document loading failed: {error}", file=sys.stderr)
        return 1
    finally:
        parser.close()

    print(
        f"Overall: {summary.status}; {summary.succeeded} succeeded, {summary.failed} failed, "
        f"{summary.skipped} skipped in {summary.elapsed_seconds:.1f}s."
    )
    return 1 if summary.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse local documents with an OCR model via Ollama, then ingest them into the RAG gateway."
    )
    parser.add_argument("directory", type=Path, help="Directory to scan")
    parser.add_argument("--tenant", default="default", help="Authorized tenant ID")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="RAG gateway origin")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server origin")
    parser.add_argument("--model", default="glm-ocr:latest", help="Installed Ollama parser model tag")
    parser.add_argument("--api-key", help="Gateway bearer key; defaults to GATEWAY_API_KEY")
    parser.add_argument("--include", help="Comma-separated extensions to include; defaults to supported types")
    parser.add_argument("--exclude", default="", help="Comma-separated extensions to exclude")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan files directly in the directory")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout for parser and gateway requests")
    args = parser.parse_args()
    print("Default supported types: " + ", ".join(DEFAULT_INCLUDE_EXTENSIONS))
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())