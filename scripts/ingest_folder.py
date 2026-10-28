from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest UTF-8 text and Markdown files through the RAG gateway."
    )
    parser.add_argument("folder", type=Path, help="Folder to scan recursively")
    parser.add_argument("--tenant", default="default", help="Tenant ID used for retrieval filtering")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Gateway origin URL")
    args = parser.parse_args()

    api_key = os.environ.get("GATEWAY_API_KEY")
    if not api_key:
        parser.error("Set GATEWAY_API_KEY in the environment before running ingestion")
    if not args.folder.is_dir():
        parser.error(f"Folder does not exist: {args.folder}")

    files = sorted(
        path
        for path in args.folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        print("No .txt or .md files found.")
        return 0

    endpoint = f"{args.base_url.rstrip('/')}/v1/ingest"
    headers = {"Authorization": f"Bearer {api_key}"}
    failures = 0
    with httpx.Client(timeout=180.0) as client:
        for path in files:
            relative_path = path.relative_to(args.folder).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
                response = client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "tenant_id": args.tenant,
                        "document_id": relative_path,
                        "text": text,
                        "metadata": {"source": relative_path},
                    },
                )
                response.raise_for_status()
                result = response.json()
                print(f"{relative_path}: {result['chunks_upserted']} chunks")
            except (OSError, UnicodeError, httpx.HTTPError, KeyError, ValueError) as error:
                failures += 1
                print(f"{relative_path}: failed ({error})")

    print(f"Processed {len(files)} files; failures: {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
