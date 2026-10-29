from __future__ import annotations

import asyncio
import base64
import csv
import html.parser
import importlib
import io
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

DEFAULT_INCLUDE_EXTENSIONS = (
    ".pdf",
    ".txt",
    ".md",
    ".mmd",
    ".docx",
    ".xlsx",
    ".xlsm",
    ".pptx",
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
)
SUPPORTED_EXTENSIONS = frozenset(DEFAULT_INCLUDE_EXTENSIONS)
OLLAMA_OCR_PROMPT = (
    "Extract all text from this image. Preserve the document structure in Markdown. "
    "Return only the extracted text."
)
MAX_SOURCE_FILE_BYTES = 50 * 1024 * 1024
MAX_PAGE_WIDTH = 1664
MAX_PAGE_HEIGHT = 2048


class ImageDocumentParser(Protocol):
    model: str

    def parse_image(self, image: bytes) -> str: ...


@dataclass(frozen=True)
class LoaderOptions:
    directory: Path
    tenant_id: str = "default"
    recursive: bool = True
    include_extensions: Sequence[str] | None = None
    exclude_extensions: Sequence[str] = ()
    file_paths: Sequence[str] | None = None


@dataclass(frozen=True)
class FileScan:
    files: list[Path]
    discovered: int
    skipped: int


@dataclass(frozen=True)
class IngestDocument:
    tenant_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadSummary:
    directory: str
    status: str
    discovered: int
    eligible: int
    skipped: int
    processed: int
    succeeded: int
    failed: int
    errors: list[dict[str, str]]
    elapsed_seconds: float


def normalize_extensions(
    values: Sequence[str] | None, *, validate_supported: bool = True
) -> frozenset[str]:
    if values is None:
        return SUPPORTED_EXTENSIONS
    normalized = frozenset(
        value.strip().casefold()
        if value.strip().startswith(".")
        else f".{value.strip().casefold()}"
        for value in values
        if value.strip()
    )
    unsupported = normalized - SUPPORTED_EXTENSIONS
    if validate_supported and unsupported:
        raise ValueError(f"Unsupported file extension(s): {', '.join(sorted(unsupported))}")
    if not normalized:
        raise ValueError("At least one file extension must be included")
    return normalized


def scan_directory(options: LoaderOptions) -> FileScan:
    root = options.directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Directory does not exist or is not accessible: {root}")

    included = normalize_extensions(options.include_extensions)
    excluded = (
        normalize_extensions(options.exclude_extensions, validate_supported=False)
        if options.exclude_extensions
        else frozenset()
    )
    candidates = root.rglob("*") if options.recursive else root.iterdir()
    files = sorted(
        path
        for path in candidates
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
    )
    selected = [
        path
        for path in files
        if path.suffix.casefold() in included and path.suffix.casefold() not in excluded
    ]
    return FileScan(files=selected, discovered=len(files), skipped=len(files) - len(selected))


class _VisibleTextParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "noscript"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth and data.strip():
            self.parts.append(data.strip())


class OllamaDocumentParser:
    """Extract common formats and use a local Ollama model to parse page images."""

    def __init__(
        self,
        ollama_base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        base_url = ollama_base_url.rstrip("/")
        base_url = base_url.removesuffix("/v1")
        self.base_url = base_url
        self.model = model
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def list_models(self) -> list[str]:
        response = self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return sorted(item["name"] for item in response.json().get("models", []))

    def parse_image(self, image: bytes) -> str:
        response = self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": OLLAMA_OCR_PROMPT,
                        "images": [base64.b64encode(image).decode("ascii")],
                    }
                ],
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama model {self.model!r} returned no parsed text")
        return content.strip()

    def extract(self, path: Path) -> str:
        extension = path.suffix.casefold()
        if extension in {".txt", ".md", ".mmd", ".json", ".xml", ".yaml", ".yml", ".log"}:
            return path.read_text(encoding="utf-8-sig").strip()
        if extension == ".csv":
            return self._extract_csv(path)
        if extension in {".html", ".htm"}:
            parser = _VisibleTextParser()
            parser.feed(path.read_text(encoding="utf-8-sig"))
            return "\n".join(parser.parts)
        if extension == ".docx":
            return self._extract_docx(path)
        if extension in {".xlsx", ".xlsm"}:
            return self._extract_workbook(path)
        if extension == ".pptx":
            return self._extract_presentation(path)
        if extension == ".pdf":
            return self._extract_pdf(path)
        if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return self._extract_image(path)
        raise ValueError(f"No parser is available for {extension or 'extensionless files'}")

    @staticmethod
    def _extract_csv(path: Path) -> str:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)

    @staticmethod
    def _extract_docx(path: Path) -> str:
        try:
            Document = importlib.import_module("docx").Document
        except ImportError as error:
            raise RuntimeError("Install python-docx to parse DOCX files") from error

        document = Document(path)
        sections = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table_index, table in enumerate(document.tables, start=1):
            rows = [" | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells) for row in table.rows]
            sections.append(f"Table {table_index}\n" + "\n".join(rows))
        return "\n\n".join(sections)

    @staticmethod
    def _extract_workbook(path: Path) -> str:
        try:
            load_workbook = importlib.import_module("openpyxl").load_workbook
        except ImportError as error:
            raise RuntimeError("Install openpyxl to parse Excel files") from error

        workbook = load_workbook(path, read_only=True, data_only=True)
        sections: list[str] = []
        try:
            for sheet in workbook.worksheets:
                rows = []
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value).strip() if value is not None else "" for value in row]
                    if any(values):
                        rows.append(" | ".join(values))
                if rows:
                    sections.append(f"## {sheet.title}\n" + "\n".join(rows))
        finally:
            workbook.close()
        return "\n\n".join(sections)

    @staticmethod
    def _extract_presentation(path: Path) -> str:
        try:
            Presentation = importlib.import_module("pptx").Presentation
        except ImportError as error:
            raise RuntimeError("Install python-pptx to parse PowerPoint files") from error

        presentation = Presentation(path)
        slides = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    parts.append(shape.text.strip())
                if getattr(shape, "has_table", False):
                    parts.extend(
                        " | ".join(cell.text.strip() for cell in row.cells)
                        for row in shape.table.rows
                    )
            if parts:
                slides.append(f"## Slide {slide_number}\n" + "\n".join(parts))
        return "\n\n".join(slides)

    def _extract_pdf(self, path: Path) -> str:
        try:
            fitz = importlib.import_module("fitz")
        except ImportError as error:
            raise RuntimeError("Install PyMuPDF to render PDF pages for OCR") from error

        pages = []
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                scale = min(1.5, MAX_PAGE_WIDTH / page.rect.width, MAX_PAGE_HEIGHT / page.rect.height)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                text = self.parse_image(pixmap.tobytes("png"))
                pages.append(f"## Page {page_number}\n{text}")
        return "\n\n".join(pages)

    def _extract_image(self, path: Path) -> str:
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg"}:
            return self.parse_image(path.read_bytes())

        try:
            from PIL import Image, ImageSequence
        except ImportError as error:
            raise RuntimeError("Install Pillow to convert TIFF/BMP images for Ollama OCR") from error

        parsed_pages = []
        with Image.open(path) as image:
            for frame_number, frame in enumerate(ImageSequence.Iterator(image), start=1):
                output = io.BytesIO()
                frame = frame.convert("RGB")
                frame.thumbnail((MAX_PAGE_WIDTH, MAX_PAGE_HEIGHT))
                frame.save(output, format="PNG")
                text = self.parse_image(output.getvalue())
                parsed_pages.append(f"## Image {frame_number}\n{text}")
        return "\n\n".join(parsed_pages)


ProgressCallback = Callable[[dict[str, Any]], None]
IngestCallback = Callable[[IngestDocument], Awaitable[dict[str, Any]]]


async def load_directory(
    options: LoaderOptions,
    parser: OllamaDocumentParser,
    ingest: IngestCallback,
    progress: ProgressCallback | None = None,
    max_document_chars: int = 1_000_000,
) -> LoadSummary:
    started = time.monotonic()

    def emit(stage: str, message: str, **details: Any) -> None:
        if progress is not None:
            progress({"stage": stage, "message": message, **details})

    emit("scanning", "Scanning directory", directory=str(options.directory))
    scan = await asyncio.to_thread(scan_directory, options)
    files = scan.files
    if options.file_paths is not None:
        selected_paths = set(options.file_paths)
        files = [
            path
            for path in files
            if path.relative_to(options.directory.expanduser().resolve()).as_posix()
            in selected_paths
        ]
    total = len(files)
    emit(
        "scanned",
        f"Found {scan.discovered} files; {total} match the selected extensions",
        discovered=scan.discovered,
        eligible=total,
        skipped=scan.skipped + len(scan.files) - total,
    )

    succeeded = 0
    failed = 0
    errors: list[dict[str, str]] = []
    root = options.directory.expanduser().resolve()
    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(root).as_posix()
        emit("parsing", f"Parsing {relative_path}", current_file=relative_path, current=index, total=total)
        try:
            if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                raise ValueError(f"File exceeds the {MAX_SOURCE_FILE_BYTES // (1024 * 1024)} MiB limit")
            text = (await asyncio.to_thread(parser.extract, path)).strip()
            if not text:
                raise ValueError("No text was extracted")
            if len(text) > max_document_chars:
                raise ValueError(
                    f"Parsed text exceeds MAX_DOCUMENT_CHARS ({max_document_chars:,} characters)"
                )
            emit("ingesting", f"Sending {relative_path} to the RAG gateway", current_file=relative_path)
            result = await ingest(
                IngestDocument(
                    tenant_id=options.tenant_id,
                    document_id=relative_path,
                    text=text,
                    metadata={
                        "source": relative_path,
                        "loader": "ollama-document-loader",
                        "parser_model": parser.model,
                    },
                )
            )
            succeeded += 1
            emit(
                "succeeded",
                f"Loaded {relative_path}",
                current_file=relative_path,
                chunks_upserted=result.get("chunks_upserted"),
                processed=index,
                succeeded=succeeded,
                failed=failed,
            )
        except Exception as error:  # noqa: BLE001
            failed += 1
            error_details = {"file": relative_path, "error": str(error)}
            errors.append(error_details)
            emit(
                "file_failed",
                f"Failed to load {relative_path}: {error}",
                current_file=relative_path,
                processed=index,
                succeeded=succeeded,
                failed=failed,
                error=str(error),
            )

    status = "completed_with_errors" if failed else "completed"
    elapsed = time.monotonic() - started
    emit(
        status,
        f"Loading finished: {succeeded} succeeded, {failed} failed, {scan.skipped} skipped",
        processed=total,
        succeeded=succeeded,
        failed=failed,
        skipped=scan.skipped + len(scan.files) - total,
        elapsed_seconds=elapsed,
    )
    return LoadSummary(
        directory=str(root),
        status=status,
        discovered=scan.discovered,
        eligible=total,
        skipped=scan.skipped + len(scan.files) - total,
        processed=total,
        succeeded=succeeded,
        failed=failed,
        errors=errors,
        elapsed_seconds=elapsed,
    )