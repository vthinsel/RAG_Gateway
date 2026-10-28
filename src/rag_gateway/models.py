from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=512)
    tenant_id: str = Field(default="default", min_length=1, max_length=256)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    document_id: str
    tenant_id: str
    chunks_upserted: int


class DocumentLoaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: Path
    tenant_id: str = Field(default="default", min_length=1, max_length=256)
    recursive: bool = True
    include_extensions: list[str] | None = None
    exclude_extensions: list[str] = Field(default_factory=list)
    model: str | None = Field(default=None, min_length=1, max_length=256)
