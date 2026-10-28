from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class RetrievalEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8000)
    expected_document_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank")
        return value

    @field_validator("expected_document_ids")
    @classmethod
    def validate_expected_document_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Expected document IDs must not be blank")
        return list(dict.fromkeys(normalized))


class RetrievalEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(default="default", min_length=1, max_length=256)
    cases: list[RetrievalEvaluationCase] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=50)
