from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gateway_api_key: str = ""
    api_key_registry: dict[str, list[str]] = Field(default_factory=dict)
    mongodb_uri: str = ""
    mongodb_api_model_endpoint: str = "ai.mongodb.com"
    mongodb_api_key: str = ""
    mongodb_database: str = "rag_gateway"
    mongodb_collection: str = "chunks"
    atlas_vector_index: str = "rag_chunks_vector"
    voyage_embedding_model: str = "voyage-3.5"
    embedding_dimensions: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1, le=128)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    public_model_name: str = "rag-local"
    default_tenant_id: str = "default"
    chunk_size: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_num_candidates: int = Field(default=100, ge=1)
    retrieval_min_score: float = Field(default=0.72, ge=0.0, le=1.0)
    max_document_chars: int = Field(default=1_000_000, ge=1)
    max_query_chars: int = Field(default=8_000, ge=1)
    upstream_timeout_seconds: float = Field(default=180.0, gt=0)
    document_loader_enabled: bool = False
    document_loader_roots: list[Path] = Field(default_factory=list)
    document_loader_max_active_jobs: int = Field(default=2, ge=1, le=8)
    document_parser_model: str = "glm-ocr:latest"
    document_parser_timeout_seconds: float = Field(default=300.0, gt=0)

    @property
    def mongodb_api_model_base_url(self) -> str:
        endpoint = self.mongodb_api_model_endpoint.strip().rstrip("/")
        if "://" not in endpoint:
            endpoint = f"https://{endpoint}"
        return endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
