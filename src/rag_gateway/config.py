import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    input_per_million_usd: float | None = Field(default=None, ge=0)
    output_per_million_usd: float | None = Field(default=None, ge=0)
    per_request_usd: float | None = Field(default=None, ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        if os.getenv("RAG_GATEWAY_DOTENV_PRECEDENCE", "").lower() in {"1", "true", "yes"}:
            return init_settings, dotenv_settings, env_settings, file_secret_settings
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    gateway_bind_address: str = "127.0.0.1"
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    gateway_api_key: str = ""
    api_key_registry: dict[str, list[str]] = Field(default_factory=dict)
    mongodb_uri: str = ""
    mongodb_api_model_endpoint: str = "ai.mongodb.com"
    mongodb_api_key: str = ""
    mongodb_database: str = "rag_gateway"
    mongodb_collection: str = "chunks"
    atlas_vector_index: str = "rag_chunks_vector"
    atlas_search_index: str = "rag_chunks_text"
    voyage_embedding_model: str = "voyage-4"
    embedding_dimensions: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1, le=128)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3.5:9b"
    ollama_reasoning_effort: str = Field(default="", max_length=64)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    public_model_name: str = "rag-local"
    default_tenant_id: str = "default"
    chunk_size: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_num_candidates: int = Field(default=100, ge=1)
    retrieval_min_score: float = Field(default=0.72, ge=0.0, le=1.0)
    retrieval_rerank_enabled: bool = False
    retrieval_rerank_model: str = Field(default="rerank-2.5", min_length=1)
    retrieval_rerank_candidates: int = Field(default=100, ge=1, le=1000)
    retrieval_hybrid_enabled: bool = False
    retrieval_hybrid_candidates: int = Field(default=100, ge=1, le=1000)
    max_document_chars: int = Field(default=1_000_000, ge=1)
    max_query_chars: int = Field(default=8_000, ge=1)
    upstream_timeout_seconds: float = Field(default=180.0, gt=0)
    ollama_admin_timeout_seconds: float = Field(default=1800.0, gt=0)
    document_loader_enabled: bool = False
    document_loader_roots: list[Path] = Field(default_factory=list)
    document_loader_max_active_jobs: int = Field(default=2, ge=1, le=8)
    document_parser_model: str = "glm-ocr:latest"
    document_parser_timeout_seconds: float = Field(default=300.0, gt=0)
    operations_db_path: Path = Path("data/rag_gateway_operations.sqlite3")
    otel_traces_enabled: bool = False
    otel_service_name: str = Field(default="rag-gateway", min_length=1, max_length=128)
    otel_exporter_otlp_traces_endpoint: str = "http://localhost:4318/v1/traces"
    otel_traces_sample_ratio: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    @field_validator("otel_exporter_otlp_traces_endpoint")
    @classmethod
    def validate_trace_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
        ):
            raise ValueError("Trace endpoint must be an HTTP(S) URL without credentials or query")
        return value

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

    @field_validator("ollama_reasoning_effort")
    @classmethod
    def normalize_ollama_reasoning_effort(cls, value: str) -> str:
        value = value.strip()
        if any(character in value for character in "\r\n"):
            raise ValueError("OLLAMA_REASONING_EFFORT must not contain line breaks")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
