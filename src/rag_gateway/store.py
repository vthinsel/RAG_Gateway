import logging
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pymongo import MongoClient, ReplaceOne

from rag_gateway.config import Settings

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class AtlasStore:
    def __init__(self, settings: Settings) -> None:
        if not settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI is not configured")
        self.settings = settings
        self.client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)
        self.collection = self.client[settings.mongodb_database][settings.mongodb_collection]
        self.collection.create_index([("tenant_id", 1), ("document_id", 1)])

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True

    def vector_index_status(self) -> tuple[bool, bool]:
        indexes = list(self.collection.list_search_indexes(name=self.settings.atlas_vector_index))
        if not indexes:
            return False, False
        index = indexes[0]
        queryable = index.get("queryable")
        if not isinstance(queryable, bool):
            queryable = index.get("status") == "READY"
        return True, queryable

    def list_tenant_ids(self) -> list[str]:
        tenant_ids = self.collection.distinct("tenant_id")
        return sorted(tenant_id for tenant_id in tenant_ids if isinstance(tenant_id, str))

    def list_tenant_statistics(self, tenant_ids: list[str] | None = None) -> list[dict[str, Any]]:
        tenant_filter: dict[str, Any] = {"$type": "string"}
        if tenant_ids is not None:
            tenant_filter["$in"] = tenant_ids
        match = {"tenant_id": tenant_filter}
        return list(
            self.collection.aggregate(
                [
                    {"$match": match},
                    {
                        "$group": {
                            "_id": {
                                "tenant_id": "$tenant_id",
                                "document_id": "$document_id",
                            },
                            "chunks": {"$sum": 1},
                        }
                    },
                    {
                        "$group": {
                            "_id": "$_id.tenant_id",
                            "documents": {"$sum": 1},
                            "chunks": {"$sum": "$chunks"},
                        }
                    },
                    {"$sort": {"_id": 1}},
                    {
                        "$project": {
                            "_id": 0,
                            "tenant_id": "$_id",
                            "documents": 1,
                            "chunks": 1,
                        }
                    },
                ]
            )
        )

    def list_documents(self, tenant_id: str) -> list[dict[str, Any]]:
        return list(
            self.collection.aggregate(
                [
                    {"$match": {"tenant_id": tenant_id}},
                    {
                        "$group": {
                            "_id": "$document_id",
                            "chunks": {"$sum": 1},
                            "source": {"$first": "$metadata.source"},
                            "updated_at": {"$max": "$updated_at"},
                        }
                    },
                    {"$sort": {"_id": 1}},
                    {
                        "$project": {
                            "_id": 0,
                            "document_id": "$_id",
                            "source": 1,
                            "chunks": 1,
                            "updated_at": 1,
                        }
                    },
                ]
            )
        )

    def delete_tenant(self, tenant_id: str) -> int:
        return self.collection.delete_many({"tenant_id": tenant_id}).deleted_count

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _chunk_id(tenant_id: str, document_id: str, chunk_index: int) -> str:
        key = f"{tenant_id}\0{document_id}\0{chunk_index}".encode()
        return sha256(key).hexdigest()

    def replace_document(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        now = datetime.now(UTC)
        operations = []
        for index, (text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = self._chunk_id(tenant_id, document_id, index)
            document = {
                "_id": chunk_id,
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_index": index,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
                "embedding_model": self.settings.voyage_embedding_model,
                "embedding_dim": self.settings.embedding_dimensions,
                "chunk_policy": (
                    f"char-v1:{self.settings.chunk_size}:{self.settings.chunk_overlap}"
                ),
                "content_sha256": sha256(text.encode()).hexdigest(),
                "updated_at": now,
            }
            operations.append(ReplaceOne({"_id": chunk_id}, document, upsert=True))
        if operations:
            self.collection.bulk_write(operations, ordered=True)
        self.collection.delete_many(
            {"tenant_id": tenant_id, "document_id": document_id, "chunk_index": {"$gte": len(chunks)}}
        )
        return len(operations)

    def delete_document(self, tenant_id: str, document_id: str) -> int:
        return self.collection.delete_many(
            {"tenant_id": tenant_id, "document_id": document_id}
        ).deleted_count

    def search(self, tenant_id: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.settings.atlas_vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(self.settings.retrieval_num_candidates, limit),
                    "limit": limit,
                    "filter": {"tenant_id": {"$eq": tenant_id}},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "document_id": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        candidates = list(self.collection.aggregate(pipeline))
        matches = [
            candidate
            for candidate in candidates
            if candidate["score"] >= self.settings.retrieval_min_score
        ]
        if not matches:
            if candidates:
                logger.warning(
                    "Atlas retrieval found vector candidates but filtered them all: "
                    "tenant_id=%s index=%s candidates=%d best_score=%.4f min_score=%.4f",
                    tenant_id,
                    self.settings.atlas_vector_index,
                    len(candidates),
                    max(candidate["score"] for candidate in candidates),
                    self.settings.retrieval_min_score,
                )
            else:
                logger.warning(
                    "Atlas retrieval found no vector candidates: tenant_id=%s index=%s; "
                    "check that documents exist for this tenant and the vector index is ready",
                    tenant_id,
                    self.settings.atlas_vector_index,
                )
        return matches
