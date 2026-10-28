from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pymongo import MongoClient, ReplaceOne

from rag_gateway.config import Settings


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
            {"$match": {"score": {"$gte": self.settings.retrieval_min_score}}},
        ]
        return list(self.collection.aggregate(pipeline))
