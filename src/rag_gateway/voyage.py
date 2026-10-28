import asyncio
import random
from typing import Any

import httpx

from rag_gateway.config import Settings

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_BATCH_TOKENS = 100_000


def _token_batches(
    texts: list[str], max_tokens: int = _MAX_BATCH_TOKENS, max_items: int = 128
):
    batch: list[str] = []
    budget = 0
    for text in texts:
        estimated_tokens = len(text) // 4 + 1
        if estimated_tokens > max_tokens:
            raise ValueError("An embedding input exceeds the estimated token batch limit")
        if batch and (len(batch) >= max_items or budget + estimated_tokens > max_tokens):
            yield batch
            batch, budget = [], 0
        batch.append(text)
        budget += estimated_tokens
    if batch:
        yield batch


class VoyageEmbedder:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.mongodb_api_key:
            raise RuntimeError("MONGODB_API_KEY is not configured")

        vectors: list[list[float]] = []
        batches = _token_batches(texts, max_items=self.settings.embedding_batch_size)
        for batch in batches:
            response = await self._post_with_retry(
                {
                    "model": self.settings.voyage_embedding_model,
                    "input": batch,
                    "input_type": input_type,
                }
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            batch_vectors = [item["embedding"] for item in data]
            if len(batch_vectors) != len(batch):
                raise RuntimeError("Embedding API returned a different number of vectors than inputs")
            if any(len(vector) != self.settings.embedding_dimensions for vector in batch_vectors):
                raise RuntimeError(
                    "Embedding dimensions do not match EMBEDDING_DIMENSIONS; "
                    "align the model and Atlas Vector Search index"
                )
            vectors.extend(batch_vectors)
        return vectors

    async def _post_with_retry(self, payload: dict[str, Any], attempts: int = 5) -> httpx.Response:
        for attempt in range(attempts):
            try:
                response = await self.client.post(
                    f"{self.settings.mongodb_api_model_base_url}/embeddings",
                    headers={"Authorization": "Bearer " + self.settings.mongodb_api_key},
                    json=payload,
                )
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(min(2**attempt, 30) + random.uniform(0, 0.5))
                continue
            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt == attempts - 1:
                return response
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(0.0, float(retry_after)) if retry_after else min(2**attempt, 30)
            except ValueError:
                delay = min(2**attempt, 30)
            await asyncio.sleep(delay + random.uniform(0, 0.5))
        raise RuntimeError("Voyage retry loop ended unexpectedly")
