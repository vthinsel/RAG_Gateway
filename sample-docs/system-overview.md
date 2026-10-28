# Example RAG System Overview

This fictional design illustrates a basic retrieval-augmented generation flow.

1. A client submits a question to an authenticated gateway.
2. The gateway creates an embedding for the question.
3. Vector search retrieves relevant document chunks for the caller's tenant.
4. The gateway adds retrieved evidence to the prompt sent to a language model.
5. The response includes source metadata so a client can inspect the evidence.

The embedding service and vector database may be hosted services. Before using
real documents, confirm where query text, source text, embeddings, and logs are
processed and stored.
