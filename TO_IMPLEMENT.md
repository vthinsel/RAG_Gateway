# Further implementation ideas

These are the remaining improvement ideas not included in the current change.

## Observability and operations

- [x] Add OpenTelemetry traces with request IDs and spans for embedding, Atlas retrieval, reranking, and generation. See [MONITORING.md](MONITORING.md).
- [x] Persist token usage and model/API cost estimates by tenant and model; include embedding and reranking usage where available. See [MONITORING.md](MONITORING.md).
- [ ] Make monitoring useful across restarts and multiple workers, or explicitly constrain deployments to one worker.
- [ ] Add configurable alert thresholds for error rate, latency, and dependency health.
- [ ] Replace JSON strings sent through the Uvicorn error logger with a dedicated structured logger and configurable log level.
- [ ] Add an audit trail for document and tenant deletion operations.
- [ ] Reduce repeated empty-retrieval warning noise through sampling or rate limiting while retaining aggregate counts.

## Answer quality

- [ ] Validate generated citations against the retrieved source set and report citation coverage or invalid citations.
- [ ] Return a deterministic insufficient-evidence response when retrieval yields no usable sources.
- [ ] Extend the evaluation workbench with answer correctness, faithfulness, and citation-quality evaluations.
- [ ] Add user feedback linked to request IDs and make it available for quality analysis.
- [ ] Preserve document headings and titles in chunks to provide better retrieval context.

## Performance and retrieval

- [ ] Add tenant- and configuration-safe query embedding/retrieval caching with invalidation on ingestion.
- [ ] Run the independent vector and lexical hybrid-search branches concurrently and evaluate latency under load.
- [ ] Reuse embeddings for unchanged chunks when a document is re-ingested.
