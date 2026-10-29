# Synthetic demo corpus

> Atlas RAG Gateway is an independent personal project and is not affiliated with, sponsored by, or endorsed by MongoDB, Inc.

These fictional documents are provided for local ingestion and retrieval demos.
They contain no real infrastructure inventory, credentials, or personal records.

From the repository root, ingest them with:

```bash
python -m pip install -e ".[loader]"
python scripts/document_loader.py ./sample-docs --tenant default
```

The folder includes Markdown (`.md`), Mermaid (`.mmd`), PDF, Excel (`.xlsx`),
Word (`.docx`), PowerPoint (`.pptx`), and JPEG (`.jpg`) examples. The PDF and
JPEG examples are intentionally rendered as readable text and require a
configured Ollama image/OCR model. Set `OLLAMA_BASE_URL` and
`DOCUMENT_PARSER_MODEL` to a compatible local Ollama service and installed model.
The office formats use local text/table extraction.

The sample facts describe a fictional community greenhouse. They include
deliberately simple details for testing retrieval, citations, and questions with
no matching evidence. The basic `scripts/ingest_folder.py` only handles `.txt`
and `.md`; use the document-loader CLI above to ingest every included format.
