# Architecture

## System Overview

```
ERPNext
  │  REST API (initial full ingest)
  │  Webhooks (incremental updates)
  ▼
┌─────────────────────────────────┐
│        Ingestion Layer          │
│  erpnext_client → document_     │
│  parser → chunker → embedder    │
└──────────────┬──────────────────┘
               │ upsert (idempotent, by docname+chunk_index)
               ▼
┌──────────────────────────────────┐
│           Qdrant                 │
│  Collection: procurement         │
│  Indexed payload: source_doctype,│
│  supplier, status, start_date,   │
│  end_date                        │
└──────────────┬───────────────────┘
               │
     ┌─────────▼──────────┐
     │   Retrieval Layer   │
     │  BM25 + vector →   │
     │  RRF fusion →      │
     │  Cross-encoder     │
     │  rerank            │
     └─────────┬──────────┘
               │ top-5 chunks + metadata
               ▼
┌──────────────────────────────────┐
│         Pipeline Layer           │
│  QueryRewriter (HyDE/step-back)  │
│  → HybridSearch                  │
│  → Reranker                      │
│  → Prompt builder                │
│  → GPT-4o                        │
│  → Answer + source citations     │
└──────────┬──────────┬────────────┘
           │          │ Langfuse spans
           ▼          ▼
        FastAPI    Langfuse
        /query      (trace)
           │
           ▼
        Streamlit
```

## Query Pipeline — Step by Step

1. **Query rewriting** — HyDE generates a hypothetical answer; its embedding becomes the query vector. This improves recall for abstract questions.
2. **Metadata filter extraction** — simple heuristics (and optionally a small LLM call) parse supplier names, date ranges, and doctype hints from the original question.
3. **Hybrid search** — BM25 (lexical) and Qdrant vector search run in parallel. Results are fused with Reciprocal Rank Fusion (`k=60`), returning 20 candidates. **Important:** metadata filters (supplier, doctype, status) are applied only to the Qdrant vector-search path — BM25 has no metadata support and searches the full corpus. Caller filters therefore narrow vector results but do not guarantee that every result in the fused set matches the filter.
4. **Cross-encoder reranking** — `ms-marco-MiniLM` scores all 20 `(query, chunk)` pairs and returns the top 5.
5. **Generation** — GPT-4o receives the top-5 chunks as context with a structured prompt that requires source citations.
6. **Tracing** — each step is a Langfuse child span; the full trace is linked to the question.

## Document Indexing Strategy

### Structured documents (PO, Invoice, Scorecard)

Serialized to a single natural-language string before embedding. No chunking — the entire doc is one vector. This avoids fragmenting relational fields across chunks.

```
Purchase Order PO-2024-00123 issued to Vendor ABC on 2024-03-15.
Total value: 45,000 USD. Delivery expected by 2024-04-10.
Items: Steel Pipes, Welding Rods. Payment terms: Net 30. Status: Submitted.
```

### Unstructured documents (Contract, T&C, PDFs)

- HTML stripped via BeautifulSoup before chunking
- `RecursiveCharacterTextSplitter`: `chunk_size=512`, `chunk_overlap=64`
- Each chunk stores `chunk_index` and `total_chunks` for reconstruction

### Supplier Metadata Enrichment

`supplier_group` (used in the payload schema and as a filter field) is **not** present on `Purchase
Order` records — it only exists on `Supplier` (and is, separately, already a direct field on
`Purchase Invoice`). For `Purchase Order` and `Contract`, `erpnext_client.py` must fetch/cache
`Supplier` records and join `supplier_group` onto the record before serialization in
`document_parser.py`.

### Qdrant Payload Schema

```json
{
  "text": "...",
  "metadata": {
    "source_doctype": "Contract",
    "docname": "CON-2024-00042",
    "supplier": "Supplier-XYZ",
    "supplier_group": "Hardware",
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
    "status": "Active",
    "company": "My Company",
    "chunk_index": 2,
    "total_chunks": 8
  }
}
```

Point ID is derived as `uuid5(NAMESPACE_DNS, f"{docname}:{chunk_index}")` — deterministic, enabling idempotent upserts.

## Incremental Indexing via Webhooks

ERPNext fires a webhook on `on_submit` and `on_update` for:
- `Purchase Order`
- `Contract`
- `Supplier Scorecard`

The webhook handler:
1. Verifies `X-Frappe-Webhook-Signature` (HMAC-SHA256)
2. Fetches the full document via REST API
3. Calls `vector_store.delete_by_docname(docname)` — removes all old chunks
4. Re-runs the full parse → chunk → embed → upsert pipeline for that document only

This ensures updated contract terms or revised POs are reflected in the next query without a full re-index.

## BM25 Index

The BM25 index is built in memory at API startup by fetching all payload texts from Qdrant. It is rebuilt after each webhook upsert to stay current. For large collections (>100k docs), consider moving to Qdrant's sparse vector support instead.

### Filter behaviour — verified in live testing (2026-06-19)

Metadata filters (supplier, doctype, status) apply **only to the Qdrant vector-search leg** of the hybrid search. BM25 is corpus-wide by design; it cannot apply field conditions. Consequences:

- A supplier filter narrows the Qdrant candidate set to that supplier but BM25 may still surface documents from other suppliers that score highly on lexical overlap.
- The pipeline's `_parse_sources` only includes documents that GPT-4o explicitly cites, so uncited BM25 hits are silently discarded — the practical leakage is lower than the raw candidate count suggests.
- If strict supplier isolation is required (e.g. a multi-tenant deployment), a post-rerank filter on `source.supplier` must be added in `query_pipeline.py` before the generation step.

## Observability

Every query creates a Langfuse trace with the following spans:
- `query_rewrite`
- `metadata_filter_extraction`
- `hybrid_search`
- `rerank`
- `llm_generation`

Inputs, outputs, token counts, and latencies are logged per span. The Langfuse UI is accessible at `http://localhost:3000`.

## Security Notes

- ERPNext API key/secret never leaves the server — the webhook handler only exposes an HMAC-protected endpoint
- The `/ingest/full` endpoint is protected by `X-Admin-Secret` header
- All secrets are loaded from environment variables; `.env` is gitignored
- No user input is interpolated into Qdrant filter expressions directly — filters are constructed programmatically
