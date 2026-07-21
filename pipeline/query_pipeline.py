"""Query orchestration pipeline.

Wraps the full request-response cycle for a single user question:
  1. QueryRewriter.rewrite()       → rewritten text + query vector
  2. _extract_filters()            → metadata filter dict (heuristic)
  3. HybridSearch.search()         → top-20 candidate chunks
  4. Reranker.rerank()             → top-5 chunks
  5. GPT-4o generation             → answer + [docname] citations
  6. _parse_sources()              → structured SourceDoc list

Every step is wrapped in a Langfuse child span when a Langfuse instance is
provided.  Passing ``langfuse=None`` disables tracing entirely so the pipeline
works without Langfuse credentials.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from config import OPENAI_MODEL
from pipeline.query_rewriter import QueryRewriter
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker

if TYPE_CHECKING:
    from langfuse import Langfuse

_ANSWER_SYSTEM = """\
You are a contract analyst assistant. Answer the user's question using ONLY \
the context below.

Rules:
- Cite every claim with [docname] immediately after the relevant sentence.
- The context contains exact field values (status codes, dates, etc.) from \
contract records. You may use ordinary language understanding to relate the \
user's wording to those exact values -- e.g. "signed" may match "Unsigned" as its \
negation; "terminated"/"ended" may match a status like "Cancelled". Interpreting \
the plain meaning of a value that IS present in the context is not "outside knowledge."
- Do not invent facts, entities, values, or numbers that do not appear in the context.
- If the context genuinely contains nothing relevant to the question, respond with \
exactly: "I could not find relevant information in the contract documents."

Context:
{context}
"""

# Keywords used for heuristic doctype detection in the question text.
_DOCTYPE_KEYWORDS: dict[str, list[str]] = {
    "Contract": ["contract"],
    "Terms and Conditions": ["terms and conditions"],
}

_STATUS_KEYWORDS: list[str] = [
    "cancelled",
    "active",
    "unsigned",
]


@dataclass
class SourceDoc:
    docname: str
    source_doctype: str
    supplier: str
    chunk_index: int


class QueryPipeline:
    def __init__(
        self,
        rewriter: QueryRewriter,
        hybrid_search: HybridSearch,
        reranker: Reranker,
        api_key: str | None = None,
        langfuse: Langfuse | None = None,
    ) -> None:
        self._rewriter = rewriter
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._langfuse = langfuse

    def run(self, question: str, filters: dict[str, Any] | None = None) -> dict:
        """Run the full RAG pipeline for a user question.

        Returns ``{"answer": str, "sources": list[SourceDoc]}``.
        ``filters`` from the caller (e.g. frontend sidebar) are merged with
        any filters extracted heuristically from the question text; explicit
        caller filters take precedence on key conflicts.
        """
        trace = (
            self._langfuse.trace(name="query", input={"question": question})
            if self._langfuse
            else None
        )

        try:
            rewritten, _vector = self._span(
                trace, "rewrite", lambda: self._rewriter.rewrite(question)
            )

            extracted = self._span(trace, "filter_extraction", lambda: _extract_filters(question))
            merged_dict = {**(extracted or {}), **(filters or {})}
            merged = merged_dict if merged_dict else None

            candidates = self._span(
                trace,
                "hybrid_search",
                lambda: self._hybrid_search.search(rewritten, merged, top_k=20),
                summarize=_docnames,
            )

            top_chunks = self._span(
                trace,
                "rerank",
                lambda: self._reranker.rerank(question, candidates, top_n=5),
                summarize=_docnames,
            )

            context = _build_context(top_chunks)
            if trace:
                gen_span = trace.span(name="generate")
                try:
                    answer = self._generate(question, context)
                    gen_span.end(output=answer)
                except Exception:
                    gen_span.end(level="ERROR")
                    raise
            else:
                answer = self._generate(question, context)

            sources = _parse_sources(answer, top_chunks)

            if trace:
                trace.update(output={"answer": answer, "source_count": len(sources)})

            return {"answer": answer, "sources": sources}
        except Exception:
            if trace:
                trace.update(level="ERROR")
            raise

    def _generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM.format(context=context)},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _span(trace: Any, name: str, fn, summarize=None):
        """Execute ``fn`` inside a Langfuse span if tracing is active.

        ``summarize``, if given, converts the result into a small span output
        (e.g. docnames only) instead of dumping the full result -- avoids
        duplicating chunk text/embeddings into Langfuse's trace storage.
        """
        if trace is None:
            return fn()
        span = trace.span(name=name)
        try:
            result = fn()
            span.end(output=summarize(result) if summarize else None)
            return result
        except Exception:
            span.end(level="ERROR")
            raise


def _extract_filters(question: str) -> dict[str, Any]:
    """Heuristic metadata filter extraction from the question text.

    Detects doctype and status keywords.  The pipeline merges this dict with
    any explicit sidebar filters from the frontend (sidebar wins on conflicts).
    """
    lower = question.lower()
    filters: dict[str, Any] = {}

    for doctype, keywords in _DOCTYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            filters["source_doctype"] = doctype
            break

    for status in _STATUS_KEYWORDS:
        if status in lower:
            filters["status"] = status.capitalize()
            break

    return filters


def _docnames(chunks: list[dict]) -> list[str]:
    """Docnames only, in rank order -- enough to sanity-check retrieval without
    duplicating chunk text into Langfuse's trace storage."""
    return [c.get("docname", "") for c in chunks]


_CONTEXT_META_FIELDS = (
    "source_doctype",
    "supplier",
    "supplier_group",
    "status",
    "company",
    "start_date",
    "end_date",
    "linked_doctype",
    "linked_docname",
)


def _build_context(chunks: list[dict]) -> str:
    """Serialize top chunks into a context block for the LLM prompt.

    Surfaces every metadata field already on the chunk payload, not just
    ``supplier`` -- some doctypes (e.g. Contract) carry fields like ``status``
    that never get baked into the chunk's own ``text`` by its serializer, so
    without this they'd be invisible to generation even though retrieval has
    them.
    """
    parts: list[str] = []
    for chunk in chunks:
        docname = chunk.get("docname", "unknown")
        meta_bits = [
            f"{key}: {value}" for key in _CONTEXT_META_FIELDS if (value := chunk.get(key))
        ]
        meta_str = f" ({'; '.join(meta_bits)})" if meta_bits else ""
        text = chunk.get("text", "")
        parts.append(f"[{docname}]{meta_str}:\n{text}")
    return "\n\n---\n\n".join(parts)


def _parse_sources(answer: str, top_chunks: list[dict]) -> list[SourceDoc]:
    """Extract cited docnames from the answer and map them to SourceDoc objects.

    Only chunks whose docname appears as a ``[docname]`` citation in the answer
    are included; the first occurrence of each docname is used.
    """
    cited = set(re.findall(r"\[([^\]]+)\]", answer))
    sources: list[SourceDoc] = []
    seen: set[str] = set()
    for chunk in top_chunks:
        docname = chunk.get("docname", "")
        if docname in cited and docname not in seen:
            seen.add(docname)
            sources.append(
                SourceDoc(
                    docname=docname,
                    source_doctype=chunk.get("source_doctype", ""),
                    supplier=chunk.get("supplier", ""),
                    chunk_index=chunk.get("chunk_index", 0),
                )
            )
    return sources
