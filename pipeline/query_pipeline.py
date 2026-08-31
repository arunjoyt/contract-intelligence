"""Query orchestration pipeline.

Wraps the full request-response cycle for a single user question:
  1. QueryRewriter.rewrite_text()  → rewritten text (chat only)
  2. QueryRewriter.embed()         → query vector (embedding only)
  3. _extract_filters()            → metadata filter dict (heuristic)
  4. HybridSearch.search()         → top-20 candidate chunks (dense leg reuses the
                                     vector from step 2 -- no second embed call)
  5. Reranker.rerank()             → top-5 chunks
  6. GPT-4o generation             → answer + [docname] citations
  7. _parse_sources()              → structured SourceDoc list

Every step is wrapped in a Langfuse child observation when a Langfuse instance is
provided -- ``rewrite``, ``embed_query`` and ``generate`` as ``generation``s (so
their token cost is tracked), the rest as plain spans.  ``rewrite`` is skipped
for ``QUERY_REWRITE_STRATEGY=none`` (no LLM call).  Passing ``langfuse=None``
disables tracing entirely so the pipeline works without Langfuse credentials.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from config import EMBEDDING_MODEL, OPENAI_MODEL, REWRITE_MODEL
from pipeline.constants import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_META_FIELDS,
    GENERATION_MAX_TOKENS,
    METADATA_FILTER_DOCTYPE_KEYWORDS,
    METADATA_FILTER_STATUS_KEYWORDS,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
)
from pipeline.query_rewriter import QueryRewriter
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker

if TYPE_CHECKING:
    from langfuse import Langfuse


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
            rewritten, query_vector = self._rewrite_and_embed(trace, question)

            extracted = self._span(trace, "filter_extraction", lambda: _extract_filters(question))
            merged_dict = {**(extracted or {}), **(filters or {})}
            merged = merged_dict if merged_dict else None

            candidates = self._span(
                trace,
                "hybrid_search",
                lambda: self._hybrid_search.search(
                    rewritten, merged, top_k=RETRIEVAL_TOP_K, query_vector=query_vector
                ),
                summarize=_docnames,
            )

            top_chunks = self._span(
                trace,
                "rerank",
                lambda: self._reranker.rerank(question, candidates, top_n=RERANK_TOP_N),
                summarize=_docnames,
            )

            context = _build_context(top_chunks)
            if trace:
                gen_span = trace.generation(name="generate", model=OPENAI_MODEL)
                try:
                    answer, usage = self._generate(question, context)
                    gen_span.end(output=answer, usage=usage)
                except Exception:
                    gen_span.end(level="ERROR")
                    raise
            else:
                answer, _usage = self._generate(question, context)

            sources = _parse_sources(answer, top_chunks)

            if trace:
                trace.update(output={"answer": answer, "source_count": len(sources)})

            return {"answer": answer, "sources": sources}
        except Exception:
            if trace:
                trace.update(level="ERROR")
            raise

    def _rewrite_and_embed(self, trace: Any, question: str) -> tuple[str, list[float]]:
        """Rewrite the question (chat) and embed the result, as two separate
        Langfuse `generation`s.

        - ``rewrite`` (REWRITE_MODEL) -- skipped entirely for
          ``QUERY_REWRITE_STRATEGY=none``, which makes no LLM call.
        - ``embed_query`` (EMBEDDING_MODEL) -- always runs. Previously this embed
          happened inside ``QueryRewriter.rewrite`` and its latency/cost was
          bundled into the ``rewrite`` span; now it is attributable on its own,
          and the vector is threaded into ``HybridSearch.search`` so the dense
          leg does not embed the same text a second time (#138).

        Recording both as `generation`s keeps their token cost in the trace total
        (#130) -- the query embed rounds to ~$0 but completes the picture.
        """
        rewritten = self._traced_rewrite_text(trace, question)
        query_vector = self._traced_embed(trace, rewritten)
        return rewritten, query_vector

    def _traced_rewrite_text(self, trace: Any, question: str) -> str:
        if trace is None or self._rewriter.strategy == "none":
            return self._rewriter.rewrite_text(question)

        rewrite_span = trace.generation(name="rewrite", model=REWRITE_MODEL)
        try:
            rewritten = self._rewriter.rewrite_text(question)
            rewrite_span.end(output=rewritten, usage=self._rewriter.last_usage)
            return rewritten
        except Exception:
            rewrite_span.end(level="ERROR")
            raise

    def _traced_embed(self, trace: Any, text: str) -> list[float]:
        if trace is None:
            return self._rewriter.embed(text)

        embed_span = trace.generation(name="embed_query", model=EMBEDDING_MODEL)
        try:
            vector = self._rewriter.embed(text)
            embed_span.end(usage=self._rewriter.last_embed_usage)
            return vector
        except Exception:
            embed_span.end(level="ERROR")
            raise

    def _generate(self, question: str, context: str) -> tuple[str, dict[str, int]]:
        response = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        usage = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }
        return response.choices[0].message.content or "", usage

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

    Detects doctype and status keywords from the (per-client configurable)
    vocabulary in ``pipeline.constants``.  The pipeline merges this dict with
    any explicit sidebar filters from the frontend (sidebar wins on conflicts).
    """
    lower = question.lower()
    filters: dict[str, Any] = {}

    for doctype, keywords in METADATA_FILTER_DOCTYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            filters["source_doctype"] = doctype
            break

    for keyword, status_value in METADATA_FILTER_STATUS_KEYWORDS.items():
        if keyword in lower:
            filters["status"] = status_value
            break

    return filters


def _docnames(chunks: list[dict]) -> list[str]:
    """Docnames only, in rank order -- enough to sanity-check retrieval without
    duplicating chunk text into Langfuse's trace storage."""
    return [c.get("docname", "") for c in chunks]


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
            f"{key}: {value}" for key in CONTEXT_META_FIELDS if (value := chunk.get(key))
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
