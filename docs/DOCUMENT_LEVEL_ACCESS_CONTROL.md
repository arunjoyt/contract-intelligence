# Document-Level Access Control (proposed, not implemented)

**Status:** Not implemented. Revisit if #60's accepted-risk conditions break.

## Context

Today, authorization is a single yes/no gate: does the authenticated user hold one of the four
allowed roles (`Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`). Once past
that gate, every user sees the same answers and sources, drawn from the **entire** indexed corpus —
there is no per-document, per-company, or per-department filtering. See `docs/ARCHITECTURE.md`
§ "Known Limitation — access control is role-level, not document-level" and issue #60 for the full
decision record (accepted conditionally; the assumption is that all four roles are already meant to
see the same procurement data in the current ERPNext setup).

This document sketches the solution for when that assumption breaks and #60 is reopened.

## Proposed solution

Four pieces, roughly medium complexity (~2-3 days), mostly plumbing through mechanisms that already
exist for the current optional query filters (supplier/doctype/status):

### 1. Fetch the user's ERPNext permission scope at login

`api/auth/oauth2.py:fetch_user_roles()` already calls the ERPNext REST API with the admin key to
pull `User.roles` at login time. Add a parallel call to:

```
GET /api/resource/User Permission?filters=[["user","=","<docname>"]]
```

`User Permission` is ERPNext's own doctype for scoping a user to specific values of a linked
doctype (e.g. specific `Company` records). Same shape of call, same admin key, one more
round-trip. An empty/absent result means "unrestricted" — matching ERPNext's own convention.

### 2. Embed it in the JWT

`api/auth/jwt_handler.py:mint_token()` already embeds `roles`; add an `allowed_companies: list[str]`
claim alongside it. Trivial, one-line addition.

### 3. Thread it through the query pipeline as a filter

`api/main.py` → `QueryPipeline.run()` → `HybridSearch.search()` → `VectorStore.search()` already
pass a `filter_conditions` dict end-to-end for the existing supplier/doctype/status filters. Add
`company: allowed_companies` to that dict on the way in — a new parameter threaded through the same
3-4 call sites, not new architecture.

### 4. Fix BM25 to hard-filter, not just narrow

`retrieval/hybrid_search.py` states outright that "BM25 results are unfiltered by design" — today's
filters are advisory narrowing on the Qdrant leg only, since `rank_bm25.BM25Okapi` has no native
filter support (unlike Qdrant, which filters as part of the vector query itself). That's fine for
optional relevance filters but not for a security boundary: a restricted document could still surface
via the lexical leg and get fused into the top-20 regardless of the company filter.

This is **not** a reason BM25 itself is unusable here — `BM25Okapi.get_scores()` already returns a
score per corpus document (`hybrid_search.py:69`); the fix is a cheap post-filter (drop candidates
whose `company` isn't in `allowed_companies`) over already-computed scores, before RRF fusion. No
BM25-internals change, no reindexing required. This only becomes a real constraint at large scale
(>100k docs, per `docs/ARCHITECTURE.md`'s BM25 section), where the filter would want to be pushed
into the search itself (e.g. Qdrant sparse vectors) rather than post-filtered in Python.

## Open product question, not just a code question

What's the actual scoping dimension, and does the data support it? `company` is already captured in
the payload for Purchase Order/Invoice/Contract, but `Terms and Conditions` and `Supplier Scorecard`
explicitly set `company: None` today (those ERPNext doctypes have no company field) — a naive
"filter by company" leaves those two doctypes either always-visible to everyone or needs a different
rule.

This needs an org-specific decision — and ERPNext data with real `User Permission` records
configured to test against — before implementation begins. At the time of writing, the connected
ERPNext instance has no such restrictions configured, so there's nothing to validate the fix against
yet.
