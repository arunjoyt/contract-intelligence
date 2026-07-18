# Security Status

**Last reviewed:** 2026-06-30
**Method:** Static analysis of all source files (`api/`, `ingestion/`, `retrieval/`, `pipeline/`, `frontend/`, `docker-compose.yml`)

---

## Findings

| ID | Severity | Category | Location | Status |
|----|----------|----------|----------|--------|
| F1 | High | Auth bypass | `ingestion/webhook_handler.py` — `_verify_signature` | Resolved |
| F2 | High | Hardcoded credential | `docker-compose.yml` — `NEXTAUTH_SECRET`, `SALT` | Resolved |
| F3 | Medium | Hardcoded credential | `docker-compose.yml` — `LANGFUSE_INIT_USER_PASSWORD` | Resolved |
| F4 | High | Path traversal / SSRF | `ingestion/webhook_handler.py`, `ingestion/erpnext_client.py` | False positive |

### F1 — Webhook authentication bypass via unconfigured secret

When `WEBHOOK_SECRET` was not set, the HMAC was computed with an empty key, allowing an attacker to forge a valid webhook signature by computing `HMAC-SHA256(key=b"", msg=body)` independently.

### F2 — Hardcoded Langfuse JWT signing secret

`NEXTAUTH_SECRET` (signs Langfuse session JWTs) and `SALT` (hashes API keys) were committed as literal `changeme-*` strings, enabling session token forgery on any deployment using the defaults.

### F3 — Hardcoded Langfuse admin password

`LANGFUSE_INIT_USER_PASSWORD` was committed as `changeme`. The Langfuse port is bound to all host interfaces, so any LAN-reachable attacker could log in and read all RAG query traces.

### F4 — Path traversal via `docname` in webhook payload *(false positive)*

`docname` is passed unsanitised to the ERPNext API URL. Ruled not exploitable: the HMAC check (F1) blocks unauthenticated access before any payload field is read, and the attacker controls only the URL path — not the host — so this does not meet the SSRF threshold.

---

## Accepted risks

None currently.

## Resolved risks

| Item | Resolution |
|------|------------|
| `/query` had no authentication | Resolved — Option B OAuth2 + JWT auth implemented (PR #32). `POST /query` now requires `Authorization: Bearer <jwt>`; users without allowed roles receive 403. |
| `lf-pk-local` / `lf-sk-local` hardcoded in `docker-compose.yml` | Resolved — `docker-compose.yml` now reads `${LANGFUSE_PUBLIC_KEY}`/`${LANGFUSE_SECRET_KEY}` from `.env`; no literal key values remain in a tracked file. |

---

## Areas checked (as of 2026-06-30 review)

- Webhook HMAC verification
- Admin endpoint (`/ingest/full`) secret check
- Qdrant filter construction (injection)
- ERPNext client URL construction (SSRF)
- Streamlit frontend output (XSS)
- Subprocess / eval / pickle / unsafe YAML usage
- Hardcoded secrets in tracked files
- JWT / session handling

---

## Changes since last full review (spot notes, not a re-review)

The items below landed after 2026-06-30 and touch security-relevant surface, but have not been
through the same systematic static-analysis pass as the findings above — noted here so they aren't
silently missing from this document, not as a certification that they're clean.

- **OAuth callback duplicate-request tolerance** (`api/routers/auth.py`'s `oauth_completed` cache,
  commit `5884a48`) — caches a completed login's redirect (including the minted JWT) for 60s, keyed
  by the OAuth `state` value, so a duplicate `/auth/callback` request (from ERPNext's confirmation
  page double-firing its click handler) replays the same redirect instead of erroring on an
  already-consumed `state`. Assessed as low risk: `state` is a single-use, short-lived (60s) random
  token already transmitted through the user's own browser redirect chain — the cache doesn't create
  a way to mint a *new* token, only to re-serve the same one within a narrow window to a request
  presenting the same `state`.
- **Docker network/topology hardening** (commit `34025bd`) and **loopback-only Qdrant/Langfuse
  ports** (commits `0469871`/`2923f2e`) — Qdrant and Langfuse are now bound to `127.0.0.1` only
  (reachable via SSH tunnel, never the public internet), which further mitigates F3's original
  concern about the Langfuse port being reachable on all host interfaces.
- **`restart: unless-stopped` on all services** (commit `be89ad6`) — availability/ops change, no
  security implication identified.
- **Nginx envsubst domain templating** (commit `de694ea`) — `FRONTEND_DOMAIN`/`API_DOMAIN` are
  deployer-set `.env` values substituted into the nginx config at container start, not
  user-controlled input; no injection surface identified.
