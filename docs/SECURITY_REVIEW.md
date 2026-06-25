# Security Status

**Last reviewed:** 2026-06-25
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

| Item | Rationale |
|------|-----------|
| `/query` has no authentication | By design — auth is planned (Steps 12/13 of `IMPLEMENTATION_PLAN.md`) |
| `lf-pk-local` / `lf-sk-local` hardcoded in `docker-compose.yml` | Local dev keys only; production must override via `.env` |

---

## Areas checked

- Webhook HMAC verification
- Admin endpoint (`/ingest/full`) secret check
- Qdrant filter construction (injection)
- ERPNext client URL construction (SSRF)
- Streamlit frontend output (XSS)
- Subprocess / eval / pickle / unsafe YAML usage
- Hardcoded secrets in tracked files
- JWT / session handling
