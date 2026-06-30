# Deployment Strategy

## Overview

The procurement RAG system is deployed alongside ERPNext, with access restricted to specific roles (`Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`). Two deployment options are available — choose based on your ERPNext hosting type.

| | Option A: Frappe App | Option B: Standalone + OAuth2 |
|---|---|---|
| **ERPNext hosting** | Self-hosted bench, or Frappe Cloud (paid plan) | Any — cloud, SaaS, self-hosted |
| **User experience** | Native ERPNext desk — no separate login | Separate URL, "Login with ERPNext" button |
| **Role enforcement** | `frappe.get_roles()` — live on every request | JWT with 8-hour expiry window |
| **Custom app required** | Yes | No |
| **Best for** | Teams who live in the ERPNext desk all day | ERPNext hosted where bench access is unavailable |

---

## Option A: Frappe Custom App (Native ERPNext Desk Integration)

### How It Works

A lightweight Frappe app (`procurement_rag`) is installed on the ERPNext site. It adds a **Workspace shortcut and Desk Page** visible only to the allowed roles — controlled via Frappe's standard Role Permissions Manager. The page contains a chat widget that calls a **whitelisted Python method** server-side, which enforces the role check and proxies the query to the FastAPI backend over the private network.

The FastAPI RAG backend still runs as a separate Docker Compose service. The Frappe app is only the UI and auth layer.

### ERPNext Setup

1. `bench get-app https://github.com/your-org/procurement_rag`
2. `bench --site erp.example.com install-app procurement_rag`
3. `bench --site erp.example.com migrate`
4. In ERPNext desk: **Role Permissions Manager** → set Page `Procurement Intelligence` visible to `Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`

> On **Frappe Cloud**: custom app installation is available on Business / Unlimited plans. On ERPNext SaaS (erpnext.com), use Option B instead.

### Auth Flow

1. User is already logged into ERPNext desk — no separate login required
2. User opens the "Procurement Intelligence" Workspace item (hidden from all other roles)
3. The Desk Page calls `frappe.call('procurement_rag.api.query', {question: ...})`
4. The whitelisted method runs inside the bench process; `frappe.session.user` is already resolved
5. Method checks allowed roles: `if not any(r in frappe.get_roles() for r in ALLOWED_ROLES): raise frappe.PermissionError`
6. If authorized, method calls `http://127.0.0.1:8000/query` with a short-lived HMAC token (`HMAC-SHA256(username + timestamp, INTERNAL_TOKEN_SECRET)`, 30s TTL)
7. FastAPI validates the HMAC and timestamp window, then runs the RAG pipeline
8. Result is returned through the whitelisted method to the chat widget

### New Code Modules

**Frappe app (separate repository):**
```
apps/procurement_rag/
├── procurement_rag/
│   ├── hooks.py                    # app metadata, workspace registration
│   ├── api.py                      # @frappe.whitelist() def query(question, filters)
│   │                               #   → role check → HMAC token → POST 127.0.0.1:8000/query
│   └── public/js/
│       └── procurement_chat.js     # chat widget embedded in the Desk Page
└── setup.py
```

**Inside `procurement-rag` project:**
```
api/
└── auth/
    └── internal_token.py           # validate_hmac_token() — FastAPI dependency for /query
```

### New Environment Variables

```bash
# Shared between bench site_config.json and Docker Compose .env
INTERNAL_TOKEN_SECRET=<random-256-bit-hex>
INTERNAL_TOKEN_TTL_SECONDS=30
```

### Infrastructure Topology

```
Internet (443)
    └── ERPNext nginx (bench-managed)
          └── erp.example.com → ERPNext desk
                  └── [loopback: 127.0.0.1:8000]
                        └── FastAPI (Docker, loopback-bound)
                              [Docker network: rag_internal]
                              ├── Qdrant    (private — no public port)
                              └── Langfuse  (private — no public port)
                                    └── Postgres (private)
```

**`docker-compose.yml` for Option A:**
```yaml
services:
  app:
    ports:
      - "127.0.0.1:8000:8000"   # loopback only — bench can reach it; internet cannot
```

ERPNext webhook URL (internal — no TLS needed):
```
http://127.0.0.1:8000/webhook/erpnext
```

### Verification

1. `docker compose up -d` — FastAPI starts, bound to `127.0.0.1:8000`
2. Log into ERPNext desk as `Purchase Manager` — "Procurement Intelligence" Workspace item is visible
3. Log in as a user with no allowed role — Workspace item is hidden; direct `frappe.call` raises `PermissionError`
4. Submit a query from the desk page — answer and sources returned
5. `curl http://server-ip:8000/query` from outside — connection refused (loopback binding)

---

## Option B: Standalone App with ERPNext OAuth2 + JWT Sessions

### How It Works

The app is deployed as a separate service. Users log in with their **normal ERPNext username and password** via OAuth2 Authorization Code flow with PKCE — the RAG app never handles the password. After login, FastAPI mints a JWT for the session.

Works with any ERPNext hosting. No custom app installation required.

### One-Time ERPNext Setup

OAuth2 is built into Frappe — no custom app needed:

1. Go to ERPNext desk → **Integrations → OAuth Client**
2. Create a new OAuth2 client
3. Set **Redirect URI** to `https://procurement-rag.example.com/auth/callback`
4. Note the generated `client_id` and `client_secret`

### Auth Flow

1. User navigates to `https://procurement-rag.example.com` — sees a **"Login with ERPNext"** button
2. Clicking redirects to:
   ```
   {ERPNEXT_URL}/api/method/frappe.integrations.oauth2.authorize
     ?client_id=...&redirect_uri=...&response_type=code&code_challenge=...&scope=openid+all
   ```
3. User logs in on the ERPNext site — the RAG app never sees the password
4. ERPNext redirects back to `https://procurement-rag.example.com/auth/callback?code=...`
5. FastAPI exchanges the code for an access token via `POST {ERPNEXT_URL}/api/method/frappe.integrations.oauth2.get_token`
6. FastAPI fetches the user's roles via `GET {ERPNEXT_URL}/api/resource/User/{username}?fields=["name","roles"]`
7. If no allowed roles → `403 Forbidden`
8. If authorized, FastAPI mints a signed **JWT** (`sub=username`, `roles=[...]`, `exp=now+8h`) and returns it
9. Streamlit stores the JWT in `st.session_state`; every `POST /query` carries `Authorization: Bearer <jwt>`
10. FastAPI validates the JWT on each request — no ERPNext round-trip on queries

Role changes in ERPNext take effect at next login (8-hour JWT expiry).

### Allowed Roles

| ERPNext Role | Access |
|---|---|
| `Purchase Manager` | Yes |
| `Purchase User` | Yes |
| `Accounts User` | Yes |
| `System Manager` | Yes (admin/testing) |
| All other roles | No |

### New Code Modules

```
api/
├── auth/
│   ├── oauth2.py           # build_authorize_url(), exchange_code_for_token(), fetch_user_roles()
│   ├── pkce.py             # generate_code_verifier(), generate_code_challenge()
│   ├── jwt_handler.py      # mint_token(), decode_token()
│   └── dependencies.py     # FastAPI Depends(): get_current_user, require_allowed_role
├── routers/
│   └── auth.py             # GET /auth/login (redirect), GET /auth/callback (token exchange + role check)

frontend/
├── auth_ui.py              # "Login with ERPNext" button, logout, OAuth redirect handling
└── app.py                  # gate main chat UI behind st.session_state.jwt
```

Key additions to `api/main.py`: mount `auth_router`; apply `require_allowed_role` dependency to `POST /query`.

### New Environment Variables

```bash
# OAuth2 client — created in ERPNext desk → Integrations → OAuth Client
ERPNEXT_OAUTH_CLIENT_ID=<from ERPNext>
ERPNEXT_OAUTH_CLIENT_SECRET=<from ERPNext>
OAUTH_REDIRECT_URI=https://procurement-rag.example.com/auth/callback

# JWT session
JWT_SECRET=<random 256-bit hex>
JWT_EXPIRY_HOURS=8

# Role gate
ALLOWED_ROLES=Purchase Manager,Purchase User,Accounts User,System Manager
```

### Infrastructure Topology

```
Internet (443/80)
    └── Nginx
          ├── procurement-rag.example.com     → Streamlit  (internal: 8501)
          └── api.procurement-rag.example.com → FastAPI    (internal: 8000)
                  [Docker network: rag_internal — no external routing]
                  ├── Qdrant    (6333, private)
                  └── Langfuse  (3000, private)
                        └── Postgres (5432, private)

ERPNext cloud instance
    └── Called by FastAPI during /auth/callback (token exchange + role fetch)
    └── Sends webhooks to https://api.procurement-rag.example.com/webhook/erpnext
```

**`docker-compose.yml` for Option B:**
```yaml
networks:
  rag_internal:
    driver: bridge
    internal: true        # containers reach each other; unreachable from outside

services:
  app:                    # FastAPI — no ports: mapping
    networks: [rag_internal]

  frontend:               # Streamlit — no ports: mapping
    networks: [rag_internal]

  qdrant:
    networks: [rag_internal]

  langfuse:
    networks: [rag_internal]

  postgres:
    networks: [rag_internal]

  nginx:
    ports: ["80:80", "443:443"]
    networks: [rag_internal]  # sole bridge to the outside world
```

**Nginx configuration notes:**
- `proxy_read_timeout 120s` on the FastAPI location — LLM calls take 20–30s
- `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for`
- TLS via certbot / Let's Encrypt
- Streamlit requires WebSocket: `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`

**ERPNext webhook URL:**
```
https://api.procurement-rag.example.com/webhook/erpnext
```

### Verification

1. `docker compose up -d` — all services start; nginx on 443
2. `https://procurement-rag.example.com` — "Login with ERPNext" button visible
3. Click → redirected to ERPNext login → log in as `Purchase Manager` → redirected back to chat UI
4. Repeat with a user outside allowed roles → `403 Access Denied`
5. Submit a procurement query → answer with source citations returned
6. Set `JWT_EXPIRY_HOURS=0` temporarily → next request redirects back to login
7. Trigger ERPNext webhook → document re-indexed; updated answer returned
8. `POST /query` with no `Authorization` header → `401 Unauthorized`
9. Direct TCP to `server-ip:6333` (Qdrant) → connection refused (no public port)

---

## ERPNext Webhook Setup (both options)

This is a **one-time ERPNext configuration** required for incremental re-indexing. It applies regardless of which auth option you chose. Webhooks fire when documents are submitted or updated; the handler at `POST /webhook/erpnext` verifies the HMAC signature, fetches the full document from ERPNext, and re-indexes it in Qdrant.

### Required webhook records

Create the following Webhook records in ERPNext desk (or via the REST API). For each one:
- **Request URL**: `http://127.0.0.1:8000/webhook/erpnext` (Option A / local) or `https://api.procurement-rag.example.com/webhook/erpnext` (Option B / prod)
- **Request Method**: POST
- **Request Structure**: JSON
- **Webhook Secret**: must match `WEBHOOK_SECRET` in your `.env`
- **Webhook Data** (two rows — maps Frappe's `name` field to the key our handler reads):

  | Fieldname | Key      |
  |-----------|----------|
  | `doctype` | `doctype`|
  | `name`    | `docname`|

| Webhook Name              | Doctype           | Event                    | Why                                              |
|---------------------------|-------------------|--------------------------|--------------------------------------------------|
| `po-on-submit`            | Purchase Order    | `on_submit`              | Index POs when first submitted                   |
| `po-on-update-submitted`  | Purchase Order    | `on_update_after_submit` | Re-index if allowed fields change on a live PO   |
| `po-on-cancel`            | Purchase Order    | `on_cancel`              | Re-index with status=Cancelled when PO cancelled |
| `contract-on-submit`      | Contract          | `on_submit`              | Index contracts when submitted                   |
| `contract-on-update`      | Contract          | `on_update`              | Re-index contracts saved/modified                |
| `scorecard-on-update`     | Supplier Scorecard| `on_update`              | Scorecards are not submittable; update only      |

> **Note:** `on_submit` is not valid for Supplier Scorecard — it is not a submittable doctype in ERPNext.

### Via REST API (scripted setup)

```python
import urllib.request, json

BASE = "http://127.0.0.1:8005"   # your ERPNext URL
AUTH = "token <api_key>:<api_secret>"
SECRET = "<your WEBHOOK_SECRET>"
URL = "http://127.0.0.1:8000/webhook/erpnext"

WEBHOOKS = [
    ("po-on-submit",           "Purchase Order",     "on_submit"),
    ("po-on-update-submitted", "Purchase Order",     "on_update_after_submit"),
    ("po-on-cancel",           "Purchase Order",     "on_cancel"),
    ("contract-on-submit",     "Contract",           "on_submit"),
    ("contract-on-update",     "Contract",           "on_update"),
    ("scorecard-on-update",    "Supplier Scorecard", "on_update"),
]

DATA_FIELDS = [
    {"doctype": "Webhook Data", "fieldname": "doctype", "key": "doctype"},
    {"doctype": "Webhook Data", "fieldname": "name",    "key": "docname"},
]

for wname, doctype, event in WEBHOOKS:
    payload = json.dumps({
        "doctype": "Webhook", "name": wname,
        "webhook_doctype": doctype, "webhook_docevent": event,
        "request_url": URL, "request_method": "POST",
        "request_structure": "JSON", "enabled": 1,
        "webhook_secret": SECRET, "webhook_data": DATA_FIELDS,
    }).encode()
    req = urllib.request.Request(f"{BASE}/api/resource/Webhook", data=payload,
        headers={"Authorization": AUTH, "Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req)
    print(json.loads(resp.read())["data"]["name"])
```

### How modifications are handled

The webhook handler (`ingestion/webhook_handler.py`) is event-agnostic: for any event it calls `vector_store.delete_by_docname(docname)` then re-indexes the full document. This means:

- **New PO submitted** → `on_submit` fires → indexed fresh
- **PO amended** → amendment creates a new PO (new `docname`) → `on_submit` fires again on the amendment
- **PO field updated after submit** → `on_update_after_submit` fires → old vectors deleted, new ones upserted
- **Contract updated** → `on_update` fires → re-indexed with latest content
- **Scorecard updated** → `on_update` fires → re-indexed

---

## Implementation Sequence (both options)

Inserts into `IMPLEMENTATION_PLAN.md` at:

| Step | Option A | Option B |
|---|---|---|
| **Step 12** (FastAPI) | Add `api/auth/internal_token.py`; bind to `127.0.0.1:8000` | Add `api/auth/` OAuth2 modules + `routers/auth.py` |
| **Step 13** (Streamlit) | Build Frappe app (`hooks.py`, `api.py`, `procurement_chat.js`) | Add `auth_ui.py`; gate chat UI behind session JWT |
| **Step 15** (Docker Compose) | Bind `app` to `127.0.0.1:8000`; all others on `rag_internal` | `rag_internal` network; nginx as sole public gateway |
