# Deployment rehearsal findings

Running log of issues found while exercising the deployment path. Entries here become
GitHub issues when ready. Committed — but check before committing updates.

Context: full green-field rehearsal of `docs/CLIENT_DEPLOYMENT_RUNBOOK.md` on a fresh
EC2 box, tracked in #139.

---

## Filed

### #141 — CLIENT_DEPLOYMENT_RUNBOOK.md fixes (2026-09-01)

<https://github.com/arunjoyt/contract-intelligence/issues/141>

1. Phase 2 is missing the 5th webhook (`contract-on-update-after-submit`); heading says
   "× 4", `docs/DEPLOYMENT.md` correctly says 5. The #96 doc-sync (`4dce21b`) missed the
   runbook.
2. Phase 0 understates the service-user privilege — `api/auth/oauth2.py:fetch_user_roles()`
   reads `User.roles`, which needs System Manager, not a minimal read role. Security note
   ties to #63.
3. Phase 4a references `scripts/sample_ingest.py`, which does not exist (only a sketch).
   Add a real script.
4. Webhook creation should be a committed `scripts/setup_erpnext_webhooks.py`, replacing
   the inline snippet in `docs/DEPLOYMENT.md` § "Via REST API".

Fix branch: after Phase 5. No PR (keep #17 in sync per workflow).

---

## Not yet filed

### App logout does not end the ERPNext SSO session (2026-09-04)

`frontend/auth_ui.py:show_logout_button()` only does `del st.session_state.jwt` — it clears
the app's JWT but never touches the ERPNext session. There is no RP-initiated logout
(no call to an `end_session_endpoint`, no prompt=login on re-auth). Effect: after clicking
Logout, the next "Login with ERPNext" skips the credential screen and lands straight on the
OAuth consent page, silently re-authenticating as the same ERPNext user. On a shared
machine this means a "logged out" user's session is trivially reusable, and it blocks
testing role-based access with a second user without an incognito window.

- Impact: security/UX. Ties loosely to #63 (role enforcement) and #33 (JWT persistence).
- Fix options: (a) after clearing the JWT, redirect the browser to ERPNext's logout
  (`/api/method/logout`) then back to the app; (b) send `prompt=login` on the authorize
  request so re-login always re-prompts; (c) document the limitation in the runbook Phase 5
  ("use an incognito window to test a second role").
- Found while running runbook Phase 5 smoke test "user outside ALLOWED_ROLES -> 403".

### OAuth Client `allowed_roles` defaults to `Desk User` and gates `authorize` (2026-09-04)

Frappe v15/16 gives `OAuth Client.allowed_roles` a default of `Desk User` that
**cannot be cleared** (re-populates on save). Frappe's `authorize` endpoint then
requires the user to hold one of those roles, and a mismatch (including an
unauthenticated Guest) surfaces as `{"error":"invalid_request","description":"Invalid
client_id parameter value.","status_code":400}` — a misleading message that sends you
hunting for a wrong client_id when the real cause is roles.

Consequences the runbook must cover:
- `docs/DEPLOYMENT.md` Phase-2 OAuth Client steps (1-4) say nothing about `allowed_roles`.
  They must state that it defaults to `Desk User` and that **every** app user needs the
  `Desk User` role, OR the four `ALLOWED_ROLES` roles must be added to the OAuth Client's
  Allowed Roles list. Otherwise a legitimate `Accounts User` who is a Website-only user is
  blocked at the ERPNext layer with the "Invalid client_id" error.
- The negative ("403") test needs a user that clears ERPNext OAuth but fails the app gate:
  give the test user **only** `Desk User`. Runbook Phase 5 should say this explicitly.
- Consider having the app map the upstream 400 to a clearer message in `api/auth/oauth2.py`.

- Found while running runbook Phase 5, incognito login as the negative-test user.

### evaluate.py spams a scary per-question ERROR on a fresh deploy (2026-09-04)

Runbook Phase 5's `python evaluation/evaluate.py --split test --no-judge` logs
`ERROR Internal error occurred. This is an unusual occurrence and we are monitoring
it closely. For help, please contact support: https://langfuse.com/support.` once
per question on a fresh deployment. Cause: `_link_dataset_run()` tries to link each
trace into a Langfuse Dataset run, but the golden-set dataset
(`contract-intelligence-golden-set`) has not been pushed to the new Langfuse project
(`evaluation/push_dataset.py` is a separate manual step, not in the runbook). The
per-stage traces still land fine and `scripts/benchmark_from_langfuse.py` (filters by
`name=eval_question`) is unaffected — the ERROR is cosmetic — but it looks like a
hard failure mid-run.

Fixes: (a) runbook Phase 5 should either run `push_dataset.py` first or state the
ERROR is benign when `--no-judge` and no dataset is pushed; (b) `_link_dataset_run`
is documented as "best-effort / silently skipped" but only wraps `get_dataset_item`
in try/except — the `item.link()` call and the SDK's async ingestion of the
dataset-run-item are outside it; make the whole thing genuinely silent.

- Found running runbook Phase 5 latency-baseline step.

### benchmark_from_langfuse.py can't run in the container or (easily) on the host (2026-09-04)

Runbook Phase 5 calls `python scripts/benchmark_from_langfuse.py` right after
`evaluate.py`. But: inside the `app` container it does `dotenv_values("/app/.env")`
and there is no `.env` file there (compose passes config as env vars, and `.env` is
gitignored / not COPYed) -> `KeyError: 'LANGFUSE_PUBLIC_KEY'`. On the host it needs
the repo's Python deps (imports `evaluation.evaluate`, `dotenv`, ...), which a bare
EC2 box doesn't have. Workaround used: `docker compose cp .env app:/app/.env`, run,
`rm` it. Fixes: (a) make the script fall back to `os.environ` when the `.env` file
is absent; (b) runbook should show the `docker compose cp .env ...` dance or a
dedicated `docker compose run` recipe. Same latent issue applies to any Phase 5/6
script the runbook expects to "just run".

### Phase 5 "403 negative test" is not doable via the real OAuth flow with a safe test user (2026-09-04)

The runbook says "user outside `ALLOWED_ROLES` -> 403". But to reach the app's role check,
the user must first clear ERPNext's `authorize` (System User + a role in the OAuth Client's
`allowed_roles`). A safe least-privilege test account (Website User, no roles) is blocked
earlier at ERPNext with the "Invalid client_id" error, and nobody should make a throwaway
account a System User on a shared/prod bench. Practical substitute used here: temporarily
set `ALLOWED_ROLES` on the box to a role the working test user lacks
(`ALLOWED_ROLES=Accounts Manager`), log in as that user, confirm
`{"detail":"Access denied — insufficient ERPNext roles"}` (403), revert `.env`, restart
`app`. Runbook Phase 5 should document this substitute as the default.

### docs/BENCHMARKS.md numbers are M1-laptop and partly stale (2026-09-04)

The committed latency/cost tables were measured on an M1 laptop at an older build.
The Phase 5 t3.large run diverges enough that the doc misleads a real deployment:

| stage (p50) | BENCHMARKS.md (M1) | t3.large (this run) |
|---|---|---|
| `rewrite` | 0.156 s | 1.46 s |
| `hybrid_search` | 0.166 s | 0.011 s |
| `rerank` | 0.150 s | 1.15 s |
| `generate` | 1.007 s | 0.94 s |
| end-to-end | 1.487 s | **3.88 s** (p95 5.48 s) |
| cost/query | $0.0042 | ~$0.0026 |
| full ingest | ~10 s / 61 ch | 17.7 s / 59 ch |
| webhook re-index | ~0.6 s | 1.66 s |

- The doc's `rewrite` p50 (0.156 s) is not credible for a gpt-4o-mini HyDE call and
  predates the #138 embed split — it should be re-measured regardless of instance.
- `rerank` and `rewrite` are the real t3.large gap (cross-encoder on 2 vCPU, no fast
  Apple PyTorch path); `hybrid_search` is *faster* now (#138 removed the duplicate
  dense-leg embed).
- Fix: refresh `docs/BENCHMARKS.md` with the t3.large figures (from
  `~/eval-results-139/benchmark.txt` on the box) on a branch, and mark the methodology
  row with the instance type. Do NOT touch `evaluation/results.baseline.json` — this
  was a `--no-judge` run.
