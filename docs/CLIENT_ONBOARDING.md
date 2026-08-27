# Contract Intelligence — Client Onboarding

**Audience:** client project sponsor · **Status:** draft for discussion — durations and scope
are confirmed in Phase 0.

Bringing conversational contract search to your ERPNext site: what we build, what we need from
you, and how the two weeks are spent. A separate [technical runbook](CLIENT_DEPLOYMENT_RUNBOOK.md)
covers the implementation detail.

---

## What you're getting

A private web application that answers plain-English questions about your contracts — grounded
entirely in the documents already in your ERPNext site.

Your team asks a question in ordinary language — *"What's the notice period on the Zuckerman
security contract?"* — and gets back a direct answer with a citation to the source document
every time. The system reads your **Contract** and **Terms and Conditions** records, plus any
PDFs attached to a contract. It never invents terms: if the answer isn't in your documents, it
says so.

```
ERPNext → Indexing → Search → Answer + citation → Web app
```

The application runs on a dedicated server we provision alongside your ERPNext — on your own
infrastructure or a cloud account you own. Your staff open it in a browser and sign in with
their existing ERPNext credentials; there is no new password to manage.

---

## What we need from you

Gathering these during scoping is the single biggest factor in hitting the go-live date. None of
it requires engineering work on your side.

- **An ERPNext administrator** — for a short session to create an API key, a login connection,
  and four notification rules. We supply the exact values and can drive it on a screen-share.
- **Confirmation your ERPNext site is reachable over HTTPS** from an external server. Most hosted
  and cloud ERPNext sites already are.
- **An OpenAI API key** — this powers the language understanding. Tell us during scoping if
  you'd prefer a different provider.
- **A subdomain you control** (for example `contracts.yourcompany.com`) and the ability to add
  two DNS records.
- **A named product owner and a contract domain expert** — the expert joins one validation
  session near the end.
- **Sign-off on the data & security summary below.**

---

## Data & security

Worth reading before scoping so sign-off isn't a blocker later.

- **What leaves your environment.** Contract and terms text, and the questions your staff ask,
  are sent to OpenAI to build the search index and generate answers. Nothing else is sent
  anywhere.
- **What stays.** The search index, the query-tracing database, and the application itself all
  run on the server we provision — infrastructure you own and control.
- **Who can use it.** Access is limited to four ERPNext roles: **Purchase Manager, Purchase
  User, Accounts User, System Manager**. Anyone else is refused.
- **How staff sign in.** Through your own ERPNext login. The application never sees passwords. A
  session lasts eight hours, after which the user signs in again — so removing someone's ERPNext
  access removes their access here within a day.
- **Inbound exposure.** The only endpoint reachable from outside is the ERPNext notification
  receiver, which rejects any request without a valid cryptographic signature. Your ERPNext API
  credentials never leave the server.

> **Sign-off required.** Because contract text is processed by OpenAI, this needs approval from
> whoever owns data-handling decisions on your side. We can provide OpenAI's data-processing
> terms to support that review.

---

## The seven phases

The sequence is fixed — each phase depends on the one before it. Durations are typical; contract
volume and your team's availability are the main variables.

| Phase | | Duration | Effort |
|---|---|---|---|
| 0 | Scoping & prerequisites | ~1 week | Mostly you |
| 1 | Environment setup | ~1 day | Us |
| 2 | ERPNext integration | ~½ day | Us + your admin |
| 3 | Deployment | ~½ day | Us |
| 4 | Loading your contracts | a few hours | Us |
| 5 | Validation & tuning | ~1–2 days | Us + your expert |
| 6 | Go-live & handover | ~½ day | Us |

### Phase 0 — Scoping & prerequisites

*~1 week · you lead*

**What happens.** We confirm the details: document scope (contracts, terms & conditions, and
contract-attached PDFs is the standard scope), the roles that get access, your ERPNext hosting,
and the language-model provider. You gather the items from *What we need from you*.

**You'll have.** A short scoping note and a confirmed start date.

### Phase 1 — Environment setup

*~1 day · we do this*

**What happens.** We provision the server, install the runtime, point your subdomain at it, and
issue the security certificates. No contract data is touched yet.

**You'll have.** A running — but still empty — environment on your own domain.

### Phase 2 — ERPNext integration

*~½ day · we do this, your admin joins*

**What happens.** In one short session, your ERPNext administrator creates the API key, the
"Login with ERPNext" connection, and four notification rules that tell the system when a contract
changes. We configure the access rules on our side.

**You'll have.** ERPNext and the application connected; a test login working end to end.

### Phase 3 — Deployment

*~½ day · we do this*

**What happens.** We deploy the application with your configuration and confirm every component
is healthy.

**You'll have.** A live application that passes its health check and shows the login screen at
your domain.

### Phase 4 — Loading your contracts

*a few hours · we do this*

**What happens.** A one-time import reads every existing contract and terms document from ERPNext
and builds the search index. Duration scales with how many documents you have. From this point
on, new and edited contracts are picked up automatically within seconds — no further imports
needed.

**You'll have.** Every current contract searchable.

### Phase 5 — Validation & tuning

*~1–2 days · we do this, your expert joins*

**What happens.** Your contract expert brings 10–15 real questions your team actually asks. We
run them together and review each answer and its citations. We adjust the retrieval settings
where the results warrant it.

**Optional add-on.** A formal accuracy benchmark with repeatable scores. This requires a
hand-built answer key covering 20–30 of your contracts, authored together with your expert, and
is scoped and priced separately. Most clients start with the guided review above and add the
benchmark later if they want an ongoing quality metric.

**You'll have.** A validation summary — the questions tested, the results, and any settings
changed.

### Phase 6 — Go-live & handover

*~½ day · we do this*

**What happens.** We open access to your team, hand over an operations guide and the technical
runbook, set up automated backups, and transfer all credentials into your secrets manager.

**You'll have.** A running system your team owns and operates.

---

## Timeline & effort

About **two calendar weeks** end to end. Of that, roughly **four to five working days** is our
hands-on effort. The rest is you assembling prerequisites in Phase 0 and the validation session
in Phase 5.

The two things most likely to extend the schedule: an ERPNext site that isn't yet reachable from
an external server, and a document scope wider than the standard contracts-and-terms set.

---

## After go-live

- **Contract changes are automatic.** Submit, edit, or cancel a contract in ERPNext and the
  search index updates within seconds. No scheduled re-imports.
- **Running cost.** The server plus OpenAI usage, which scales with how many questions your team
  asks — typically a small monthly figure. We'll give you a projection during scoping based on
  your expected volume.
- **Backups.** The search index and query history are backed up on a schedule you set. Your
  ERPNext data is covered by your existing ERPNext backups.
- **Support.** Ongoing support and any future scope changes are covered by a separate
  arrangement.

---

## Who does what

| Task | You | Us |
|---|---|---|
| Provide ERPNext admin access & API key | Own | Support |
| Provide OpenAI API key | Own | — |
| Subdomain & DNS records | Own | Specify |
| Data-handling sign-off | Own | Support |
| Provision server & certificates | — | Own |
| ERPNext connection & notification rules | Assist | Own |
| Deploy application | — | Own |
| Import existing contracts | — | Own |
| Validation questions & review | Own | Facilitate |
| Backups, handover, credentials transfer | Receive | Own |
| Operate the system after go-live | Own | Support\* |

\* Under a separate support arrangement.
