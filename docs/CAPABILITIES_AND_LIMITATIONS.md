# Contract Intelligence — Capabilities & Limitations

**Audience:** prospective client, evaluator, pre-sales · **Status:** honest baseline. Every
number is measured (see [`BENCHMARKS.md`](BENCHMARKS.md)); every limitation is one we can
demonstrate. Read it before scoping so nothing here is a surprise later.

One-line summary: **plain-English question answering over the contracts already in your ERPNext,
with a source citation on every answer and an explicit "not in your documents" when there is no
answer.** It is a passage-grounded question-answering tool, not a reporting or analytics engine.

---

## What it does well

| Capability | Example question |
|---|---|
| **Exact-term lookup** — clause, penalty, figure, party | *"What penalty applies if Zuckerman Security exceeds two SLA incidents in a quarter?"* |
| **Paraphrase / concept search** — your words, not the contract's | *"Which supplier bears the cost of replacing defective goods under warranty?"* |
| **Cross-document comparison** | *"Compare the payment terms across our contracts with Alpha Supplies and Summit Traders."* |
| **Multi-contract disambiguation** — one supplier, several agreements | *"Under which Zuckerman agreement do we get a penalty for excess incidents?"* (current vs superseded, base vs addendum, primary vs backup) |
| **Abstract questions with no supplier named** | *"What recourse do we have if a security vendor doesn't perform to standard?"* |
| **Attached-PDF content** — text inside contract PDF attachments is indexed | *"What is the liability cap under the attached contract PDF?"* |
| **Grounded refusal** — no invented terms | *"What's our contract value with Globex Corp?"* → *"No contract with that supplier is in your documents."* |
| **Citations on every claim** | Each answer names the source `[CON-2026-00042]` so a reviewer can verify it |
| **Stays current** — an edit in ERPNext re-indexes that contract within seconds (webhook), no nightly batch |

### Measured quality

Reference set of 92 questions, scored on the held-out test split (LLM-judged,
`text-embedding-3-small` + `gpt-4o`):

| | Score | Reading |
|---|---|---|
| Faithfulness (answer supported by the cited source) | **0.87** | The remaining ~0.13 is mostly under-citation on multi-part answers, not fabrication |
| Answer relevancy | **0.88** | |
| Context recall (did retrieval find the right passage) | **0.94** | |
| Grounded-refusal handling | **1.00** | Every "not in your documents" case was handled correctly — no hallucinated answer |

Quality is validated **per deployment** against your real contracts and your team's real
questions (a guided review session), because an LLM judge on a demo corpus does not transfer to
your corpus. These numbers are the reference bar, not a contractual SLA.

---

## What it does *not* do

These are architectural boundaries, not bugs or tuning gaps. Each has a documented decision
record. If one of these is on your critical-question list, tell us during scoping — most are
solvable with additional scope, but none are in the base build.

| Limitation | What breaks | Why |
|---|---|---|
| **Counting / totalling / full enumeration** | *"List all cancelled contracts"*, *"total PO value to supplier X"*, *"how many contracts expire in Q3"* — answers may be **silently incomplete** | Retrieval hands the model its best 5 passages for one question. Any query matching more records than that drops the rest before the model sees them. Reliable support needs a separate metadata-filtered retrieval path + aggregation-aware generation — a different architecture. |
| **Time-relative questions** | *"Which contracts are active today"*, *"expiring in the next 3 months"*, *"anything signed since March"* | No date arithmetic in the query path, and a Contract's `status` in the index only refreshes when a human edits and saves it — ERPNext's own nightly status refresh is a direct DB write that does not notify the index. A lapsed `end_date` is invisible until the record is re-saved. |
| **Document-level access control** | Every permitted user sees answers drawn from the **entire** indexed corpus | Authorization is a single role gate at the door (Purchase Manager / Purchase User / Accounts User / System Manager). Fine when those roles already share the same procurement data in ERPNext; **not** fine if ERPNext User Permissions wall off specific companies or departments today. Enforcing that needs per-user filtering pushed into retrieval. |
| **Date-range and multi-select filters** | The sidebar / metadata filters match doctype and status only, as plain substring matches — no date ranges, no "any of these three suppliers" as a true OR | Filter extraction is a keyword scan, not an LLM. The keyword vocabulary is configurable per client; the matching stays literal. |
| **Role changes lag up to 8 hours** | A revoked ERPNext role still works until the user's session token expires | Roles are read once at login and baked into an 8-hour token. |
| **Languages other than English** | Untested | The pipeline, prompts, and reference set are English-only. |
| **High availability** | A host failure is a restore-from-backup event, not an automatic failover | Single-server deployment. Reboots recover on their own; hardware loss does not. |

---

## Performance & cost envelope

Measured on the reference corpus (~60 contract chunks), single user, one laptop-class machine.
Full method and caveats in [`BENCHMARKS.md`](BENCHMARKS.md).

| | Value | Notes |
|---|---|---|
| **Answer latency** | ~1.5 s median, ~2.6 s 95th percentile | ~70% of that is the OpenAI generation call; local search is ~0.5 s and flat |
| **Cost per question** | **~$0.004** | Independent of how many contracts you have — the model always sees a fixed 5-passage context. Scales with answer length, not corpus size. |
| **Monthly cost, 200 questions/day** | ~$25 OpenAI + ~$60 server ≈ **$85/mo** | |
| **Monthly cost, 1,000 questions/day** | ~$126 OpenAI + ~$60 server ≈ **$186/mo** | |
| **Full re-index of the corpus** | ~10 s for ~40 documents; **$0.00006** | One-time and on demand |
| **Incremental re-index (one edited contract)** | ~0.6 s | Automatic, on every ERPNext save |
| **Cold start after a reboot** | ~8 s to ready | |

**What is not yet measured:** behaviour under concurrent users. The re-ranking step processes
one request at a time, so simultaneous questions queue rather than run in parallel — the latency
figures above are single-caller and will rise under load. A concurrency benchmark and the real
corpus-size envelope are captured per deployment once there is live traffic.

---

## What leaves your environment

Contract and terms text, and the questions your staff ask, are sent to OpenAI to build the
index and generate answers. Nothing else. The index, the query-tracing database, and the
application run on infrastructure you own. A different model provider can be substituted during
scoping. See [`CLIENT_ONBOARDING.md`](CLIENT_ONBOARDING.md) § Data & security for the sign-off
this requires.

---

## Related documents

- [`CLIENT_ONBOARDING.md`](CLIENT_ONBOARDING.md) — what we build, what we need from you, the two-week sequence
- [`BENCHMARKS.md`](BENCHMARKS.md) — every performance and cost number here, with method
- [`ARCHITECTURE.md`](ARCHITECTURE.md) § Known Limitations — the technical root-cause detail behind each boundary above
- [`DOCUMENT_LEVEL_ACCESS_CONTROL.md`](DOCUMENT_LEVEL_ACCESS_CONTROL.md) — the proposed design if per-document access is in scope
