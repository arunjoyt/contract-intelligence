# Guided Review — <client> — <date>

Phase 5 acceptance for a client deployment. Fill this in **live** with the client's
contract expert; it becomes the validation summary in the client's deployment
record. Copy it to `evaluation/client/<client>_guided_review.md` (gitignored, #118) —
it names real documents and clause content.

See `docs/CLIENT_DEPLOYMENT_RUNBOOK.md` § Phase 5 for where this sits in the sequence.

- **Client:** <name>
- **Corpus indexed:** <N> Contracts · <N> Terms and Conditions · <N> PDF attachments
- **Build under test:** commit `<sha>` · `QUERY_REWRITE_STRATEGY=<…>` · `RETRIEVAL_TOP_K=<…>` · `RERANK_TOP_N=<…>`
- **Present:** <engineer> · <client contract expert> · <product owner>

---

## Rubric

Each question gets an **answer verdict** and a **citation verdict**. A question
passes only if both are `ok`.

| Verdict | Answer | Citations |
|---|---|---|
| `ok` | Correct and complete; nothing invented | Every claim cites a `[docname]`; the cited docs actually contain it |
| `partial` | Correct but thin, or missing a caveat that's in the source | Cited docs are right but one claim is uncited |
| `fail` | Wrong, invented a term/number, or refused when the answer *is* in the corpus | Cites the wrong document, or the cited doc doesn't support the claim |
| `n/a-refusal` | Correctly said "I could not find relevant information…" for something genuinely not in the corpus | — |

A `fail` on retrieval grounds (right answer exists, wasn't retrieved) and a `fail`
on generation grounds (right context retrieved, answer still wrong) are different
fixes — note which in the **notes** column.

---

## Questions

10–15 real questions the client's team actually asks. Mix: single-clause lookups,
cross-document ("which of our contracts…"), and 1–2 you *expect* it to refuse
(a supplier you don't have, a clause type your contracts don't carry).

| # | Question | Answer verdict | Citation verdict | Notes (what was wrong / which doc should have been cited / retrieval vs generation) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |
| 11 | | | | |
| 12 | | | | |
| 13 | | | | |
| 14 | | | | |
| 15 | | | | |

**Tally:** `ok` __ / `partial` __ / `fail` __ / `n/a-refusal` __

---

## Findings

### Metadata-filter vocabulary

Does the client's ERPNext use doctype names or status values other than the
defaults (`Contract` / `Terms and Conditions`; `Cancelled` / `Active` / `Unsigned`)?
Check against a `status` field value in Qdrant and the client's own wording in the
questions above.

- [ ] Defaults are correct — no change
- [ ] Override needed → set in `.env` and re-test the affected questions:
  - `METADATA_FILTER_DOCTYPE_KEYWORDS` = `<json>`
  - `METADATA_FILTER_STATUS_KEYWORDS` = `<json>`

### Knob changes

Only when a group of questions fails the same way. Reach order (see
`docs/PIPELINE_TUNING.md` § Per-client tuning — "Reach order when a per-client slice
fails"): `RETRIEVAL_TOP_K` → system prompt vocabulary → `RERANK_TOP_N`. One knob at
a time; re-run the failing questions after each change.

| Knob | From | To | Which questions it fixed |
|---|---|---|---|
| | | | |

### Ingestion / data-shape issues

(Empty chunks, unparsed PDFs, missing metadata — these are ingestion fixes, not
tuning. Cross-reference the Phase 4a smoke-ingest checklist.)

-

---

## Decision

- [ ] **Accept** — deploy this config to the client-facing stack
- [ ] **Accept with the knob changes above** — promote the changed `.env` values
- [ ] **Blocked** — <reason>; re-review after <fix>

**Sign-off:** <engineer> · <client contract expert> · <date>
