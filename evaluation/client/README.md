# `evaluation/client/` — per-client eval artifacts (gitignored)

Everything in this directory is **ignored by git** except this README (see
`.gitignore`). Per the #118 decision, a real client's golden-set questions,
their contract text, and their eval scores **never enter this repo's history** —
they live only in the client's own Langfuse project and, transiently, here on
the deployer's machine.

## What goes here

| File | Produced by | Notes |
|---|---|---|
| `<client>_candidates.review.json` | `scripts/generate_eval_set.py --output evaluation/client/<client>_candidates.review.json` | LLM draft over the client's live collection; a human-review file, not a dataset |
| `<client>_dataset.json` | hand-review of the candidates with the client expert | the client's golden set — same schema as `../test_dataset.json` |
| `<client>_results.json` | `evaluate.py --dataset evaluation/client/<client>_dataset.json --output evaluation/client/<client>_results.json` | the client's eval run |

## The committed demo set is different

`../test_dataset.json` and `../results.baseline.json` are synthetic demo
fixtures (fictional suppliers, seeded from `scripts/seed_data/demo_data.yaml`) —
no real client data — so they stay committed. This directory is only for real
client corpora.

See `docs/CLIENT_DEPLOYMENT_RUNBOOK.md` § "Optional: client-specific RAGAS
benchmark" for the full flow, and `docs/ARCHITECTURE.md` § Evaluation for how
`push_dataset.py` syncs a client dataset into their own Langfuse project.
