# Switching Model Provider — Developer Steps (as of current codebase)

There is currently **no abstraction layer** for the LLM/embedding provider — `config.py`
(`OPENAI_MODEL`/`EMBEDDING_MODEL`) only centralizes *model names*, not the *provider*. Every call
site constructs a raw `openai.OpenAI()` client and reads its response shape directly. Swapping
providers today means touching each of these call sites by hand. This doc is the concrete list of
what to change — see `docs/ARCHITECTURE.md` § "Model Configuration" for how the current
OpenAI-only setup works, and issue #51 for why no abstraction exists yet (deferred — no swap was
planned when it was scoped).

## 1. Install the new provider's SDK

Add it to `requirements.txt` alongside (or replacing) `openai==1.54.0`. Check the new SDK's own
docs for its client construction pattern, message format, and response shape — these differ from
OpenAI's and drive most of the changes below (e.g. some providers take `system` as a top-level
constructor/call argument rather than a `"system"`-role message in the list; some don't support
`max_tokens`/`temperature` identically).

## 2. Swap the API key env var

Every call site reads `OPENAI_API_KEY` directly via `os.environ`:

- `pipeline/query_rewriter.py:44`
- `pipeline/query_pipeline.py:93`
- `ingestion/embedder.py:22`
- `evaluation/evaluate.py:251`

Replace with the new provider's key env var name at each site, and update `.env.example`/`.env`
(`.env.example:13`, the "OpenAI (embeddings + GPT-4o)" section).

## 3. Update `config.py`

`OPENAI_MODEL`/`EMBEDDING_MODEL` (`config.py:18-19`) hold model ID strings — rename or repurpose
them for the new provider's model IDs (e.g. update the default value and the env var name if you
want it to read something other than `OPENAI_MODEL`). If the new provider does embeddings with a
different vector dimension, add an entry to `_EMBEDDING_DIMENSIONS` (`config.py:24-28`) — this is
not optional: `retrieval/vector_store.py:32`'s `VECTOR_DIM` is derived from this dict at import
time, and an unrecognized model raises immediately (`config.py:34-40`).

**Before deploying an embedding-model swap:** recreate the Qdrant collection and run a full
re-ingest. The collection's vector size is fixed at creation (`ensure_collection` in
`retrieval/vector_store.py`), and existing points are not re-embedded automatically — a dimension
mismatch will make every existing point unsearchable, not just new ones.

## 4. Rewrite each of the 4 call sites

No shortcuts here — each one directly calls the OpenAI SDK and unpacks its response shape:

| File | Call | Response shape to replace |
|---|---|---|
| `pipeline/query_rewriter.py` | `_hyde()` (`:57-59`, temp 0.7) and `_step_back()` (`:69-71`, temp 0.3) both call `self._client.chat.completions.create(...)` | `response.choices[0].message.content` (`:67`, `:79`) |
| `pipeline/query_pipeline.py` | `_generate()` (`:156-158`, temp 0.0) | `response.choices[0].message.content` (`:166`) |
| `ingestion/embedder.py` | `embed_texts()`'s batched loop (`:36`) calls `self._client.embeddings.create(...)` | `item.embedding for item in response.data` (`:37`) |
| `evaluation/evaluate.py` | `_run_question()` (`:217-218`, temp 0.0) — mirrors `query_pipeline._generate()` for eval purposes | `response.choices[0].message.content` (`:226`) |

For each: replace the client construction, replace the API call with the new provider's equivalent,
and replace the response-shape unpacking. There are 4 independent `OpenAI()` client instances today
(one per file) — no shared client to update in one place.

## 5. Update the 3 test files that mock the OpenAI response shape

These hand-roll `MagicMock` chains matching the exact OpenAI SDK attribute path — they'll break
(or worse, silently stop testing anything real) if the code changes but the mocks don't:

- `tests/test_query_rewriter.py` — `_make_openai_response(content)` helper builds
  `resp.choices = [choice]`, `choice.message = msg`, `msg.content = content`; tests set
  `r._client = mock_client` and assert on `mock_client.chat.completions.create.call_args`.
- `tests/test_query_pipeline.py` — identical `_make_openai_response` helper; `pipeline._client =
  mock_openai`.
- `tests/test_embedder.py` — `_fake_response(vectors)` builds `response.data = [MagicMock(embedding=v)
  for v in vectors]`; the `embedder` fixture stubs `e._client.embeddings.create` directly
  (`Embedder.__init__` runs for real, so `api_key` construction must still work with whatever env
  var you swapped to in step 2).

Rewrite each mock to match the new provider's client/response shape, not the OpenAI one.

## 6. Don't forget: RAGAS's own judge LLM is separate

`evaluation/evaluate.py`'s `ragas_evaluate(...)` call (its metrics — faithfulness, answer_relevancy,
context_recall, context_precision) is not covered by step 4 — those metrics use their own
internally-configured LLM, not the `openai_client` built in `evaluate()`. If you swap the app's
provider but not RAGAS's, evaluation will keep calling OpenAI internally for scoring even though the
app itself has moved providers. Check RAGAS's own docs for configuring `llm=`/`embeddings=` on the
metric objects if this matters for your use case.

## 7. Update docs afterward

Once the code changes are made and tests pass: `docs/ARCHITECTURE.md` § "Model Configuration",
`README.md`'s Tech Stack table and Environment Variables table, and `.env.example` all currently
describe an OpenAI-only setup and should be updated to match.

## Considering an abstraction layer instead

If you expect to swap providers more than once, or need to support two providers side by side,
doing this by hand each time has an obvious cost. A hand-rolled `ChatModel`/`EmbeddingModel`
interface (so call sites depend on an interface instead of a concrete SDK) was scoped during #51
and explicitly deferred as not worth building until an actual swap was needed. If you're now doing
step 4 above for real, that's the signal to reconsider it — worth a fresh look rather than assuming
the old deferral still holds.
