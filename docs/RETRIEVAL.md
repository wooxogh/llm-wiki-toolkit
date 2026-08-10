# Retrieval

How a question becomes a ranked list of pages, and when that list is trustworthy
enough to consume without a human.

```
query ──► embed (dense)  ─┐
     └──► tokenize (BM25) ─┴─► RRF fuse over chunks ──► best chunk per page
                                                            │
                                         optional cross-encoder rerank (--rerank N)
                                                            │
                                    ┌───────────────────────┴──────────────┐
                                    │                                      │
                            ranked list (default)              --auto: answer/review/none
```

Everything runs locally. There is no API key in this path, and once the models
are on disk the only network call is to an optional resident embedding server on
`127.0.0.1`. The embedder and cross-encoder are downloaded on first use.

## Chunking

`llm_wiki/retrieval/embed_index.py`.

- `CHUNK_CHARS = 700` — target chunk size. Paragraphs are packed up to the cap.
- `_split_long()` splits a paragraph that exceeds the cap on its own, preferring
  a sentence or line break near the cap and hard-cutting only when a single
  sentence is longer than the cap.
- `CTX_CHARS = 300` — cap on the contextual prefix.

**Contextual prefix.** Every chunk is prefixed with `"<id>. <summary>"`, not just
the first. Without it, chunk #1 onward lose all page-level context and retrieve
as anonymous fragments. The prefix uses the page's own measured `summary:`, so it
is deterministic and needs no LLM.

The cap on that prefix exists because the prefix goes on *every* chunk: measured
2026-07-27, the longest summary in one vault was 2,554 characters and 18 of 179
pages had summaries over 300, which left one chunk at 3,299 characters even after
paragraph splitting. The blurb only needs to say which document a chunk belongs
to.

Chunk construction is part of the store's identity (`CHUNK_SCHEMA = "ctx-v3"`).
The incremental hash tracks raw page text, so a chunking change alone would not
re-embed anything; folding the scheme into the stored identity forces a full
rebuild instead. `META_SCHEMA = "meta-v2"` is a second, independent axis: chunk
text can be unchanged while the *record shape* in `meta.json` needs new fields on
every row.

## The embedding store

`.embeddings/` holds four files, each guarding a different drift class that
`wiki-health --mode full` checks:

| file | contents | drift it catches |
|---|---|---|
| `vectors.npy` | one row per chunk | row count vs `meta.json` record count |
| `meta.json` | per-chunk record: id, path, layer, domain, chunk no., projects, tags, confidence, status, updated, summary, text, snippet | first-stage retrieval reads *only* this, never the page — filters, sparse fields, and snippets must all be carried here (reranking is the one exception: it opens the page, see below) |
| `pages.json` | per-page SHA-1 of raw page text | pages never embedded, changed since embedding, or deleted |
| `model.txt` | `MODEL\|CHUNK_SCHEMA\|META_SCHEMA` | a model or schema change that leaves every file present and every hash matching while making every vector meaningless |

`wiki-embed` is incremental by default: it re-embeds only pages whose hash
changed, reuses cached vectors for the rest, and drops vectors for deleted pages.
It forces a full rebuild when `--full` is passed, when the store identity
changed, when no prior store exists, or on the weekly full-rebuild weekday
(Sunday). That lets a single daily call self-manage.

## Tokenization

`llm_wiki/retrieval/_retrieve.py`.

The sparse side is BM25 over a **fielded** document: id and tags repeated 3x,
summary 2x, body 1x. Repetition is the standard key-free way to express field
weights in a single-field BM25 index, so a term appearing in an id or a tag
carries three times the term frequency of the same term in the body.

Word tokens are kept exact. Hangul words additionally emit **character bigrams**,
because Korean is agglutinative and written without a space between a stem and
its particles — a query for a bare stem and a page writing the stem-plus-particle
form otherwise share no token at all. Latin and API-shaped tokens are deliberately
*not* bigrammed: splitting `orderstatuscode` into a dozen meaningless two-letter
terms would collide across unrelated pages and dilute exactly the rare-token
matching that sparse retrieval exists to provide.

### Rejected: a morphological analyser

A real Korean analyser is the linguistically correct answer, and it is still
reachable with `WIKI_TOKENIZER=kiwi` (optional dependency). It measured **worse**
(2026-07-31):

| slice | bigram | kiwi |
|---|---|---|
| pattern Hit@3 | 60.5% | 50.0% |
| overall Hit@3 | 77.1% | 73.9% |
| hard Hit@3 | 79.5% | 74.8% |

Bigrams are imprecise but give *fuzzy partial* matching, and that gold set was
dominated by indirect, conceptual queries (118 of 188) whose wording differs from
the page's. Morphology buys precision and pays for it in paraphrase recall. The
path is kept, not deleted, so the rejection stays reproducible — re-measure if
your corpus shifts toward direct, same-wording lookups.

## Fusion

Reciprocal Rank Fusion over the two chunk rankings:

```
fused[chunk] += DENSE_W  / (RRF_K + dense_rank)
fused[chunk] += SPARSE_W / (RRF_K + sparse_rank)
```

`RRF_K = 60` (standard damping), `DENSE_W = 2.0`, `SPARSE_W = 1.0`. Dense is the
stronger semantic signal, so it is weighted above sparse: BM25 then helps
exact-token queries without burying purely conceptual ones. Both weights are
env-tunable.

Two details that are easy to get wrong and were:

- **Filtering happens before rank enumeration.** RRF scores a document by its
  *rank*, so a filtered-out chunk still occupying ranks 0-1 would silently push
  every admissible result two places down. Excluded chunks must not exist as far
  as the ranking is concerned.
- **Pages aggregate by best chunk, not by sum.** Summing would let a long page
  outrank a precise short one just by having more chunks.

`mode="dense"` reproduces pure cosine, for comparison runs.

Sorting is stable (`np.argsort(kind="stable")`) and ties break on page id, so the
whole ranking is reproducible run to run.

### Filters

`--layer`, `--domain`, `--project`, `--confidence`, `--status`. `status="active"`
is the default: a superseded page is knowledge that was deliberately retired, and
serving it unasked is worse than serving nothing. `--status any` opts back in.

## Reranking

`--rerank N` fetches a pool of `N` candidates from the first stage and reorders
them with a local cross-encoder (`BAAI/bge-reranker-v2-m3`, multilingual). Off by
default in `wiki-recall`; always on in `--auto`.

Measured 2026-07-31 over a 188-case gold set:

| layer | first stage only | with `--rerank 10` |
|---|---|---|
| overall Hit@1 | 59.6% | 80.9% |
| domain | 75.0% | 81.8% |
| pattern | 26.3% | 94.7% |
| entity | 20.0% | 100% |

The cause is length bias in first-stage ranking, not content: the pages that
stole rank 1 carried 2.2x the chunks of the right answer (median 9 vs 4), and one
38-chunk page took rank 1 on six unrelated pattern queries.

**The reranker reads the full page body, not the retrieved chunk** (first 2,000
characters, frontmatter stripped). Measured 2026-07-24: chunk-text reranking
*regresses* easy cases (Hit@1 92% → 73%), while full-page reranking lifts hard
cases (Hit@1 64% → 76%) and holds easy ones.

Cost: ~5.9s cold model load per process, ~44ms warm.

`apply_rerank()` returns `(ranked, reranked: bool)` and absorbs only
import/model-load errors — corrupt frontmatter, a broken index, or a bad
embedding store still raise. That boolean is the whole point: if a model-load
failure quietly returned the un-reranked list as though it had been reranked, the
automatic-answer contract below would be unenforceable, because the caller could
not distinguish "the reranker agreed" from "the reranker never ran".

### Rejected: keeping the reranker resident

The embedding model is worth keeping warm (see below). The cross-encoder is not:
call frequency does not justify a second always-resident process, and *because*
the frequency is low, the ~5.9s cold cost is not a problem worth solving with
one.

## The `--auto` decision contract

`wiki-recall "..." --auto` emits one of three outcomes instead of a list:

- **`answer`** — safe to consume without asking.
- **`review`** — plausible but not decisive; hand the candidates to a human or
  agent.
- **`none`** — nothing clears the floor. Say so; do not serve the best of a bad
  set.

**On the `--auto` path, `answer` requires two independent signals to agree:**

1. the top candidate's absolute cosine clears `thresholds.score` **and** beats
   the runner-up by `thresholds.margin`, and
2. the cross-encoder scores that same candidate at or above `thresholds.rerank`.

Cosine says "this embeds close to the query". The cross-encoder says "this
actually answers it". Requiring only the first let a page the reranker scored ~0
be answered automatically — that is the measured failure the second signal
exists for.

`--auto` **always** runs the reranker (pool of 10, `AUTO_RERANK_POOL`). That is
what makes the second signal real rather than nominal: running without it and
then reporting the reranker as available would make the contract vacuous.

### The exact scope of that guarantee

It is a property of the `--auto` path, not an unconditional property of
`decide()`. Read the code, not the summary:

- If `reranker_available` is false, `decide()` returns `review`
  (`reason="reranker-unavailable"`) before evaluating the answer, margin, or
  cross-encoder thresholds. The `none` floor is checked *first*, though: a top
  score below `thresholds.none` returns `none` (`reason="below-none-threshold"`)
  whether or not a reranker ran. Both outcomes are fail-closed.
- If `reranker_available` is true but the top candidate carries **no** rerank
  score (`rerank_score is None`), `decide()` **skips the cross-encoder gate
  entirely** and can return `answer` on the cosine signals alone. A test pins
  this behaviour (`test_the_rerank_threshold_is_ignored_when_no_rerank_score_is_present`),
  so it is intended, not an oversight: records from a non-reranked path are
  already forced to `review` by `reranker_available=False`, and the threshold
  must not double-fire on a missing value.

Production never constructs that combination. All three callers — `wiki-recall
--auto`, `wiki-eval --auto`, and `wiki-gate` — build candidates as
`rerank_score = float(rank) if result.reranked else None` and pass
`reranker_available = result.reranked`, so either every candidate carries a
rerank score or the decision has already returned `review`.

Describe the guarantee that way. Do not state it as "`decide()` always requires
two signals", because that is not what the function does.

### Fail-closed everywhere else

- No candidates → `none`.
- Top score below `thresholds.none` → `none`.
- Missing, truncated, malformed, or unreadable threshold file → `UNCALIBRATED`,
  whose `score` and `margin` are `inf` — deliberately unreachable, so the policy
  degrades to `review`/`none` rather than answering on a guess. A partial file
  (missing `score`, `margin`, or `none`) is treated the same as an absent one
  rather than silently inheriting a lenient default. `rerank` is the one optional
  key, defaulting to 0.0.
- `decide()` never re-sorts. Order comes from retrieval; sorting by `score` here
  would silently discard the cross-encoder's ordering and re-rank by cosine — the
  very signal the reranker exists to override.

### Why confidence is cosine and not the fused score

RRF is scale-free by design: its score depends on rank position, so the top hit
of a nonsense query lands in the same narrow band as the top hit of a perfect
one. Measured 2026-07-31: negatives 0.0363–0.0489 versus positives 0.0425–0.0500
— overlapping, so no threshold separates them. Cosine, on a fixed scale, does
separate them: negatives 0.3617–0.4786 versus positives 0.4885–0.8846.

So: rank with RRF, order with the cross-encoder, decide with the cosine.

## Thresholds

Resolution order (`retrieval_policy.resolve_thresholds_path`, one implementation
shared by all three call sites):

1. an explicit `--thresholds PATH`, used exactly as given;
2. `<content root>/auto_thresholds.json`, if it exists — this is what makes your
   own `wiki-eval --calibrate` output take effect without passing a flag every
   time;
3. the packaged `llm_wiki/evaluation/auto_thresholds.json`.

The three tools resolving this differently used to mean the evaluation harness
measured something other than production.

### Recalibrating for your vault

```bash
wiki-eval --calibrate auto_thresholds.json      # fits on the calibration split only
```

`calibrate()` fits `(score, margin, none, rerank)` so that **no** calibration
record produces a false answer, then among all safe fits takes the one that
answers the most cases, breaking ties toward the *higher* thresholds so the fit
does not sit exactly on the boundary of the riskiest record it barely tolerated.
Candidate values are the score and margin values actually observed, not a fixed
grid, so the fit lands on a real decision boundary.

Two of the four floors are set directly from the calibration records' negatives
rather than by that search:

- `none` sits just above the loudest confidently-wrong negative, so unanswerable
  queries abstain instead of reviewing noise.
- `rerank` is `max(loudest negative rerank score, MIN_RERANK_FLOOR=0.5)`. A
  cross-encoder score below 0.5 means the model itself does not assert that the
  passage answers the query, and answering on that is unjustified no matter what
  a small calibration sample happened to contain. In the corpus this was fitted
  on, the calibration split held only four negatives — a floor fitted from four
  points does not generalise, and the test split must not be used to pick it.

The packaged file is a starting point calibrated on someone else's corpus, and it
says so in its own `_note`. Calibrate on yours.

## The optional resident embedding server

`embed_query()` tries `http://127.0.0.1:8477/embed` first and falls back to an
in-process model load. Warm server: ~0.1s per query. In-process cold load: ~7-9s.
The server (`integrations/macos/embed_server.py`) loads the model once at
startup, listens only on the loopback interface, and computes only when a query
arrives — the model sits idle in RAM at ~0% CPU/GPU between requests.
`integrations/macos/install.sh` registers it (and the daily ingest) as macOS
LaunchAgents. Everything works without it — just slower per cold process.

## Memory guards on Apple Silicon

Attention is O(sequence²). An uncapped ~3,000-token chunk inside a batch of 16
asks the GPU for a ~10 GiB score tensor in one shot, and the MPS caching
allocator does not hand memory back promptly. Measured 2026-07-27:

```
MPS backend out of memory (MPS allocated: 38.12 GiB, tried to allocate 12.84 GiB)
```

Three changes fixed it, and all three matter:

- `MAX_SEQ_TOKENS = 512` (`WIKI_EMBED_MAX_SEQ`). Chunks are ~700 characters
  (~280 tokens), so 512 truncates only the long tail.
- `BATCH_SIZE = 4` (`WIKI_EMBED_BATCH`), which bounds the peak linearly.
- Oversized-paragraph splitting in the chunker. Paragraph-only splitting had left
  a 7,603-character chunk and 116 chunks over 2,000 characters — which is a poor
  retrieval unit anyway, since one vector averaging many topics retrieves worse
  than several focused ones.

After those, the longest chunk fell from 7,603 to 1,053 characters and a full
rebuild peaked around 3.9 GB RSS in about 2.5 minutes.

**`torch.mps.empty_cache()` between slices was tried and measured as no help**
(3.81 GB without versus 3.98 GB with). It is deliberately absent; do not re-add
it. The out-of-memory failure came from sequence length and batch size, not from
cache retention.

## Telemetry (opt-in)

`--telemetry` or `WIKI_RECALL_TELEMETRY=1` appends one JSON line per recall to
`.local/recall-events.jsonl` (gitignored). It records the query, the filters, the
returned page **ids**, latency, the decision, and an optional outcome label. It
does **not** record page bodies, snippets, or any retrieved text — the ids are
enough to reconstruct a gold case, and page content in a log adds exposure
without adding signal.

`python -m llm_wiki.telemetry propose` prints candidate gold cases from
explicitly labelled events only, to stdout. It never writes to the gold file:
promoting a case is a human decision, because an unreviewed query/answer pair is
exactly the kind of unverified claim the vault's own rules forbid.
