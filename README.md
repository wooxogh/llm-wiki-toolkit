# llm-wiki

A local-first pipeline that turns a Markdown vault into memory an LLM agent can
trust.

Pages are plain Markdown with YAML frontmatter. Everything else — the index, the
vector store, the link-graph and community reports — is *generated* from those
pages, and every generated artifact is gated so it cannot silently fall out of
sync. Retrieval is hybrid (BM25 + dense embeddings, fused with Reciprocal Rank
Fusion) with an optional local cross-encoder rerank, and a non-interactive
`--auto` mode that returns a single `answer` / `review` / `none` decision instead
of a list of results. Nothing calls a hosted service; there is no API key
anywhere in the query path.

The rule the whole thing exists to enforce is **measured facts only**. An
unmeasured claim in a knowledge base is worse than a gap, because the next reader
cannot tell the difference. That rule is why `wiki-index` compares bytes instead
of re-validating, why the auto-answer policy fails closed, why the evaluation
harness separates the split it calibrates on from the split it reports on, and
why a linter flags hedging that carries no measurement.

## External benchmark suite

The offline external-dataset benchmark workspace is documented in
[benchmarks/README.md](benchmarks/README.md). It ships synthetic fixtures only;
obtain and configure upstream datasets locally before running a comparison.

## Quickstart

Everything below except the last block runs with the core install — no GPU, no
model download, no network.

```bash
git clone https://github.com/wooxogh/llm-wiki-toolkit.git
cd llm-wiki-toolkit
python3 -m venv .venv && .venv/bin/pip install -e .
export PATH="$PWD/.venv/bin:$PATH"
export WIKI_VAULT="$PWD/examples/vault"

wiki-init --agent codex     # only needed when the target vault has no wiki.toml
wiki-index                  # regenerate index.yaml from page frontmatter (validates)
wiki-health --mode ci       # drift gate over the generated artifacts
wiki-eval --validate-only   # gold-set schema + coverage check; embeds nothing
```

Real output against the shipped 9-page example vault:

```
$ wiki-index
⚠ orphans (0 inbound): 0/9 (0%)
✓ wrote index.yaml (9 entries)

$ wiki-health --mode ci
⚠ [unmeasured-claim] 2 hedged claim(s) with no nearby measurement across 1 page(s): unmeasured-claims-are-worse-than-gaps(2) — heuristic; hedging can be correct, read before editing
✓ healthy (mode=ci, 0 error(s), 1 warning(s))

$ wiki-eval --validate-only
cases: 12  |  pages referenced: 9/9
  split      : {'calibration': 5, 'test': 7}
  layer      : {'-': 2, 'domain': 4, 'entity': 1, 'pattern': 5}
  domain     : {'-': 8, 'research': 1, 'tooling': 3}
  category   : {'ambiguous': 1, 'direct': 4, 'indirect': 5, 'negative': 2}
  difficulty : {'easy': 7, 'hard': 5}
  project    : {}
  recent-window: 8 case(s) touch pages updated since 2026-01-12

✓ gold set valid and complete
```

Semantic recall needs the optional `ml` extra (torch + sentence-transformers,
plus a first-run model download). It was developed on Apple Silicon, so it asks
torch for the `mps` device by default; on any other machine that device is
unavailable and it falls back to CPU, printing one line on stderr to say so. Set
`WIKI_EMBED_DEVICE=cuda` (or `cpu`) to choose explicitly. **These four commands
do not work on the core install:**

```bash
pip install -e ".[ml]"

wiki-embed                                                    # build .embeddings/
wiki-recall "why fuse BM25 and dense scores" --k 5 --rerank 10
wiki-recall "why fuse BM25 and dense scores" --auto --json    # one answer/review/none decision
wiki-health --mode full                                       # adds the .embeddings/ checks
```

`--mode ci` is the subset of `--mode full` that skips everything derived from
`.embeddings/`, because CI has no embedding store. Running `--mode full` before
`wiki-embed` reports `embedding-store-missing` and exits 1 — that is correct
behaviour, not a bug.

### The first thing a new vault trips on

Give every page an inbound link and your pages become one connected component.
Label propagation then finds *communities* (2 or more members), and a community
of 3 or more members (`MIN_SYNTH_SIZE` in
`llm_wiki/reports/community_report.py`) is one for which a grounded synthesis is
*required* — its absence is a hard failure:

```
❌ [community-synthesis-stale] 1 community/communities await a grounded synthesis: vault(84ebd5e58fee) — add community_summaries.json at the vault's content root, keyed by the signature shown above (e.g. "84ebd5e58fee"), with a short grounded synthesis of that community's member pages as the value
❌ unhealthy (mode=ci, 1 error(s), 1 warning(s))
```

The remedy is in the message: write `community_summaries.json` at the vault's content root,
keyed by that signature. `examples/vault/community_summaries.json` is a worked
example. Full explanation in [docs/HYGIENE.md](docs/HYGIENE.md#community-synthesis).

## Architecture

```
   domain/ patterns/ entities/ raw/     canonical Markdown + YAML frontmatter
                 │                      (the only thing you hand-edit)
                 ▼
   wiki-index ──► index.yaml            generated; byte-compared, never merged
   wiki-embed ──► .embeddings/          generated; per-page hashes track staleness
   reports    ──► GRAPH_REPORT.md
                  COMMUNITIES.md
                 │
                 ▼
   wiki-recall     hybrid retrieval  ──► optional cross-encoder rerank
                 │
                 ├─ list mode  → ranked pages for a human
                 └─ --auto     → one answer / review / none decision (fails closed)
                 │
                 ▼
   wiki-health     one exit code over every drift class above
   wiki-eval       Hit@k / MRR / nDCG over a gold set, per layer and slice
   wiki-gate       fails when those numbers regress against a committed baseline
   wiki-lint       flags hedged claims that carry no measurement (warning only)
```

Three properties hold that whole picture together:

- **Pages are the source of truth; everything else is derived.** `wiki-index`
  regenerates `index.yaml`; `wiki-index --check` compares the *rendered bytes*
  against the committed file. A syntactically valid but stale index is a hard
  error there, because stale index entries poison recall while looking healthy.
- **Ranking and confidence are different signals.** Order comes from RRF and the
  cross-encoder; confidence is the absolute cosine. See
  [what was rejected](#rrf-scores-as-a-confidence-signal) for the measurement
  that forced the split.
- **The model lives behind a seam.** `retrieval/_embedder.py` and
  `retrieval/_rerank.py` are the only modules that touch torch, so the entire
  ranking, policy, health, and evaluation layer is unit-testable without a GPU.
  The suite is 312 tests (311 passing, 1 skipped on an optional dependency) and
  imports neither torch nor sentence-transformers.

## Commands

| command | what it does | fails when |
|---|---|---|
| `wiki-init --agent codex\|claude` | create the minimal v2 `wiki.toml`; interactive first commands run this automatically | an existing config is never overwritten |
| `wiki-index` | regenerate + validate `index.yaml` | frontmatter violates the schema; `--check` also fails on a stale index |
| `wiki-health --mode ci\|full` | drift gate over every generated artifact | index stale, embeddings missing/stale/mismatched, a report or community synthesis stale |
| `wiki-embed` | build/refresh `.embeddings/` (incremental) | needs the `ml` extra |
| `wiki-recall "..."` | hybrid search; `--auto` for one decision | needs the `ml` extra |
| `wiki-eval` | Hit@k / MRR / nDCG per slice; `--validate-only` for schema+coverage | gold schema, label, coverage, or id-leak violations |
| `wiki-gate` | retrieval-regression gate vs `eval_baseline.json` | any slice drops beyond tolerance; any gated false answer |
| `wiki-lint` | hedged claims with no nearby measurement | never — warning only, by design |
| `python -m llm_wiki.hygiene.contradict` | contradiction *candidates* (needs `.embeddings/`) | — |
| `python -m llm_wiki.hygiene.compact` | UPDATE/MERGE/SPLIT *candidates* (needs `.embeddings/`) | — |
| `python -m llm_wiki.reports.graph_report --write` | `GRAPH_REPORT.md` | — |
| `python -m llm_wiki.reports.community_report --write\|--stale` | `COMMUNITIES.md`; list communities awaiting synthesis | — |

`contradict` and `compact` deliberately have no console script: they are
candidate generators for a human pass, not routine commands.

## Using It With Codex

The core toolkit is agent-agnostic: Codex can use the same vault through the
shell commands above. Put the vault rules where Codex reads project guidance
(`AGENTS.md` in the vault, or a short pointer from your repo's own `AGENTS.md`),
then ask Codex to recall before it edits:

```bash
export WIKI_VAULT="$PWD/examples/vault"
wiki-recall "how should this repo handle stale embeddings" --k 8 --rerank 10
```

If you use the optional per-project memory cache, sync it to Codex instead of
Claude, or to both runtimes during a migration:

```bash
python -m integrations.agent_memory.sync_cache --project /abs/path/to/repo --target codex
python -m integrations.agent_memory.sync_cache --project /abs/path/to/repo --target both
```

For unattended daily ingest, set the authoring runtime in `wiki.toml`:

```toml
[ingest]
repos = ["/abs/path/to/repo"]
agent = "codex"
```

or override it for one run:

```bash
python -m integrations.ingest.ingest_pipeline --agent codex
```

The Codex ingest step runs `codex exec` non-interactively, uses the vault as its
workspace, and passes every configured repo as an additional readable/writable
root with `--add-dir`.

## Measured evidence

Measured on the author's private 214-page vault over a 188-case gold set
(2026-07-31):

| layer | first stage only | with `--rerank 10` |
|---|---|---|
| overall Hit@1 | 59.6% | **80.9%** |
| domain | 75.0% | 81.8% |
| pattern | 26.3% | **94.7%** |
| entity | 20.0% | **100%** |

The cause is not content, it is length bias in first-stage ranking: the pages
that stole rank 1 carried 2.2x the chunks of the right answer (median 9 vs 4),
and a single 38-chunk page took rank 1 on six unrelated pattern queries. A
cross-encoder scores the query against the whole page rather than against a
chunk, which undoes the bias. Cost: ~5.9s cold model load per process, ~44ms
warm.

Two consequences shaped the defaults. Rerank is **off** in plain `wiki-recall`
(the fast path stays fast) and **always on** in `--auto`, because the automatic
answer contract is stated in terms of the cross-encoder agreeing and a contract
you do not actually evaluate is vacuous. And short pages — patterns, entities —
are exactly the ones that need it, which is counterintuitive enough that it is
worth restating: `--rerank 10` matters most for conceptual questions, least for
long factual domain pages.

Those figures come from one corpus. Yours will differ. What transfers is the
*shape* of the finding and the method that found it: a gold set with layer
labels, a calibration/test split, and a committed baseline that fails the build
when a ranking change makes things worse.

### The `--auto` contract

`--auto` emits `answer` only when two independent signals agree: the top
candidate's absolute cosine clears its calibrated floor **and** beats the
runner-up by the calibrated margin, **and** the cross-encoder scores that same
candidate above the `rerank` floor in the thresholds file — which `calibrate()`
never fits below `MIN_RERANK_FLOOR` (0.5). Cosine says
"this embeds close to the query"; the cross-encoder says "this actually answers
it". Requiring only the first measurably let a page the reranker scored ~0 be
answered automatically.

This is a property of the `--auto` path specifically, not an unconditional
property of `decide()`. See
[docs/RETRIEVAL.md](docs/RETRIEVAL.md#the---auto-decision-contract) for the exact
scope and the branch it does not cover.

Everything else — an unavailable reranker, an unreadable threshold file, a tie at
the top — resolves to `review` or `none`. A wrong automatic answer is far more
expensive than an unnecessary review, because at the point of use it is
indistinguishable from a right one.

## What was tried and rejected, and why

Each of these was implemented or measured, then reverted. They are recorded
because "we considered it" is not knowledge; "we measured it and it was worse by
this much" is.

### A hosted vector database

Rejected on design grounds, not performance. Recall has to work offline, with no
API key, on a laptop, inside an agent session that may have no network. Every
external dependency in the query path is a way for recall to fail at exactly the
moment an agent is relying on it. The embedding model runs locally; the store is
a `.npy` file and a JSON sidecar. This is the one rejection with no number
attached, and it is deliberate: it is a constraint, not a benchmark result.

### Page-level RRF instead of chunk-level

Fusing ranks per *page* looked like the obvious fix for the length bias above —
score pages once instead of letting a page with many chunks occupy many ranks.
Measured: pattern Hit@8 fell **89.5% → 65.8%** and entity Hit@3 fell **60% → 20%**.
Reverted. At 214 pages the RRF discount curve is flat enough that collapsing to
page granularity throws away the signal that distinguishes candidates rather than
normalising it. The cross-encoder addresses the same bias without the loss.

This regression is also why `wiki-gate` exists: nothing caught it except a human
re-running the evaluation by hand. The change was later re-injected on purpose to
confirm the gate catches it.

### A Korean morphological analyser

The sparse side tokenises Hangul into character bigrams, which is linguistically
crude — Korean glues particles onto stems, and a real analyser splits them
properly. Switching to one (`WIKI_TOKENIZER=kiwi`, still available) measured
**worse**: pattern Hit@3 60.5% → 50.0%, overall Hit@3 77.1% → 73.9%, hard Hit@3
79.5% → 74.8%.

Why the correct-looking option loses: bigrams are imprecise but give *fuzzy
partial* matching, and this gold set is dominated by indirect, conceptual queries
(118 of 188) whose wording does not match the page's. Morphology buys precision
and pays for it in paraphrase recall. The code path is kept and documented so the
rejection stays reproducible — re-measure if your corpus shifts toward direct,
same-wording lookups.

### Keeping the reranker resident

There is a resident embedding server (`integrations/macos/embed_server.py`) that
keeps the *embedding* model warm, taking a query from ~7-9s cold to ~0.1s. The
obvious next step was to put the cross-encoder behind it too and delete the ~5.9s
cold load.

Rejected. The call frequency does not justify a second always-resident process —
and because the frequency is low, the cold cost is not a problem worth a resident
process either. The two halves of that sentence are the same observation; the
argument for the daemon and the argument against needing it cancel.

### RRF scores as a confidence signal

The fused score is right there and looks like a confidence. It is not one.

```
Confidence comes from absolute cosine, never from the fused RRF score. RRF is
scale-free by design, so the top hit for a meaningless query lands in the same
narrow band as the top hit for a real one (negatives 0.0363–0.0489 vs positives
0.0425–0.0500 — overlapping, so no threshold separates them). Cosine separates
cleanly (0.3617–0.4786 vs 0.4885–0.8846).
```

So: rank with RRF, order with the cross-encoder, and decide automatic
consumption with the cosine. Three different questions, three different signals.

## Install

Requires Python 3.11 or newer (the config loader uses stdlib `tomllib`).

```bash
pip install -e .          # core: pyyaml, numpy, rank-bm25
pip install -e ".[ml]"    # + torch, sentence-transformers (embedding + rerank)
pip install -e ".[dev]"   # + pytest
```

The core install runs indexing, health, linting, reports, the gold-set validator,
and every test. The `ml` extra is needed only to embed or to query.

Point the tools at a vault in one of three ways, in this order of precedence:
`WIKI_VAULT`, the nearest ancestor directory containing `wiki.toml`, or the
current directory. `wiki.toml.example` documents every key; the built-in defaults
are a complete configuration, so a vault with no config file works.

## Documentation

- [AGENTS.md](AGENTS.md) — the working rules for an LLM agent reading and writing the vault
- [docs/CODEX_USAGE_KO.md](docs/CODEX_USAGE_KO.md) — Codex용 한국어 사용법
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — `wiki.toml` reference, vault-root discovery, the frontmatter contract
- [docs/RETRIEVAL.md](docs/RETRIEVAL.md) — chunking, tokenizer, fusion, rerank, the `--auto` contract, threshold recalibration, memory guards
- [docs/EVALUATION.md](docs/EVALUATION.md) — gold schema, splits, coverage floors, the regression gate
- [docs/HYGIENE.md](docs/HYGIENE.md) — claim lint, contradiction and compaction passes, community synthesis
- [docs/INGEST.md](docs/INGEST.md) — the daily ingest playbook and its failure semantics
- [CONTRIBUTING.md](CONTRIBUTING.md) — layering rules and the test boundary

## License

MIT. See [LICENSE](LICENSE).
