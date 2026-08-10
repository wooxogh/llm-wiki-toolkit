# AGENTS.md — working rules for an agent using the vault

This vault is a **canonical distilled-knowledge base**, not a code mirror. It
records *why*, *what was measured*, *what was disproven*, and behaviour that no
single repository owns — then routes you to the live code for current behaviour.

These rules are written for an LLM agent, but they are the same rules a human
should follow. Each step exists because skipping it produced a wrong answer.

## The routing loop

1. **Recall first.** `wiki-recall "<question>" --k 8`. Do this *before* grepping
   code or reading `index.yaml`. The point is to avoid repeating an
   investigation someone already wrote down.
2. **Read the returned page.** The recall snippet is not the answer. Open the
   canonical file it points at.
3. **Fall back to live code when the page is thin or stale.** The vault routes;
   code is the source of truth for current behaviour. Read the paths the page
   cites. If the answer lives in a different repository, the page's `projects:`
   tag says which — go read it there.
4. **Update with measured facts only.** See [Writing](#writing).
5. **Regenerate.** See [After editing](#after-editing).

## Recall

```bash
wiki-recall "how does the incremental embed store detect staleness" --k 8
wiki-recall "..." --rerank 10                    # ★ conceptual / pattern-shaped questions
wiki-recall "..." --layer domain --domain tooling
wiki-recall "..." --project some-repo --json
wiki-recall "..." --auto --json                  # one answer/review/none decision
```

Cross-lingual matching works: ask in whichever language you think in, regardless
of what the page is written in. One index serves both.

**Use `--rerank 10` for conceptual or pattern-shaped questions.** Measured
2026-07-31 over a 188-case gold set: first-stage ranking alone gets pattern Hit@1
26.3% and entity Hit@1 20.0%; adding the cross-encoder makes those 94.7% and
100% (overall 59.6% → 80.9%). Short pattern pages lose rank 1 to long domain
pages — the pages that stole it carried 2.2x the chunks of the right answer — and
the cross-encoder undoes that. Cost: ~5.9s cold model load per process, ~44ms
warm. `--auto` already reranks; plain `wiki-recall` does not, so ask for it.

`--auto` is the **non-interactive** entry point. It always runs the reranker
(pool of 10), and returns `answer` only when two independent signals agree: the
absolute cosine confidence clears its floor and beats the runner-up, **and** the
cross-encoder also scores that candidate as relevant. Otherwise `review` (you
decide) or `none` (the vault does not know). It fails closed — treat `review` as
"read the candidates", never as a weak answer. First call in a process pays the
model load.

Superseded pages are excluded by default. `--status any` opts back in.

## Writing

- **Measured facts only.** If it was not observed, it does not go in. An
  unmeasured claim in a knowledge base is worse than a gap, because the next
  reader cannot tell the difference. `wiki-lint` flags the *shape* of a guess —
  hedging with no number, date, PR reference, or code path near it. It is a
  WARNING and a heuristic: hedging is correct when a page deliberately records
  uncertainty, and it cannot catch a confident fabrication at all. Read the
  flagged lines; do not bulk-edit them.
- **Filename stem == `id`**, so `[[id]]` wikilinks resolve.
- Required frontmatter: `id, layer, projects, tags, confidence, status, summary`
  (plus `domain` when `layer: domain`). Types are enforced — lists must be lists,
  `updated` must be `YYYY-MM-DD`, `id` must be kebab-case.
- **Always quote `summary:`.** An unquoted ` #` starts a YAML comment, so
  everything after it is silently dropped from the index — and the summary is
  prefixed onto every chunk of that page. `wiki-index` now rejects this, but
  quoting is the habit that makes it a non-issue. See
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md#always-quote-summary).
- **Corrections keep the old claim.** Strike it through and say why it was wrong.
  A silently-edited page destroys the record of what was believed and disproven.
- Superseding knowledge → `status: superseded` on the old page + `links:` to the
  successor.
- Link related pages with `[[id]]`, and give a new page at least one inbound link
  so it is discoverable. Note the consequence: linking pages into one component
  creates a *community*, which needs an entry in `community_summaries.json`
  before `wiki-health` passes. See
  [docs/HYGIENE.md](docs/HYGIENE.md#community-synthesis).

## After editing

```bash
wiki-index                                             # regenerate index.yaml (validates)
wiki-embed                                             # semantic vectors (incremental)
python -m llm_wiki.reports.graph_report --write        # if you keep GRAPH_REPORT.md
python -m llm_wiki.reports.community_report --write    # if you keep COMMUNITIES.md
wiki-health --mode full                                # drift gate — must pass
```

`wiki-index --check` is exact: a valid-but-stale `index.yaml` is a hard failure,
because stale index entries silently poison recall.

If `community_report --stale` prints anything, read those communities' member
pages and write a grounded one-to-two-sentence synthesis into
`community_summaries.json` under the printed signature. Membership changes change
the signature, so a stale synthesis is dropped rather than served.

## Verifying

```bash
python -m pytest tests -q            # no torch, no network, no embedding server
wiki-health --mode ci                # CI-safe subset (skips .embeddings/ checks)
wiki-eval --validate-only            # gold schema, labels, coverage, id leakage
wiki-eval --k 8 --mode hybrid        # Hit@k / MRR / nDCG per slice
wiki-gate                            # ★ retrieval regression gate — before pushing ranking changes
```

`wiki-gate` compares per-layer Hit@k against `eval_baseline.json` and exits 1 on
a regression. Tolerance is whichever is larger — a flat 3 percentage points or
one case — because a flat gate fires on noise for small slices (an entity slice
with n=5 means one case is 20pp). `false_answers_gated` has **no** tolerance: it
is a safety property, not a quality metric. It is not in CI because measuring it
needs the embedding model.

## Daily ingest

`integrations/ingest/ingest_pipeline.py` owns ordering, fail-fast, and the
success stamp. The stamp is written **only after** every artifact has been
rebuilt and the health gate has passed — so a failed day is retried rather than
recorded as done. Details and failure semantics:
[docs/INGEST.md](docs/INGEST.md).

## Do not

- Do not index whole source trees into the vault. Live grep/read is the code's
  canon.
- Do not add a hosted vector DB or any external API dependency — recall is local
  and key-free by design.
- Do not hand-edit `index.yaml`, `GRAPH_REPORT.md`, or `COMMUNITIES.md`; they are
  generated.
- Do not put personal working-style notes or session goals here — those belong in
  your agent's per-project memory tier, not in canonical knowledge.
- Do not let `contradict` or `compact` rewrite a page. They produce *candidates*;
  the judgement and the edit are yours, after reading both pages.
- Do not update `eval_baseline.json` or `auto_thresholds.json` to make a gate go
  green. Both are only ever regenerated with a measured justification recorded
  alongside the change.
