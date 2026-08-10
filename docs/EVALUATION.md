# Evaluation

Retrieval quality is measurable, so it is measured. `wiki-eval` scores a gold set
of (query → expected page) cases through the same retrieval path production uses;
`wiki-gate` fails the build when those numbers get worse.

```bash
wiki-eval --validate-only                    # schema + labels + coverage; embeds nothing
wiki-eval --k 8 --mode hybrid                # Hit@k / MRR / nDCG, overall and per slice
wiki-eval --split test --auto                # automatic-decision scoring (false answers must be 0)
wiki-eval --calibrate auto_thresholds.json   # fit thresholds on the calibration split
wiki-gate                                    # regression gate vs eval_baseline.json
wiki-gate --update                           # rewrite the baseline (justify it)
```

Only `--validate-only` is CI-safe. Everything else embeds queries, which needs
the `ml` extra and the embedding store.

## The gold schema

One JSON array, one object per case. Every field is explicit; nothing is
inferred from the query text.

```json
{"q": "why fuse BM25 and dense scores instead of picking one",
 "expect": ["hybrid-ranking-tradeoffs"],
 "expect_none": false,
 "split": "calibration",
 "category": "indirect",
 "difficulty": "easy",
 "layer": "domain",
 "domain": "tooling",
 "projects": ["some-repo"]}
```

| field | values | notes |
|---|---|---|
| `q` | non-empty string | must be unique across the file |
| `expect` | list of page ids | omit (or empty) only with `expect_none: true` |
| `expect_none` | bool | a query the corpus *should not* answer |
| `split` | `calibration` \| `test` | default `test` |
| `category` | `direct` \| `indirect` \| `ambiguous` \| `negative` | default `direct` |
| `difficulty` | `easy` \| `hard` | default `easy` |
| `layer` | `domain` \| `pattern` \| `entity` \| `raw` | required for every non-negative case |
| `domain` | vault-specific string | required when `layer` is `domain` |
| `projects` | list of strings | optional coverage label |

Loading is strict: a malformed case raises rather than being skipped, because a
silently-dropped case quietly inflates every score computed afterwards.

`--validate-only` additionally checks the gold set against the live index and
fails on:

- an `expect` id that does not exist in `index.yaml`;
- an `expect` id whose page is `status: superseded` — default recall excludes it,
  so the case can never pass;
- a `layer`/`domain` label that disagrees with the expected page's own
  frontmatter;
- `category: negative` without `expect_none`, or `expect_none` without
  `category: negative`;
- a missing `layer` label on a non-negative case;
- **id leakage**: an `indirect` or `ambiguous` query whose text literally contains
  its own expected page id. Such a case measures string matching, not recall.

## Why the calibration/test split exists

The `--auto` thresholds are *fitted* numbers. Fitting them on the same cases you
then report accuracy on makes them look safe purely by memorising those cases.
So:

- **Calibrate on the calibration split only** (`wiki-eval --calibrate` ignores
  everything else and refuses to run if that split is empty).
- **Report on the test split only** (`wiki-eval --split test`).

This is the same discipline as any train/test separation, and it is worth stating
because the failure mode is silent: nothing crashes, the numbers simply stop
meaning what they appear to mean.

## Coverage floors

A gold set that only asks easy questions about well-linked domain pages will
report excellent numbers and tell you nothing. `--validate-only` therefore
enforces curation minimums from `[eval.minimums]` in `wiki.toml`
(see [CONFIGURATION.md](CONFIGURATION.md#evalminimums)):

- `total` — overall case count.
- `recent_cases` — cases whose expected page was updated inside the newest
  N-day window *of the vault itself* (default 7 days). The newest knowledge is
  the most likely to be wrong-but-confident, so it has to be represented.
- per-`layer`, per-`domain`, and per-`category` floors. `ambiguous` and
  `negative` have their own floors because a corpus with no unanswerable
  questions cannot detect a system that answers everything.

The defaults are sized for a couple of hundred pages. Scale them to your corpus —
the example vault lowers `total` to 10 for nine pages, which is the intended use
of the setting, not a workaround.

Coverage output also reports `pages_referenced / pages_total`, which is the
honest measure of how much of the vault the gold set actually touches.

## Metrics

Per case: Hit@1, Hit@3, Hit@8, reciprocal rank, nDCG@3, and the rank of the first
correct hit. Aggregates are plain means over cases, reported overall and sliced
by difficulty, layer, domain, and category.

Two deliberate simplifications:

- **nDCG@3 collapses to the positional discount** of the first relevant hit
  within the cutoff, because there is a single graded-relevance level (a page is
  right or it is not), so IDCG is 1.
- **A negative case is scored on abstention**: it is correct exactly when
  retrieval returned nothing, and that correctness fills the same Hit@k fields
  rather than a parallel metric nobody would look at.

`wiki-eval` also prints every case that ranked worse than 3 or missed entirely,
with its expected ids. That list, not the aggregate, is what tells you what to
fix.

## Automatic-decision scoring

`--auto` scores the answer/review/none policy instead of the ranking:

```
AUTOMATIC CONSUMPTION
  answer  :  ...  (rate)  correct=...
  review  :  ...  (rate)
  none    :  ...  (rate)  correct=...
  ❌ FALSE ANSWERS: 0
```

A **false answer** is an `answer` decision on a case the corpus cannot answer
(`expect_none`) or one where the answered id is not in `expect`. This is the
number that matters. Answer rate is a convenience metric; false answers are a
safety property.

Decision records are built by reranking exactly as `wiki-recall --auto` does.
Calibrating on un-reranked candidates and then serving reranked ones would fit
thresholds to a distribution production never sees.

## The regression gate

`wiki-gate` measures the current state, compares it against `eval_baseline.json`,
and exits 1 on any regression.

It exists because of a specific incident: a ranking change (page-level RRF
instead of chunk-level) dropped pattern Hit@8 from 89.5% to 65.8% and entity
Hit@3 from 60% to 20%, and **nothing caught it except a human re-running the
evaluation by hand**. The gate was then verified by deliberately re-injecting
that change and confirming it fires.

Slices compared: `overall`, `layer:<name>`, `difficulty:<name>`, and `auto`.

**Tolerance is whichever is larger — a flat 3 percentage points, or one case.**
A flat percentage is wrong for small slices: an entity slice with five cases
means one case is 20 percentage points, and a 3pp gate would fire on noise
forever. `--tolerance-pp` adjusts the flat floor. A small epsilon absorbs the
rounding in the stored baseline, so an exact one-case drop does not fire
spuriously against a 1/n tolerance.

**`false_answers_gated` has no tolerance at all.** It counts false answers over
the `negative` and `ambiguous` test cases. It is a safety property, not a quality
metric, so any increase fails.

**A shrinking gold set is itself a regression.** If a slice has fewer cases than
the baseline, the gate reports it: rates can hold while coverage falls, which
hides real regressions.

`wiki-gate` is not a pytest test and is not in CI. Measuring means embedding every
gold query, which would drag the model into a suite that must stay torch-free.
The comparison arithmetic *is* unit-tested; this command is only the runner.

### Updating the baseline

`wiki-gate --update` rewrites it and prints a warning. A baseline change must be
justified by measurement, recorded in the commit message. Updating a baseline to
make a red gate go green, with no explanation of why the new numbers are correct,
converts the gate into decoration.

## Growing the gold set

The gold corpus can only contain questions someone thought to write down. Opt-in
telemetry (see [RETRIEVAL.md](RETRIEVAL.md#telemetry-opt-in)) records real
queries, and `python -m llm_wiki.telemetry propose` turns explicitly labelled
events into candidate cases on stdout:

- `useful` → a `direct` case expecting the top result;
- `none-correct` → a `negative` case with `expect_none`;
- `wrong` / `not-useful` → a known gap, emitted with `needs_human_label: true`
  and no expected id.

Nothing is ever written into the gold file automatically.
