# External benchmark suite

This workspace normalizes and scores externally obtained benchmark datasets
against recorded system predictions. It is offline by design: it does not
download datasets, call a model, or retrieve from a vault.

## Data and configuration

The repository contains only small synthetic fixtures in `fixtures/`, shaped
like each release's real container format but not sampled from it; it does
**not** include LongMemEval, HoH, VitaminC, RGB, FactLens, or any other
upstream benchmark data. You are responsible for obtaining each dataset under
its own license and terms, and for recording the precise source release or
revision in your local configuration.

| Suite | Source | License | Container | Evidence ids |
| --- | --- | --- | --- | --- |
| `longmemeval` | HF `xiaowu0162/longmemeval-cleaned` | MIT | JSON array | upstream |
| `hoh` | HF `russwest404/HoH-QAs` | Apache-2.0 | Parquet (needs `benchmarks[hoh]`) | synthesized |
| `vitaminc` | HF `tals/vitaminc` (`test.jsonl`) | CC-BY-SA-3.0 | JSONL | upstream |
| `rgb_base` | GitHub `chen700564/RGB` (`en_refine.json`) | none declared | JSONL | synthesized |
| `rgb_integration` | GitHub `chen700564/RGB` (`en_int.json`) | none declared | JSONL | synthesized |
| `rgb_counterfactual` | GitHub `chen700564/RGB` (`en_fact.json`) | none declared | JSONL | synthesized |
| `factlens` | GitHub `megagonlabs/factlens` (`benchmark/fact_lens_benchmark.csv`) | BSD-3-Clause | CSV | upstream |

Measured against the full released source for every suite (`conformance`
passes end to end on all seven):

| Suite | Records | Split used |
| --- | --- | --- |
| `longmemeval` | 500 (oracle) | `oracle` |
| `hoh` | 111,972 | `240601_241201` |
| `vitaminc` | 55,197 (test) | `test` |
| `rgb_base` | 300 | `en_refine` |
| `rgb_integration` | 100 | `en_int` |
| `rgb_counterfactual` | 100 | `en_fact` |
| `factlens` | 733 | `benchmark` |

LongMemEval ships `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json`,
and `longmemeval_oracle.json` (note the `_cleaned` suffix on the full files).
VitaminC ships `train.jsonl`, `dev.jsonl`, `test.jsonl`. HoH ships a single
`hoh_qas_240601_241201.parquet`.

State plainly what these numbers are and are not:

- **Synthesized evidence ids** (`hoh` and all three `rgb_*` suites) mean a
  retrieval score for those suites is computed over identifiers this adapter
  assigned, not over identifiers the dataset itself carries — those releases
  ship no document identifiers at all. `upstream` (`longmemeval`, `vitaminc`,
  `factlens`) means the score is computed over the dataset's own ids. Every
  run's `manifest.json` records `evidence_id_origin` so this is never
  ambiguous after the fact.
- **RGB's `noise_rate` and `passage_num`** (see `rgb_base` below) are recorded
  from configuration into the manifest for provenance and used to assemble
  nothing. RGB supplies document pools; this suite scores recorded
  predictions and never calls a model or builds a prompt from those pools.
- **RGB declares no license.** Acquiring `chen700564/RGB` is your decision;
  no RGB data is redistributed in this repository.
- **`longmemeval_oracle.json` cannot measure retrieval.** Measured directly:
  its cases have a median of 2 haystack sessions with a median of 2 gold
  sessions, so essentially the whole haystack is the answer and a random
  ranker scores ~1.0 recall@1 by construction. It is a reader-evaluation
  file only. Use `longmemeval_s_cleaned.json` (or `_m_cleaned.json`) for a
  meaningful retrieval score, at the memory cost noted below.
- **Fixtures in `fixtures/` are synthetic**, built to the released container
  shape (JSON array, Parquet, JSONL, CSV) so tests exercise real readers; they
  are not samples drawn from any dataset.

### Memory

`read_json_array` and `read_parquet` (used by `longmemeval` and `hoh`) load
their whole source into memory before normalizing; there is no cheap streaming
alternative because stdlib `json` has no incremental array parser. This is a
known, accepted limitation, not a bug — but two sources make it matter in
practice: `longmemeval_s_cleaned.json` is **277 MB** on disk, and
`hoh_qas_240601_241201.parquet` holds **111,972 records**. Loading either
needs several GB of resident memory; budget for it before pointing a suite at
the full file. The practical LongMemEval file for this suite,
`longmemeval_oracle.json`, is only **15 MB** — but see the retrieval caveat
above before choosing it.

Copy the example configuration and replace each `path` with your locally
acquired dataset path. Keep the `version` values specific enough to reproduce
a comparison (for example, an upstream release tag, commit SHA, or archive
checksum); set `expected_digest` once you know the content digest a
`conformance` run reports, to catch a silently swapped file on a later run.
`run_parameters` is a free-form mapping recorded into the manifest verbatim —
RGB's `noise_rate`/`passage_num` are the only entries used today. Local copied
configurations, input data, and generated results are ignored by Git.

```bash
cp benchmarks/configs/suite.example.yaml benchmarks/configs/suite.local.yaml
# edit suite.local.yaml: use benchmarks/data/... paths, real source versions,
# and keep the hoh split ("240601_241201") quoted
```

Every suite above is a required entry except `factlens`, which is optional:
omit it from the local configuration when it is not available. The tracked
example config points at `benchmarks/data/...` paths that are not part of this
repository, so it is safe to commit but will correctly fail `validate` in a
checkout with no downloaded data (see Verification below) — it is not a
substitute for pointing the config at a real, downloaded release.

## Run a benchmark

Install the benchmark package, then run commands from the repository root so
the paths in the configuration resolve consistently:

```bash
pip install -e 'benchmarks[dev]'   # or 'benchmarks[hoh]' for pyarrow alone

python -m llm_wiki_bench validate --config benchmarks/configs/suite.local.yaml
python -m llm_wiki_bench conformance --config benchmarks/configs/suite.local.yaml --suite rgb_base
python -m llm_wiki_bench run \
  --config benchmarks/configs/suite.local.yaml \
  --suite rgb_base \
  --predictions preds.jsonl
python -m llm_wiki_bench report --run-dir benchmarks/results/<timestamp>-rgb_base
```

- `validate --config PATH [--sample N]` checks the configuration's structure,
  confirms every configured suite's `path` exists, and reads up to `--sample`
  records (default 5) of each configured/enabled suite through its adapter as
  a conformance smoke check. Fails loudly, naming every missing path, on a
  config with no downloaded data.
- `conformance --config PATH [--suite NAME]` reads an entire configured
  source (or just `--suite NAME` when given) through its adapter, reporting
  record count, a content digest, and every record that failed to normalize
  (capped in the printed list, never in the reported count). Exits non-zero on
  any failure.
- `run --config PATH --suite NAME --predictions PATH [--allow-skips]` scores
  one suite's normalized cases against a JSON Lines predictions file keyed by
  `case_id`. A run requires one prediction per normalized case unless
  `--allow-skips` is supplied. Prints its timestamped output directory.
- `report --run-dir PATH` regenerates `report.md` from a run directory's
  `manifest.json`, `metrics.json`, and `per_case.jsonl`.

Each run writes:

- `manifest.json` — suite, source path/version, configured split and `top_k`,
  plus total/evaluated/skipped/error counts.
- `metrics.json` — aggregate metrics and, for label tasks, the confusion matrix.
- `per_case.jsonl` — the recorded prediction, score fields, and status for each
  normalized case.
- `skips.jsonl` — skipped or scoring-error cases, with the reason.
- `report.md` — a deterministic Markdown rendering of the metrics.

Treat the configuration together with this manifest as run provenance. Preserve
the local configuration copy and the recorded prediction file alongside any
published comparison.

## Tests

The benchmark tests run from either root. From the repository root they are
collected alongside the main suite; from `benchmarks/` they run standalone
against the source tree:

```bash
pytest                  # repository root: main suite + benchmark tests
cd benchmarks && pytest # benchmark tests only
```

## Metric semantics

For evidence-bearing tasks, `recall@k` means **any** expected evidence item
appears in the first `k` returned identifiers; `mrr` and `ndcg@3` score the
first matching evidence item. `evidence_set_coverage` separately reports the
fraction of the complete expected evidence set that was retrieved, which is
essential for multi-evidence cases.

Citation precision measures correct cited identifiers among citations returned;
citation recall measures expected identifiers that were cited. When a system
returns no citations, citation precision is `null` (undefined), never a
successful score. Answer tasks use Unicode-normalized exact match and
whitespace-token F1. Label tasks report label accuracy and a deterministic
expected-label-by-predicted-label confusion matrix.

## Comparing runs

Only compare runs that use the same source version, split, normalization
adapter, `top_k`, prediction/skip policy, and task coverage. Different dataset
releases or partial prediction sets are separate experiments, not regressions.
Use the manifest counts to make exclusions visible before comparing aggregates.

This suite evaluates external, locally acquired datasets and recorded system
outputs. It is separate from `wiki-eval` and `wiki-gate`: those commands assess
the repository's committed vault gold data and retrieval-regression baseline,
respectively. Do not replace their gold data or baseline with external benchmark
results.
