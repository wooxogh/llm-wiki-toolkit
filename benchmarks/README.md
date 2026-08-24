# External benchmark suite

This workspace normalizes and scores externally obtained benchmark datasets
against recorded system predictions. It is offline by design: it does not
download datasets, call a model, or retrieve from a vault.

## Data and configuration

The repository contains only small synthetic fixtures in `fixtures/`; it does
**not** include LongMemEval, HoH, VitaminC, RGB, FactLens, or any other upstream
benchmark data. You are responsible for obtaining each dataset under its own
license and terms, and for recording the precise source release or revision in
your local configuration.

Copy the example configuration and replace each `path` with your locally
acquired dataset path. Keep the `version` values specific enough to reproduce a
comparison (for example, an upstream release tag, commit SHA, or archive
checksum). Local copied configurations, input data, and generated results are
ignored by Git.

```bash
cp benchmarks/configs/suite.example.yaml benchmarks/configs/suite.local.yaml
# edit suite.local.yaml: use benchmarks/data/... paths and source versions
```

`longmemeval`, `hoh`, `vitaminc`, and `rgb` are required suite entries.
`factlens` is optional: omit it from the local configuration when it is not
available. The tracked example uses fixtures only, so it is safe for validation
and test smoke checks but is not a substitute for an upstream evaluation.

## Run a benchmark

Install the benchmark package, then run commands from the repository root so
the paths in the configuration resolve consistently:

```bash
pip install -e 'benchmarks[dev]'

python -m llm_wiki_bench validate --config benchmarks/configs/suite.local.yaml
python -m llm_wiki_bench run \
  --config benchmarks/configs/suite.local.yaml \
  --suite hoh \
  --predictions /path/to/hoh-predictions.jsonl
python -m llm_wiki_bench report --run-dir benchmarks/results/<timestamp>-hoh
```

Predictions are JSON Lines records keyed by `case_id`. A run requires one
prediction per normalized case unless `--allow-skips` is supplied. The command
prints its timestamped artifact directory; use that directory as the input to a
later `report` command when regenerating its Markdown summary.

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
