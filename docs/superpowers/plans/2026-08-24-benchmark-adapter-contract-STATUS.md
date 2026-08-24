# Benchmark adapter contract redesign — complete

All 16 tasks of `docs/superpowers/plans/2026-08-24-benchmark-adapter-contract.md` are
implemented and reviewed, followed by a whole-branch review and one fix wave.

## What this closed

An external review rejected the original `benchmarks/` workspace with four findings.

1. **Adapters normalized a fixture contract, not the released formats.** Every record was
   required to carry its own `source_version` and `split`, which no release provides, and the
   loader read only JSONL. Now: four container readers, `split` supplied by configuration,
   version derived as a SHA-256 digest of the bytes read, and each adapter written against the
   actual release.
2. **`validate` did not load data.** It reported `valid` for paths that did not exist. Now it
   stats every configured path and normalizes a sample of every enabled suite; a path that
   exists but holds an unnormalizable record fails naming the record number.
3. **The manifest's version could disagree with the records scored.** `version` is now an
   explicit human label carried alongside a derived `content_digest` and `record_count`, and a
   configuration may pin an `expected_digest` that aborts the run on mismatch.
4. **Standalone install and root pytest collection.** Fixed separately in `df5e069`, the merge
   base of this work.

## The design change underneath

A task profile **declares** which capabilities are scored. The runner used to infer a metric's
applicability from whether a field happened to be populated (`if case.evidence_ids:`), so a
dataset without evidence identifiers silently dropped the retrieval metrics and reported the
rest — a result that looks complete and is not. A metric absent from a report is now absent
because the profile does not support it.

Adapters select a profile **per record** where the data genuinely differs: LongMemEval's
abstention questions have no answering turn to point at, and RGB's base file mixes single- and
multi-answer questions.

## Verified against real data, not fixtures

`conformance` was run over every record of every released source:

| Suite | Records | Result |
| --- | --- | --- |
| `hoh` | 111,972 | ok |
| `vitaminc` | 55,197 | ok |
| `factlens` | 733 | ok |
| `longmemeval` | 500 | ok |
| `rgb_base` | 300 | ok |
| `rgb_integration` | 100 | ok |
| `rgb_counterfactual` | 100 | ok |

168,902 real records normalized, each run recording a content digest.

## Tests

`cd benchmarks && pytest` — 180 passed, 0 skipped.
Repository root `pytest` — 584 passed, 1 pre-existing unrelated skip (`kiwipiepy`).

## Defects the work found in its own plan

Recorded because they are the substance, not incidents. Each was measured, not inferred.

- **LongMemEval was unloadable.** `memory_qa` declared `fine_retrieval` mandatory, but 21 of 500
  real records have zero answering turns — correctly, since an abstention question has none.
  Fixed by per-record profile selection.
- **FactLens would have rejected 68% of its benchmark.** Its `labels` column mixes `str` and
  `bool`; 502 of 733 rows carry at least one non-string.
- **`rgb_base` silently over-scored multi-answer questions.** 12 of 300 real records ask for
  several distinct answers; flattening them into one alias list gave full credit for answering
  part of the question.
- **A shared helper hardcoded one key name**, so any other key silently got the wrong validation.
- **A config-level error produced 111,972 identical per-record failures**, burying its own cause.

The full decision record, including every controller ruling and its reasoning, was kept in the
session's SDD ledger.
