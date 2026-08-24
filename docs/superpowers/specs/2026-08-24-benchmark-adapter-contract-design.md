# Benchmark Adapter Contract Redesign

## Why

The `benchmarks/` workspace landed with adapters written against a contract no
released dataset uses. Every adapter required a per-record `source_version` and
`split`, and the shared loader read only JSONL. The suite passed 41 tests
because the fixtures were authored to match the adapters rather than the
releases.

The five upstream formats were then read directly. Findings, each confirmed
against real bytes rather than documentation alone:

| Suite | Container | Record id | Evidence model | Data license |
| --- | --- | --- | --- | --- |
| LongMemEval | JSON array | `question_id` | `answer_session_ids` plus turn-level `has_answer` | MIT |
| HoH | Parquet | **none** | current `evidence` vs `outdated_infos[].evidence` | Apache-2.0 |
| VitaminC | JSONL | `unique_id` | inline `evidence` string, no corpus | CC-BY-SA-3.0 |
| RGB | JSONL (`.json` extension) | `id` (integer) | `positive`/`negative` document text, no ids; schema differs per variant | NOASSERTION |
| FactLens | CSV | `ind` | none — sub-claim decomposition | BSD-3-Clause |

Sources: `xiaowu0162/longmemeval-cleaned`, `russwest404/HoH-QAs`,
`tals/vitaminc` (all Hugging Face), `chen700564/RGB` and `megagonlabs/factlens`
(GitHub).

Four container formats. Not one of the five carries a per-record
`source_version` or `split`. Three of five have no upstream evidence
identifiers. Two have no retrieval task at all.

## Goal

Rebuild the adapter contract so the suite normalizes the released formats, and
so a run cannot report a metric it did not actually measure. Keep the existing
boundaries intact: no vendored dataset data, offline after acquisition, and
`wiki-eval` untouched as the vault-native regression source of truth.

## Non-goals

Downloading licensed data in CI, publishing benchmark claims, wiring these
metrics into `wiki-gate`, and integrating the live LLM Wiki retrieval path. The
runner keeps consuming recorded predictions.

RGB's negative-rejection protocol is also out of scope. It is produced by
feeding a model only negative documents (`--noise_rate 1`) and measuring
refusal, so it belongs to the prediction step, which does not exist yet.

## Design

### 1. Task profiles declare capabilities

`runner._score_case` currently infers what to measure from the data:

```python
if case.evidence_ids:
    scores.update(score_retrieval(...))
```

A dataset with no evidence identifiers therefore drops the retrieval metrics and
reports the rest, producing a result that looks complete and is not. This is the
structural twin of the fixture problem: the code adapts to whatever it is given
instead of asserting what it requires.

Each case declares a `profile`, and each profile declares the capabilities it
supports. Metrics are computed per declared capability. A case whose profile
declares a capability but lacks the data for it is an error, not a silent skip.

| Profile | Capabilities | Suite |
| --- | --- | --- |
| `memory_qa` | `retrieval` (two granularities), `answer`, `abstention` | LongMemEval |
| `retrieval_qa` | `retrieval`, `answer`, `citations` | `rgb_base` |
| `multi_slot_retrieval_qa` | `retrieval`, `multi_slot_answer`, `citations` | `rgb_integration` |
| `counterfactual_qa` | `retrieval`, `answer`, `distractor_rejection` | `rgb_counterfactual` |
| `temporal_discrimination` | `answer`, `distractor_rejection` | HoH |
| `grounded_verification` | `label` | VitaminC |
| `claim_decomposition` | `sub_claim_labels` | FactLens |

The report names the capabilities scored for the run. A metric absent from the
report is absent because the profile does not support it, never because the data
was missing.

### 2. Provenance moves out of the record

`source_version` and `split` leave the record contract.

`split` comes from the source selection in configuration: a file variant for
LongMemEval (`_s_cleaned`, `_m_cleaned`, `_oracle`) and RGB (`en` or `en_refine`
for `rgb_base`, `en_int`, `en_fact`, and the `zh` equivalents), or a released
split file for VitaminC (`train`, `dev`, `test`). HoH and FactLens ship one file each; their
`split` is a configured label defaulting to the source file stem, so the field
stays non-blank without inventing a partition that does not exist.

Version becomes three recorded facts rather than one asserted string:

- `version` — the label declared in configuration, for human reference only.
- `content_digest` — SHA-256 of the source file, computed at load.
- `record_count` — records read.

The manifest records all three. When configuration pins an `expected_digest`,
a mismatch fails the run. This closes the defect where a manifest could name a
release the records did not come from: the digest is derived from the bytes
scored, so it cannot disagree with them.

### 3. Container readers separate from field mapping

`adapters/base.py` splits. A new `readers.py` provides `read_json_array`,
`read_jsonl`, `read_csv`, and `read_parquet`, each yielding
`(record_number, record)` pairs and reporting malformed input with file and
record number. An adapter declares its reader and implements `normalize`.

`read_parquet` needs `pyarrow`, which is an optional extra (`benchmarks[hoh]`).
The repository's CI is deliberately dependency-light, so HoH support is opt-in;
without `pyarrow` the HoH adapter fails with an actionable message rather than
an `ImportError`. VitaminC needs no Hugging Face `datasets` dependency because
its released `train`/`dev`/`test.jsonl` files download directly.

### 4. Synthesized evidence identifiers are labeled

RGB, HoH, and FactLens have no upstream document identifiers. Their adapters
synthesize stable ones from the upstream field the documents came from — RGB
uses `{case_id}:positive:{index}` and `{case_id}:negative:{index}` — and emit
`context` in the same order, so a predictor returning document positions can be
scored. The pattern is the adapter's own; only the labeling below is shared.

The manifest records `evidence_id_origin` as `upstream` or `synthesized` per
suite. A retrieval score computed over adapter-invented identifiers is a
different claim from one computed over the dataset's own, and the artifact says
which it is.

### 5. Dataset-specific structure the old contract could not express

**LongMemEval abstention.** A `question_id` ending in `_abs` marks an
abstention question. The case sets `expects_abstention`, and the `abstention`
capability scores abstention precision, recall, and F1. The original design
required these metrics; the implementation omitted them because the contract had
nowhere to put the signal.

**LongMemEval two-level evidence.** `answer_session_ids` gives session-level
evidence; the per-turn `has_answer` flag gives turn-level evidence. Both are
retained — `evidence_ids` for sessions and `fine_evidence_ids` for
`{session_id}:{turn_index}` — and recall is reported at both levels.

**HoH has no record id.** The identifier is synthesized from `document.id` and
the record number. Because the record number participates, identifiers are
stable only for a fixed file; the digest in the manifest is what makes that
stability checkable.

**HoH is not multi-hop QA.** Its task is discriminating current evidence from
the outdated variants in `outdated_infos`. The `temporal_discrimination` profile
scores answer correctness against the current `answer`, and
`distractor_rejection` as the rate at which the prediction matched none of the
`outdated_infos[].answer` values. Reproducing an outdated answer is a distinct
failure from being merely wrong, and is reported separately.

**RGB is three suites, not one.** Its variants do not share a schema, so one
adapter cannot serve them without branching on the data — the failure mode this
redesign exists to remove. Measured shapes:

| Suite | File | `answer` | `positive` | Extra fields |
| --- | --- | --- | --- | --- |
| `rgb_base` | `en`, `en_refine` | `list[list[str]]` — slots of aliases | `list[str]` | — |
| `rgb_integration` | `en_int` | `list[list[str]]` — two slots | `list[list[str]]` — grouped per sub-question | `asnwer1` (upstream typo, sic), `answer2` |
| `rgb_counterfactual` | `en_fact` | `str` | `list[str]` | `fakeanswer`, `positive_wrong` |

`en_refine` is the corrected edition of `en` with the same schema, not a separate
task. `rgb_counterfactual` scores `distractor_rejection` against `fakeanswer`,
which makes it structurally the same measurement HoH needs.

**RGB context is a pool, not a context.** RGB supplies `positive` and `negative`
document pools; the upstream harness assembles a context from them at a chosen
`noise_rate` and `passage_num`. Those are run parameters of the prediction step,
not properties of the dataset. Since this suite scores recorded predictions and
never calls a model, the adapters emit both pools with labeled identifiers and
leave assembly to whatever produced the predictions. `noise_rate` and
`passage_num` are recorded in the manifest as declared values, for
comparability, and are not used to build anything.

**VitaminC labels carry spaces.** The released labels are `SUPPORTS`,
`REFUTES`, and `NOT ENOUGH INFO`. The current adapter accepts only
`NOT_ENOUGH_INFO` and rejects every real record of that class. The mapping keys
on the released spelling.

**FactLens has no evidence.** Its `sub_claims` and `labels` columns hold
Python-repr lists of strings, not JSON, and must be parsed as such. The profile
scores per-sub-claim labels and the aggregate.

### 6. `validate` loads data; `conformance` reads all of it

`validate` currently inspects YAML only and reports `valid` for dataset paths
that do not exist. It gains: path existence, reader construction, digest
computation, and normalization of the first N records (default 5).

A new `conformance` subcommand normalizes **every** record in a configured
source, reports each failure with its record number, and checks any pinned
digest. It is the manual gate run against real local data; `validate` stays fast
enough for routine use.

### 7. Fixtures follow the released shapes

The fixtures are rewritten to the real schema and the real container for each
suite — `longmemeval.json` as a JSON array, `vitaminc.jsonl`, one JSONL fixture
per RGB variant, `factlens.csv` — with synthetic content. Synthetic content keeps the repository
clear of CC-BY-SA-3.0 and NOASSERTION data while the *shape* is no longer
invented.

The HoH Parquet fixture is generated at test time rather than committed, so no
binary lands in the repository and the test skips cleanly when `pyarrow` is
absent.

## Testing

Per-suite adapter tests assert normalization against the real-shaped fixtures
and rejection of malformed records with file and record number. Profile tests
assert that a declared capability with missing data raises rather than degrades.
Reader tests cover each container, including the `.json`-extension-but-JSONL
case that RGB presents. Each RGB variant gets its own adapter test, including
the `list[list[str]]` answer slots and the upstream `asnwer1` typo. Manifest tests assert digest, record count, and
`evidence_id_origin` are recorded, and that a pinned-digest mismatch fails.
`validate` tests assert a nonexistent path is now rejected.

No test downloads upstream data, imports the ML stack, or requires network.

## Corrections to the prior spec

`docs/superpowers/specs/2026-08-24-external-benchmark-suite-design.md` is
superseded on three points, each contradicted by the released data:

- ~~HoH: multi-hop evidence retrieval/QA~~ — HoH is outdated-versus-current
  evidence discrimination. It has no hop structure.
- ~~`validate` checks file presence, upstream-record shape, normalized-case
  invariants~~ — described but never implemented; section 6 implements it.
- ~~Tests use only the synthetic fixtures~~ — correct as a policy, but the
  fixtures encoded an invented schema. Synthetic *content* with released
  *shape* is the working form.

## Risks

Writing five adapters against five formats in one change is what failed the
first time. The mitigation is that all five sources are now downloadable, so
each adapter is verified by `conformance` against real bytes before it is
claimed to work. An adapter that has not been run against its real source is
reported as unverified rather than done.

RGB carries no license declaration (`NOASSERTION`). Its data stays out of the
repository and the README states the condition rather than assuming permission.
