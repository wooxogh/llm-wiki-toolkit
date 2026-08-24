# RESUME — benchmark adapter contract redesign

Paused 2026-08-24 at the user's request, mid-execution.

## Where things stand

**Worktree:** `/Users/taeho-woo-ilevit-com/dev/llm-wiki-toolkit/.worktrees/external-benchmark-suite`
**Branch:** `refactor(benchmarks)/adapter-contract`, branched from PR #6's head `fe54520`
**HEAD:** `4c415a2`
**Working tree:** clean, nothing uncommitted, nothing pushed yet
**Suite:** `cd benchmarks && pytest -q` → 102 passed, 6 failed. **The 6 failures are expected**, not regressions: `test_registry` (3), `test_runner` (2), `test_cli` (1) — owned by Tasks 12, 14 and 15, which have not run yet.

**Spec:** `docs/superpowers/specs/2026-08-24-benchmark-adapter-contract-design.md`
**Plan:** `docs/superpowers/plans/2026-08-24-benchmark-adapter-contract.md` (16 tasks)
**Ledger:** `progress.md` in this directory — the authoritative record. Trust it and `git log` over anything remembered.

## Task status

| Task | State | Commits |
| --- | --- | --- |
| 1 profiles + capabilities | complete, review clean | `417bf11` |
| 2 container readers + digest | complete, clean after 2 fix rounds | `3f15764`..`d48d2b7` |
| 3 case schema carries profile | complete, review clean | `c952d93` |
| 4 adapter base derives provenance | complete, review clean | `8ef88a6` |
| — amendment: per-record profile | reviewed inside Task 5's review | `823f732`, `dc0fc58` |
| 5 LongMemEval | complete, clean after 2 fix rounds | `172e0b7`..`5e97d0b` |
| 6 VitaminC | complete, review clean | `a0cc127` |
| 7 RGB base + shared helpers | complete, clean after 2 fix rounds | `59df40c`..`3d2e225` |
| 8 RGB integration | complete, review clean | `2731058` |
| **9 RGB counterfactual** | **implemented, NOT REVIEWED** | `4c415a2` |
| 10 HoH (Parquet) | not started | — |
| 11 FactLens (CSV) | not started | — |
| 12 registry, 7 suites | not started | — |
| 13 new metrics | not started | — |
| 14 runner: capability scoring | not started | — |
| 15 validate + conformance | not started | — |
| 16 config, README, full verify | not started | — |

## Resume here

1. **Dispatch Task 9's task review first.** It is implemented but ungated. Build the package with
   `scripts/review-package docs/superpowers/plans/2026-08-24-benchmark-adapter-contract.md 2731058 4c415a2`
   and review against `task-9-brief.md` PLUS the three corrections recorded in the ledger (explicit `required=` on all three `require_documents` calls; a test for the list-shaped `answer`; the `__init__.py`/`registry.py` appends). Do not mark it complete without that review.
2. Then continue with Tasks 10–16 in order.
3. The plan's briefs are already extracted as `task-N-brief.md` in this directory. **Several are stale** — see "Corrections the briefs do not contain" below. Read that section before dispatching any of them.
4. Follow `superpowers:subagent-driven-development`: fresh implementer per task, task review after each, scoped re-review per fix round, ledger line per event.

## Corrections the briefs do not contain

These were discovered during execution and must be carried into the relevant dispatches; the brief files were NOT rewritten.

- **`require_documents` signature changed** (Task 7). It is now `require_documents(record, key, path, record_number, *, required: bool)` with **no default**. Any brief showing a call without `required=` is wrong and will raise `TypeError`. Affects Tasks 8 and 9 (already handled) and any later reuse.
- **Adapters select a profile per record** (Task 5 amendment). `normalize` may return a `profile` key; `load` validates it and falls back to the class attribute. A new profile `memory_qa_abstention` = `{retrieval, answer, abstention}` exists. **Task 14's manifest must record the SET of profiles observed in a run, not a single profile**, and its per-case scoring must dispatch on `case.profile` rather than the adapter's class attribute. The plan's Task 14 text predates this and is wrong on that point.
- **FactLens `labels` mixes types** (measured, before Task 11 ran). In the real 733-row CSV: `'true'` str ×518, `'false'` str ×175, `True` bool ×281, `False` bool ×221 — **502 of 733 rows contain a non-str label**. The brief's `_repr_list` requires every item to be a `str` and would reject 68% of the benchmark. Ruling: accept both `str` and `bool` and canonicalise to lowercase `'true'`/`'false'`; `sub_claims` stays strings-only (measured all str, never empty, lengths never mismatched). A test must pin a mixed-type row.
- **`en_fact`'s `answer` is not always a bare string** (measured). `str` in 72 of 100 records, `list` in 28. The spec said otherwise. `flatten_answers` handles both; Task 9 was told to add the list-shape test.
- **Task 14/15 tests use repo-root-relative fixture paths** (`FIXTURES = "benchmarks/fixtures"`), but the plan runs tests with `cd benchmarks`. Ruling: resolve fixtures as `Path(__file__).parent.parent / "fixtures"` and pass absolute paths into config dicts.
- **Task 15 must rewrite any pre-existing `test_cli.py` case that feeds `suite.example.yaml`.** That config now points at `benchmarks/data/...` files the repo deliberately does not carry, so validating it must FAIL — that is the fix, not a regression.
- **`pyarrow` goes in both the `hoh` extra and `dev`** (Task 10), so the HoH tests actually run in CI rather than skipping.
- **Task 16 README needs a memory note.** `read_json_array` and `read_parquet` slurp, and stdlib `json` has no incremental array parser. LongMemEval's `longmemeval_s_cleaned.json` is 277 MB and HoH is 111,972 records; the practical LongMemEval file for this suite is `longmemeval_oracle.json` (15 MB).

## Real-data measurements already done (do not redo)

Downloaded copies live in the session scratchpad, which **will not survive**; the numbers are what matter.

| Source | Scanned | Result |
| --- | --- | --- |
| LongMemEval `longmemeval_oracle.json` | all 500 records | 30 `_abs`; **21 have zero `has_answer` turns**; `answer_session_ids` never empty. Drove the per-record-profile amendment. Adapter verified over all 500: 479 `memory_qa` / 21 `memory_qa_abstention`. |
| HoH `hoh_qas_240601_241201.parquet` | all 111,972 records | No defect. `outdated_infos` never empty, `document.id` always present, `question`/`answer`/`evidence` never blank, every `outdated_infos` entry has `answer` + `evidence`. `last_modified_time` is a `datetime` (non-JSON-serializable). `document.id` yields only 90,571 distinct values, so it is NOT unique — combining it with the record number is required. |
| VitaminC | sample rows | Labels are `SUPPORTS` / `REFUTES` / **`NOT ENOUGH INFO`** (spaces). Released files `train/dev/test.jsonl`, so no HF `datasets` dependency needed. |
| RGB `en_refine.json` | all 300 | `positive` never empty, `answer` always list, `positive` items always str. |
| RGB `en_int.json` | all 100 | `positive` always list-of-lists, never empty. Answer slot counts: 2 ×94, 3 ×3, 4 ×1, 6 ×2 — all ≥2. |
| RGB `en_fact.json` | all 100 | `answer` str ×72 / list ×28. `fakeanswer` and `positive_wrong` always present. |
| FactLens `fact_lens_benchmark.csv` | all 733 | See the mixed-type `labels` finding above. |

Licenses: LongMemEval MIT, HoH Apache-2.0, VitaminC CC-BY-SA-3.0, RGB **no declaration**, FactLens BSD-3-Clause. No dataset content is committed; fixtures are synthetic in the released shape.

## Deferred minors for the final whole-branch review

- `vitaminc.py`'s `_CONSUMED` is unused and inaccurate; six sibling adapters use `_metadata(record, _CONSUMED)` while VitaminC hand-writes its metadata dict. Decide the family-wide convention.
- VitaminC drops `FEVER_id` and `big_bench_canary` instead of keeping them under `source_fields`.
- No test or fixture exercises >2 answer slots in `rgb_integration`, though 6 of 100 real records have 3, 4 or 6.
- `_flatten_positive` tolerates an empty sub-list inside a grouped `positive` while rejecting a bare `[]` at top level.
- `profiles._REQUIREMENTS` field-path strings are not validated against real `BenchmarkCase` attributes.
- `_has_requirement` treats every `bool` as present, so the `abstention` capability's per-case check is vacuous; abstention correctness rests entirely on Task 5's `_abs` tests.
- `_metadata`'s `source_fields` copies raw upstream values with no coercion — a `NaN` or a pyarrow/numpy scalar could crash a report written with `allow_nan=False`. Loud failure, not a silent wrong number.
- The malformed-CSV test writes a 200 KB field each run; the NUL-byte trigger is cheaper.
- `retrieval` and `citations` both map to `("evidence_ids",)` — identical tuples read as possible copy-paste.
- Minor regex hygiene: unescaped `.` in some `pytest.raises(match=...)` patterns; one test embeds a filesystem path in a regex.

## Not done, and deliberately so

- **Nothing pushed.** The branch is local only.
- **No PR opened** for this work. PR #6 (`feat(phase2-benefit)/external-benchmark-suite`) is still open with the packaging fix `df5e069` and is a separate deliverable awaiting the user's merge.
- **No adapter has been run through `conformance` against real data**, because `conformance` does not exist until Task 15. The plan's pre-PR checklist requires that, or an explicit "unverified against real data" statement in the PR body. LongMemEval is the exception: its adapter was run over all 500 real records.
