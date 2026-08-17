# v2 Incremental Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make repeated v2 builds process only changed documents, changed Concept vectors, and affected relation candidates while preserving provenance and fail-closed behavior.

**Architecture:** Thread the target vault through Concept extraction and cache access. Persist enough prior Concept/index state to merge unchanged records, then compute a dirty Concept set from source changes and reconcile only relation proposals involving that set. Full rebuild remains the safe fallback whenever artifact identity changes.

**Tech Stack:** Python 3.11+, dataclasses, JSONL artifacts, NumPy, pytest, existing v2 adapter/index/NET modules.

**Spec:** `docs/superpowers/specs/2026-08-17-v2-incremental-build-design.md`

## Global Constraints

- Markdown remains the canonical source; `.llm_wiki_v2/` is derived only.
- Exact `source_quote` provenance must be preserved for every Concept.
- `CONTRADICTS`, `SUPERSEDES`, and `OVERRIDES` remain human-approval-only.
- Model, prompt, schema, chunk-target, and vector-index identity changes force a full rebuild.
- Existing approved/rejected decisions and user-owned NET structure must survive incremental builds.
- Do not change the semantic model or introduce a hosted dependency.

---

### Task 1: Add failing tests for changed-document and vault-scoped extraction

**Files:**
- Modify: `tests/test_v2_concepts.py`
- Modify: `tests/test_v2_final_contract.py`
- Modify: `src/llm_wiki/v2/concept_extraction.py`
- Modify: `src/llm_wiki/v2/concept_store.py`

**Interfaces:**
- New extraction signature: `extract(chunk: Chunk, adapter: UserLLMAdapter, model_identity: str = "offline", vault: Path | None = None) -> list[ConceptProposal]`.
- Updated helper signature: `concepts_from_chunks(chunks, adapter=None, progress=None, vault=None) -> list[Concept]`.

- [ ] **Step 1: Write the failing tests.**

  Add a test adapter with a call counter and assert `build_concepts(vault, changed_only=True)` calls it only for a newly added or modified document. Add a second test that runs extraction for two temporary vaults with the same content and asserts each vault owns its own `cache/concepts/<key>.json` artifact.

- [ ] **Step 2: Run the focused tests to verify they fail.**

  Run: `pytest tests/test_v2_concepts.py tests/test_v2_final_contract.py -q`

  Expected: the changed-document test demonstrates the current all-document chunk/extraction path or cache behavior, and the vault-isolation test finds the cache under the process default instead of the target vault.

- [ ] **Step 3: Implement the minimal vault threading.**

  Pass `vault` from `build_concepts()` to `concepts_from_chunks()`, from there to `extract()`, and use `artifacts.artifact_path(..., vault)` for cache reads and writes. Keep the existing content-addressed key fields unchanged.

- [ ] **Step 4: Run the focused tests to verify they pass.**

  Run: `pytest tests/test_v2_concepts.py tests/test_v2_final_contract.py -q`

- [ ] **Step 5: Commit.**

  Run: `git add tests/test_v2_concepts.py tests/test_v2_final_contract.py src/llm_wiki/v2/concept_extraction.py src/llm_wiki/v2/concept_store.py && git commit -m "fix: scope v2 concept cache to target vault"`

### Task 2: Make chunk construction and Concept persistence incremental

**Files:**
- Modify: `src/llm_wiki/v2/concept_store.py`
- Modify: `tests/test_v2_final_contract.py`
- Modify: `tests/test_v2_hardening.py`

**Interfaces:**
- Internal helper: `_changed_document_ids(live_docs: list[Document], stored_docs: list[Document]) -> set[str]`.
- Internal helper: `_build_changed_chunks(vault: Path | None, docs: list[Document], changed_ids: set[str], target_chars: int) -> list[Chunk]`.

- [ ] **Step 1: Write the failing tests.**

  Add tests that monkeypatch or instrument `chunk_document` and verify an unchanged document is not chunked during `changed_only=True`. Add a deletion test asserting deleted document chunks and Concepts disappear from the returned and persisted artifacts. Add an equivalence test comparing a full build with an incremental build over the same final vault.

- [ ] **Step 2: Run the focused tests to verify they fail.**

  Run: `pytest tests/test_v2_final_contract.py tests/test_v2_hardening.py -q`

- [ ] **Step 3: Implement changed-only chunking.**

  In `build_concepts()`, collect live documents and compare their content hashes to stored documents before calling `build_chunks()`. For `changed_only=True`, call `chunk_document()` only for changed document paths, retain stored chunks for unchanged document IDs, and discard stored chunks/Concepts whose document IDs are no longer live. Preserve deterministic document/chunk ordering before writing JSONL.

- [ ] **Step 4: Run focused tests and inspect artifact counts.**

  Run: `pytest tests/test_v2_final_contract.py tests/test_v2_hardening.py -q`

  Confirm the test output shows no stale Concept references and that full/incremental final artifacts contain the same live documents and Concepts.

- [ ] **Step 5: Commit.**

  Run: `git add src/llm_wiki/v2/concept_store.py tests/test_v2_final_contract.py tests/test_v2_hardening.py && git commit -m "feat: reuse unchanged v2 chunks and concepts"`

### Task 3: Implement incremental Concept vector indexing

**Files:**
- Modify: `src/llm_wiki/v2/concept_index.py`
- Modify: `src/llm_wiki/v2/concepts_cli.py`
- Modify: `tests/test_v2_completion.py`
- Modify: `tests/test_v2_final_contract.py`

**Interfaces:**
- New index function: `build_index(vault: Path | None = None, show_progress: bool = False, changed_only: bool = False) -> int`.
- Internal helpers: `_index_identity(vault) -> dict`, `_load_previous_index(root) -> tuple[np.ndarray, list[dict], str | None]`, `_changed_concept_ids(previous_meta, concepts) -> set[str]`.

- [ ] **Step 1: Write the failing tests.**

  Add a test that builds an index, adds one Concept, rebuilds with `changed_only=True`, and asserts unchanged vector rows and metadata are retained while exactly one new row is embedded. Add a deletion test. Add an identity-change test asserting a full re-embed occurs when the model/index identity differs.

- [ ] **Step 2: Run the focused tests to verify they fail.**

  Run: `pytest tests/test_v2_completion.py tests/test_v2_final_contract.py -q`

- [ ] **Step 3: Implement incremental index merge.**

  Load the previous vectors and metadata, map Concept IDs to rows, reuse rows whose Concept ID and indexed text/hash are unchanged, embed only new or changed Concepts, omit deleted IDs, then rewrite `vectors.npy`, `meta.json`, and `model.txt` in current Concept order. Store or derive a stable index identity containing model ID, dimension, and indexed text schema. Force the existing full path when identity or shape is incompatible.

- [ ] **Step 4: Thread the CLI flag.**

  Pass `args.changed` from `wiki-concepts build --changed` to `build_index(..., changed_only=True)`. Keep ordinary `wiki-concepts build` as a full rebuild.

- [ ] **Step 5: Run focused tests and commit.**

  Run: `pytest tests/test_v2_completion.py tests/test_v2_final_contract.py -q`

  Commit with: `git add src/llm_wiki/v2/concept_index.py src/llm_wiki/v2/concepts_cli.py tests/test_v2_completion.py tests/test_v2_final_contract.py && git commit -m "feat: incrementally update v2 concept index"`

### Task 4: Reconcile only dirty relation candidates

**Files:**
- Modify: `src/llm_wiki/v2/net_builder.py`
- Modify: `src/llm_wiki/v2/relation_candidates.py`
- Modify: `tests/test_v2_net_review_query_health.py`
- Modify: `tests/test_v2_completion.py`

**Interfaces:**
- New `build_net` parameter: `build_net(vault=None, adapter=None, allow_ai_topic_creation=True, progress=None, changed_only=False)`.
- Internal helper: `_dirty_concept_ids(vault, concepts, previous_nodes) -> set[str]`.
- Internal helper: `_discover_relations(..., dirty_ids: set[str] | None = None)`.

- [ ] **Step 1: Write the failing tests.**

  Add a test that builds a two-Concept NET, adds a third Concept, runs incremental NET build, and asserts the adapter classifies pairs involving the third Concept while not reclassifying an unrelated existing pair. Add a test that an approved relation remains committed after the incremental build. Add a deleted-endpoint test that removes only invalid open proposals/edges.

- [ ] **Step 2: Run the focused tests to verify they fail.**

  Run: `pytest tests/test_v2_net_review_query_health.py tests/test_v2_completion.py -q`

- [ ] **Step 3: Implement dirty-set relation reconciliation.**

  Derive dirty IDs from added/removed/changed Concept IDs and prior NET node metadata. Preserve terminal proposals and approved edges. Remove only open proposals and unapproved relation edges touching dirty/deleted endpoints. Pass `dirty_ids` to relation discovery and classify candidates where either endpoint is dirty; continue using the existing candidate top-k discovery and approval gates.

- [ ] **Step 4: Thread the CLI flag and preserve full rebuild behavior.**

  Pass `--changed` through `wiki-net build`; if false, retain current all-Concept relation discovery. Ensure user-owned topic/collection placement and operation logs remain unchanged.

- [ ] **Step 5: Run focused tests and commit.**

  Run: `pytest tests/test_v2_net_review_query_health.py tests/test_v2_completion.py -q`

  Commit with: `git add src/llm_wiki/v2/net_builder.py src/llm_wiki/v2/relation_candidates.py tests/test_v2_net_review_query_health.py tests/test_v2_completion.py && git commit -m "feat: reconcile only dirty v2 relations"`

### Task 5: Add end-to-end health and regression coverage

**Files:**
- Modify: `src/llm_wiki/v2/health.py`
- Modify: `tests/test_wiki_health.py`
- Modify: `tests/test_v2_hardening.py`
- Modify: `docs/V2_USAGE_KO.md`

**Interfaces:**
- No public API changes beyond the `--changed` flags from Tasks 3 and 4.

- [ ] **Step 1: Write the failing tests.**

  Add an end-to-end synthetic-vault test that runs full build, adds one document, runs changed builds, and asserts `check_v2_health(vault) == []`. Add tests that stale index identity, stale source quote, and deleted source artifacts still block query after an incremental build. Add CLI documentation assertions only through behavior tests, not string matching.

- [ ] **Step 2: Run the focused tests to verify they fail.**

  Run: `pytest tests/test_wiki_health.py tests/test_v2_hardening.py -q`

- [ ] **Step 3: Implement any required health identity checks.**

  Extend health only for the new incremental index identity and dirty relation bookkeeping. Keep existing error wording and fail-closed query behavior where possible; do not downgrade stale conditions to warnings.

- [ ] **Step 4: Document the incremental workflow.**

  Update `docs/V2_USAGE_KO.md` with the commands `wiki-concepts build --changed` and `wiki-net build --changed`, explain what is reused, and state which model/schema changes force a full rebuild.

- [ ] **Step 5: Run the complete verification suite.**

  Run: `pytest -q`

  Then run: `python -m compileall -q src tests`, `wiki-health --mode ci`, and the v2 synthetic smoke command from the test suite. Record the final pass count and any environment limitation before claiming completion.

- [ ] **Step 6: Commit.**

  Run: `git add src/llm_wiki/v2/health.py tests/test_wiki_health.py tests/test_v2_hardening.py docs/V2_USAGE_KO.md && git commit -m "test: verify v2 incremental build integrity"`

## Final Review Checklist

- [ ] Full rebuild and incremental rebuild produce equivalent final canonical artifacts.
- [ ] Unchanged documents are not sent to the Concept adapter.
- [ ] Unchanged Concept vectors are reused.
- [ ] Deleted documents cannot remain searchable.
- [ ] Approved risky relations are never silently removed or rewritten.
- [ ] Stale identity/provenance still blocks v2 query.
- [ ] `pytest -q` passes, or the exact missing dependency/environment limitation is reported.
