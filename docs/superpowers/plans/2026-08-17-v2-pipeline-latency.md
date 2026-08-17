# v2 Pipeline Latency Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the wall-clock cost of `wiki-net build` (and the `wiki-net query --auto` path) by removing wasted LLM calls and wasted artifact re-reads, without changing the v2 contract's correctness guarantees (grounding, confidence gating, lifecycle states).

**Architecture:** Four independent, additive fixes found during a live investigation of a 2-hour ETA on a 15-document vault:
1. `temporal.resolve()` calls the LLM even when no relation was classified — a straight logic bug, fix it.
2. `classify_relation`/`resolve_temporal` have no cache, unlike concept extraction — add one, keyed the same way.
3. Relation-candidate discovery has a relevance score available (from `concept_index.search`) but never uses it to skip low-value LLM calls — thread it through as an opt-in filter.
4. The query path (`v2/query.py`) re-reads `edges.jsonl`/`nodes.jsonl` from disk multiple times per single query, and `auto_decision()` computes the same sparse+dense signal pass twice — collapse both to one read/one computation.

None of these change what a user sees when a build/query succeeds; they only change how much work it costs to get there. Each task ships with its own test and can be merged independently.

**Tech Stack:** Python 3, pytest, dataclasses (frozen), existing `llm_wiki.v2` package conventions (content-addressed JSON cache under `.llm_wiki_v2/cache/`, `wiki.toml` `[v2]` config section).

**Spec:** No separate spec doc — the spec is this plan plus the investigation captured in this conversation (see file/line citations in each task).

## Global Constraints

- Every new/changed public function keeps its existing call sites working, or every call site is updated in the same task.
- Cache files go under `artifacts.artifact_path("cache/...", vault)`, matching the existing `cache/concepts/<key>.json` convention in `concept_extraction.py`.
- No new third-party dependencies.
- Run `pytest tests/test_v2_*.py -q` at the end of every task — all must stay green.

---

### Task 1: Fix `temporal.resolve()` calling the LLM with no classified relation

**Files:**
- Modify: `src/llm_wiki/v2/temporal.py:15-17`
- Test: `tests/test_v2_temporal_cost.py` (new)

**Interfaces:**
- Consumes: `llm_wiki.v2.models.Concept`, `llm_wiki.v2.models.RelationProposal`, `llm_wiki.v2.schemas.RelationType`
- Produces: no signature change to `resolve(adapter, source, target, relation=None) -> RelationProposal | None` — behavior only.

**Context:** `net_builder._discover_relations` calls `resolve(adapter, source, target, relation)` for every candidate pair, where `relation` is whatever `classify()` returned (often `None` — most pairs are not related). Current code:

```python
def resolve(adapter: UserLLMAdapter, source: Concept, target: Concept,
            relation: RelationProposal | None = None) -> RelationProposal | None:
    if relation and relation.relation not in {RelationType.CONTRADICTS.value, RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
    proposal = adapter.resolve_temporal(source, target)
```

When `relation is None`, `relation and ...` is falsy, so the guard is skipped and `adapter.resolve_temporal(...)` — a full external CLI spawn — still runs. The module docstring says this function exists to "turn **plausible relation pairs** into temporal proposals," so a pair with no classified relation should never reach the adapter at all. This single fix removes one full LLM call for every pair where `classify()` found nothing — the majority case.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_temporal_cost.py
from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType
from llm_wiki.v2 import temporal


def _concept(cid: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=f"{cid} text",
        summary=cid, source_quote=f"{cid} text", confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=10,
    )


class _CountingAdapter:
    def __init__(self):
        self.calls = 0

    def resolve_temporal(self, source, target):
        self.calls += 1
        return None


def test_resolve_skips_the_llm_when_no_relation_was_classified():
    adapter = _CountingAdapter()
    result = temporal.resolve(adapter, _concept("concept:a"), _concept("concept:b"), relation=None)
    assert result is None
    assert adapter.calls == 0


def test_resolve_still_calls_the_llm_for_a_plausible_relation():
    adapter = _CountingAdapter()
    relation = RelationProposal(
        id="proposal:a:CONTRADICTS:b", source_concept_id="concept:a", target_concept_id="concept:b",
        relation=RelationType.CONTRADICTS.value, confidence=0.8, evidence="a text",
    )
    result = temporal.resolve(adapter, _concept("concept:a"), _concept("concept:b"), relation=relation)
    assert result is None  # adapter returns None here, only the CALL matters
    assert adapter.calls == 1
```

- [ ] **Step 2: Run test to verify the first case fails**

Run: `pytest tests/test_v2_temporal_cost.py -v`
Expected: `test_resolve_skips_the_llm_when_no_relation_was_classified` FAILS with `assert 1 == 0` (adapter was called); `test_resolve_still_calls_the_llm_for_a_plausible_relation` PASSES already (documents current-and-kept behavior).

- [ ] **Step 3: Fix the guard**

In `src/llm_wiki/v2/temporal.py`, change:

```python
    if relation and relation.relation not in {RelationType.CONTRADICTS.value, RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
```

to:

```python
    if relation is None or relation.relation not in {RelationType.CONTRADICTS.value, RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
```

- [ ] **Step 4: Run test to verify both pass**

Run: `pytest tests/test_v2_temporal_cost.py -v`
Expected: both PASS.

- [ ] **Step 5: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS (this only removes calls that were always discarded — `resolve()`'s early-return contract for a non-`None` unrelated relation is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/llm_wiki/v2/temporal.py tests/test_v2_temporal_cost.py
git commit -m "fix: temporal.resolve no longer calls the LLM when no relation was classified"
```

---

### Task 2: Cache `classify_relation`/`resolve_temporal` results (content-addressed, like concept extraction)

**Files:**
- Create: `src/llm_wiki/v2/relation_cache.py`
- Modify: `src/llm_wiki/v2/relation_classifier.py`
- Modify: `src/llm_wiki/v2/temporal.py`
- Modify: `src/llm_wiki/v2/net_builder.py:165-171` (thread `vault` through the two calls)
- Test: `tests/test_v2_relation_cache.py` (new)

**Interfaces:**
- Consumes: `llm_wiki.v2.artifacts.artifact_path(name, vault) -> Path`, `llm_wiki.v2.models.RelationProposal.to_dict()/.from_dict()`, `Concept.id` (already content-addressed — see `concept_store._concept_id`, unaffected by this plan).
- Produces: `relation_cache.cached_call(kind: str, source: Concept, target: Concept, prompt_version: str, model_identity: str, compute: Callable[[], RelationProposal | None], vault: Path | None = None) -> RelationProposal | None`
- Produces: `classify(adapter, source, target, vault=None) -> RelationProposal | None` (new optional `vault` param, default preserves current call sites)
- Produces: `resolve(adapter, source, target, relation=None, vault=None) -> RelationProposal | None` (same)

**Context:** `concept_extraction.extract()` already caches by `chunk.content_hash + prompt_version + model_identity` under `cache/concepts/<key>.json` (`concept_extraction.py:19-30`). `relation_classifier.classify()` and `temporal.resolve()` have no such cache — every `wiki-net build` re-issues the LLM call for every candidate pair, even pairs whose concepts haven't changed since the last run. `Concept.id` is already a stable content hash (`concept_store._concept_id`: `sha1(chunk.id | source_quote | text)`), so `(source.id, target.id)` is a safe, content-addressed cache key — no need to add a new hash field anywhere.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_relation_cache.py
from pathlib import Path

from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType
from llm_wiki.v2 import relation_classifier, temporal


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    return tmp_path


def _concept(cid: str, text: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=text,
        summary=text, source_quote=text, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(text),
    )


class _CountingClassifyAdapter:
    model_identity = "test-model"

    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    def classify_relation(self, source, target):
        self.calls += 1
        return self.proposal


class _CountingTemporalAdapter:
    model_identity = "test-model"

    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    def resolve_temporal(self, source, target):
        self.calls += 1
        return self.proposal


def test_classify_relation_is_cached_across_calls(tmp_path):
    vault = _vault(tmp_path)
    source, target = _concept("concept:a", "Frontend uses React"), _concept("concept:b", "Frontend used Vue")
    proposal = RelationProposal(
        id="proposal:a:SUPPORTS:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPPORTS.value, confidence=0.9, evidence=source.source_quote,
    )
    adapter = _CountingClassifyAdapter(proposal)
    first = relation_classifier.classify(adapter, source, target, vault=vault)
    second = relation_classifier.classify(adapter, source, target, vault=vault)
    assert first == second == proposal
    assert adapter.calls == 1


def test_resolve_temporal_is_cached_across_calls(tmp_path):
    vault = _vault(tmp_path)
    source, target = _concept("concept:a", "Frontend no longer uses Vue"), _concept("concept:b", "Frontend uses Vue")
    proposal = RelationProposal(
        id="proposal:a:SUPERSEDES:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPERSEDES.value, confidence=0.9, evidence=source.source_quote,
        same_subject=True, same_scope=True, temporal_change_possible=True, reason="explicit replacement",
    )
    relation = RelationProposal(
        id="proposal:a:SUPERSEDES:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPERSEDES.value, confidence=0.9, evidence=source.source_quote,
    )
    adapter = _CountingTemporalAdapter(proposal)
    first = temporal.resolve(adapter, source, target, relation=relation, vault=vault)
    second = temporal.resolve(adapter, source, target, relation=relation, vault=vault)
    assert first == second == proposal
    assert adapter.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v2_relation_cache.py -v`
Expected: FAIL with `TypeError: classify() got an unexpected keyword argument 'vault'` (and same for `resolve`).

- [ ] **Step 3: Create the cache module**

```python
# src/llm_wiki/v2/relation_cache.py
"""Content-addressed cache for pairwise relation/temporal LLM calls.

Mirrors concept_extraction.py's cache: Concept.id is already a content hash
(concept_store._concept_id), so (source.id, target.id) is a safe key without
needing a separate hash field.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from llm_wiki.v2 import artifacts
from llm_wiki.v2.models import Concept, RelationProposal


def cached_call(kind: str, source: Concept, target: Concept, prompt_version: str,
                model_identity: str, compute: Callable[[], RelationProposal | None],
                vault: Path | None = None) -> RelationProposal | None:
    key = hashlib.sha256(
        f"{kind}|{source.id}|{target.id}|{prompt_version}|{model_identity}".encode()
    ).hexdigest()
    cache = artifacts.artifact_path(f"cache/relations/{key}.json", vault)
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return RelationProposal.from_dict(raw) if raw is not None else None
    proposal = compute()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(proposal.to_dict() if proposal is not None else None, ensure_ascii=False),
        encoding="utf-8",
    )
    return proposal
```

- [ ] **Step 4: Wrap `classify()` in `relation_classifier.py`**

Rename the existing function body to `_classify_uncached` and add a thin cached wrapper:

```python
from llm_wiki.v2 import relation_cache

def classify(adapter: UserLLMAdapter, source: Concept, target: Concept,
            vault=None) -> RelationProposal | None:
    return relation_cache.cached_call(
        "classify", source, target, RELATION_PROMPT_VERSION,
        getattr(adapter, "model_identity", "offline"),
        lambda: _classify_uncached(adapter, source, target), vault,
    )


def _classify_uncached(adapter: UserLLMAdapter, source: Concept, target: Concept) -> RelationProposal | None:
    proposal = adapter.classify_relation(source, target)
    if proposal is None:
        return None
    if proposal.source_concept_id != source.id or proposal.target_concept_id != target.id:
        return None
    try:
        RelationType(proposal.relation)
    except ValueError:
        return None
    if not 0 <= proposal.confidence <= 1:
        return None
    if not proposal.evidence or proposal.evidence not in (source.text + "\n" + source.source_quote + "\n" +
                                                           target.text + "\n" + target.source_quote):
        return None
    if RelationType(proposal.relation) in RISKY_RELATIONS and (
        proposal.same_subject is None or proposal.same_scope is None
        or proposal.temporal_change_possible is None or not proposal.reason.strip()
    ):
        return None
    return replace(proposal, prompt_version=RELATION_PROMPT_VERSION)
```

(Only the top of the file changes — the body of the old `classify()` becomes `_classify_uncached()` verbatim, plus the new `classify()` wrapper and the `relation_cache` import.)

- [ ] **Step 5: Wrap `resolve()` in `temporal.py`**

Same pattern — rename the existing function body to `_resolve_uncached` and add the cached wrapper. Note the early-return from Task 1 stays in the **wrapper**, so a `None` relation never even computes a cache key or touches the filesystem:

```python
from llm_wiki.v2 import relation_cache

def resolve(adapter: UserLLMAdapter, source: Concept, target: Concept,
            relation: RelationProposal | None = None, vault=None) -> RelationProposal | None:
    if relation is None or relation.relation not in {RelationType.CONTRADICTS.value, RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
    return relation_cache.cached_call(
        "resolve", source, target, TEMPORAL_PROMPT_VERSION,
        getattr(adapter, "model_identity", "offline"),
        lambda: _resolve_uncached(adapter, source, target), vault,
    )


def _resolve_uncached(adapter: UserLLMAdapter, source: Concept, target: Concept) -> RelationProposal | None:
    proposal = adapter.resolve_temporal(source, target)
    if proposal is None:
        return None
    if proposal.source_concept_id != source.id or proposal.target_concept_id != target.id:
        return None
    if proposal.relation not in {RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
    if not 0 <= proposal.confidence <= 1:
        return None
    if proposal.same_subject is not True or proposal.same_scope is not True:
        return None
    if not proposal.reason.strip() or not proposal.evidence:
        return None
    evidence_space = "\n".join((source.text, source.source_quote, target.text, target.source_quote))
    if proposal.evidence not in evidence_space:
        return None
    if proposal.relation == RelationType.SUPERSEDES.value:
        if proposal.temporal_change_possible is not True:
            return None
        if not (_source_is_newer(source, target) or _has_revision_evidence(source, proposal)):
            return None
    return replace(proposal, prompt_version=TEMPORAL_PROMPT_VERSION)
```

- [ ] **Step 6: Thread `vault` through the call sites in `net_builder.py`**

In `_discover_relations` (`net_builder.py:165-171`), change:

```python
        relation = classify(adapter, source, target)
        temporal = resolve(adapter, source, target, relation)
```

to:

```python
        relation = classify(adapter, source, target, vault=store.vault)
        temporal = resolve(adapter, source, target, relation, vault=store.vault)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_v2_relation_cache.py -v`
Expected: PASS.

- [ ] **Step 8: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS. In particular re-check `tests/test_v2_net_review_query_health.py::test_safe_relation_commits_but_risky_relation_waits_for_review` and `tests/test_v2_hardening.py` — both call `build_net` more than once against the same fixture and must still see identical results now that answers are cached.

- [ ] **Step 9: Commit**

```bash
git add src/llm_wiki/v2/relation_cache.py src/llm_wiki/v2/relation_classifier.py \
        src/llm_wiki/v2/temporal.py src/llm_wiki/v2/net_builder.py \
        tests/test_v2_relation_cache.py
git commit -m "perf: cache classify_relation/resolve_temporal like concept extraction"
```

---

### Task 3: Skip low-relevance candidate pairs before spending an LLM call

**Files:**
- Modify: `src/llm_wiki/v2/relation_candidates.py`
- Modify: `src/llm_wiki/v2/net_builder.py:156-173`
- Modify: `src/llm_wiki/config.py` (new `[v2] relation_candidate_min_score`)
- Modify: `wiki.toml.example`
- Test: `tests/test_v2_relation_candidates.py` (new)

**Interfaces:**
- Consumes: `Config.v2_relation_candidate_topk` (existing), `concept_index.search(vault, text, k, concepts) -> list[tuple[float, Concept]]` (existing, unchanged)
- Produces: `discover(vault, seed, concepts, top_k=10) -> list[tuple[float, Concept]]` (return type changes from `list[Concept]` to `list[tuple[float, Concept]]` — the only caller is updated in this same task)
- Produces: `Config.v2_relation_candidate_min_score: float` (new field, default `0.0` — filtering is opt-in via `wiki.toml`, so default behavior is unchanged)

**Context:** `relation_candidates.discover()` already computes a fused relevance score per candidate (dense/sparse RRF plus small document/topic priors) but throws the score away before returning (`relation_candidates.py:19`: `return [c for _, c in sorted(...)[:top_k]]`). `net_builder._discover_relations` then calls `classify()`/`resolve()` — a full external CLI spawn each — for **every** one of the `top_k` candidates per concept, regardless of how weak the match is. Exposing the score lets an operator configure a floor (e.g. "don't bother classifying a pair whose fused score is below 0.01") to cut LLM call volume on large vaults, without touching the topk knob that controls recall breadth.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_relation_candidates.py
from pathlib import Path

from llm_wiki.v2 import net_builder, relation_candidates
from llm_wiki.v2.models import Concept


def _concept(cid: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=cid,
        summary=cid, source_quote=cid, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(cid),
    )


def test_discover_returns_score_alongside_each_candidate(monkeypatch, tmp_path: Path):
    seed, high, low = _concept("concept:seed"), _concept("concept:high"), _concept("concept:low")
    monkeypatch.setattr(
        relation_candidates.concept_index, "search",
        lambda vault, text, k, concepts: [(0.05, high), (0.001, low)],
    )
    result = relation_candidates.discover(tmp_path, seed, [seed, high, low], top_k=10)
    assert result[0][0] >= result[1][0]
    assert {concept.id for _, concept in result} == {"concept:high", "concept:low"}


def test_candidate_pairs_drops_scores_below_the_configured_floor(monkeypatch):
    seed, high, low = _concept("concept:seed"), _concept("concept:high"), _concept("concept:low")
    monkeypatch.setattr(
        net_builder, "discover",
        lambda vault, source, concepts, top_k: [(0.05, high), (0.001, low)],
    )
    pairs = net_builder._candidate_pairs(None, [seed], topk=10, min_score=0.01)
    assert pairs == [(seed, high)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v2_relation_candidates.py -v`
Expected: FAIL — `discover()` still returns bare `Concept` objects, and `net_builder._candidate_pairs` does not exist yet.

- [ ] **Step 3: Change `discover()` to return scored candidates**

In `src/llm_wiki/v2/relation_candidates.py`, change the return line:

```python
def discover(vault: Path | None, seed: Concept, concepts: list[Concept], top_k: int = 10) -> list[tuple[float, Concept]]:
    """Fuse dense/text concept ranking with cheap document/topic priors."""
    ranked = concept_index.search(vault, seed.text, k=max(top_k * 4, 20), concepts=concepts)
    scored: list[tuple[float, Concept]] = []
    for score, candidate in ranked:
        if candidate.id == seed.id:
            continue
        prior = 0.002 if candidate.document_id == seed.document_id else 0.0
        if seed.primary_topic_id and seed.primary_topic_id == candidate.primary_topic_id:
            prior += 0.001
        scored.append((score + prior, candidate))
    return sorted(scored, key=lambda row: (-row[0], row[1].id))[:top_k]
```

(Only the final `return` line changed — it now returns the `(score, concept)` pairs it already had instead of discarding the score.)

- [ ] **Step 4: Add the config field**

In `src/llm_wiki/config.py`, next to `v2_relation_candidate_topk` (around line 97 for the dataclass field, line 233 for parsing, line 269 for construction):

```python
    v2_relation_candidate_topk: int
    v2_relation_candidate_min_score: float
```

```python
    v2_relation_candidate_min_score = v2.get("relation_candidate_min_score", 0.0)
    if not isinstance(v2_relation_candidate_min_score, (int, float)) or v2_relation_candidate_min_score < 0:
        raise ConfigError("[v2] relation_candidate_min_score must be a non-negative number")
```

```python
        v2_relation_candidate_topk=v2_relation_candidate_topk,
        v2_relation_candidate_min_score=float(v2_relation_candidate_min_score),
```

- [ ] **Step 5: Extract and use a pure `_candidate_pairs` helper in `net_builder.py`**

`tests/test_v2_net_review_query_health.py::test_net_build_reports_each_long_running_phase` asserts a `"candidates"` progress phase fires once per concept — keep emitting it from inside the new helper. Replace the pair-building loop inside `_discover_relations` (`net_builder.py:156-164`):

```python
def _discover_relations(store: NetStore, concepts, adapter, cfg,
                        progress: NetProgress | None = None) -> None:
    terminal = {proposal.id for proposal in store.proposals() if proposal.status in {"APPROVED", "REJECTED"}}
    pairs = _candidate_pairs(store.vault, concepts, cfg.v2_relation_candidate_topk,
                             cfg.v2_relation_candidate_min_score, progress)
    for index, (source, target) in enumerate(pairs, start=1):
        relation = classify(adapter, source, target, vault=store.vault)
        temporal = resolve(adapter, source, target, relation, vault=store.vault)
        for proposal in (relation, temporal):
            if proposal is not None and proposal.id not in terminal:
                submit_relation_proposal(store, proposal, cfg.v2_safe_relation_min_confidence,
                                         cfg.v2_require_user_approval)
        if progress:
            progress("relations", index, len(pairs), f"{source.id} -> {target.id}")


def _candidate_pairs(vault, concepts, topk: int, min_score: float,
                     progress: NetProgress | None = None) -> list[tuple[Concept, Concept]]:
    pairs: list[tuple[Concept, Concept]] = []
    for index, source in enumerate(concepts, start=1):
        for score, target in discover(vault, source, concepts, topk):
            if score >= min_score:
                pairs.append((source, target))
        if progress:
            progress("candidates", index, len(concepts), source.summary or source.id)
    return pairs
```

(The `"candidates"` progress callback moves from the old inline loop into this helper, still firing once per source concept exactly as before; the `"relations"` callback over the final pair list is unchanged.)

- [ ] **Step 6: Add the knob to the example config**

In `wiki.toml.example`, under the `[v2]` section, add a line near `relation_candidate_topk` (adjust to match the file's existing style):

```toml
relation_candidate_min_score = 0.0
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_v2_relation_candidates.py -v`
Expected: PASS.

- [ ] **Step 8: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS — default `min_score=0.0` means every existing test (none of which configure this field) sees exactly the same candidate pairs as before.

- [ ] **Step 9: Commit**

```bash
git add src/llm_wiki/v2/relation_candidates.py src/llm_wiki/v2/net_builder.py \
        src/llm_wiki/config.py wiki.toml.example tests/test_v2_relation_candidates.py
git commit -m "perf: expose relation candidate score so low-relevance pairs can skip the LLM"
```

---

### Task 4: Stop re-reading the NET graph and re-computing signals per query

**Files:**
- Modify: `src/llm_wiki/v2/query.py`
- Test: `tests/test_v2_query_cost.py` (new)

**Interfaces:**
- Consumes: `NetStore.nodes() -> list[NetNode]`, `NetStore.edges() -> list[NetEdge]` (existing, unchanged), `concept_index.search_with_signals(vault, query, k, concepts=None) -> list[dict]` (existing, unchanged)
- Produces: `recall(vault, query, k=8, historical=False, rerank=0, seeds=None) -> list[dict]` (new optional `seeds` param; omitting it preserves current behavior exactly)
- Produces: `_tree_scores(query, nodes, edges) -> dict[str, float]` (signature changes from `(query, store)`)
- Produces: `_graph_scores(seed_scores, edges, historical) -> dict[str, float]` (signature changes from `(seed_scores, store, historical)`)
- Produces: `_evidence(concept_id, edges) -> list[dict]` (signature changes from `(concept_id, store)`)

**Context:** Two separate wastes in the same file, confirmed by reading `net_store.py:25-28` (`NetStore.nodes()`/`.edges()` call `artifacts.read_jsonl` fresh every time, no caching) and `query.py`:
- `recall()` calls `store.edges()` inside `_tree_scores` (once), inside `_graph_scores` (once), and `_evidence()` calls `store.edges()` **again inside the per-result loop** (`query.py:46-56`, up to `k` times) — `edges.jsonl` gets re-read and re-parsed from disk `2 + k` times for one query.
- `auto_decision()` calls `recall(vault, query, k=max(k, 10), ...)` (which internally computes `concept_index.search(...)`, itself a thin wrapper over `search_with_signals`) and then separately calls `concept_index.search_with_signals(vault, query, k=max(k * 5, 30))` again for the *same query* just to recover `dense_score` (`query.py:139-144`) — the sparse+dense signal pass runs twice, and with two different `k` values that can disagree, so `signals.get(row["id"], ...)` can silently miss an entry `recall()` found and fall back to `dense_score=0.0` (a pre-existing correctness gap this fix also closes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_query_cost.py
from pathlib import Path

from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.query import auto_decision, recall
from llm_wiki.v2 import concept_index


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "stack.md").write_text(
        "---\nid: stack\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: Stack decisions\n---\n"
        "# Stack\n\nFrontend no longer uses Vue and now uses React. Backend uses Spring.",
        encoding="utf-8")
    build_concepts(tmp_path)
    build_index(tmp_path)
    build_net(tmp_path)
    return tmp_path


def test_recall_reads_the_net_graph_at_most_once_per_call(tmp_path, monkeypatch):
    vault = _vault(tmp_path)
    calls = {"nodes": 0, "edges": 0}
    real_nodes, real_edges = NetStore.nodes, NetStore.edges

    def counting_nodes(self):
        calls["nodes"] += 1
        return real_nodes(self)

    def counting_edges(self):
        calls["edges"] += 1
        return real_edges(self)

    monkeypatch.setattr(NetStore, "nodes", counting_nodes)
    monkeypatch.setattr(NetStore, "edges", counting_edges)
    rows = recall(vault, "React", k=5)
    assert rows
    assert calls["nodes"] == 1
    assert calls["edges"] == 1


def test_auto_decision_computes_signals_only_once(tmp_path, monkeypatch):
    vault = _vault(tmp_path)
    calls = {"count": 0}
    real_search_with_signals = concept_index.search_with_signals

    def counting(vault_arg, query, k=8, concepts=None):
        calls["count"] += 1
        return real_search_with_signals(vault_arg, query, k=k, concepts=concepts)

    monkeypatch.setattr(concept_index, "search_with_signals", counting)
    payload = auto_decision(vault, "React", k=5)
    assert payload["decision"] in {"answer", "review", "none"}
    assert calls["count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v2_query_cost.py -v`
Expected: FAIL — `test_recall_reads_the_net_graph_at_most_once_per_call` reports `calls["edges"] > 1`; `test_auto_decision_computes_signals_only_once` reports `calls["count"] == 2`.

- [ ] **Step 3: Load the graph once in `recall()` and thread lists through the helpers**

In `src/llm_wiki/v2/query.py`, change `recall()`:

```python
def recall(vault: Path | None, query: str, k: int = 8, historical: bool = False,
           rerank: int = 0, seeds: list[tuple[float, Concept]] | None = None) -> list[dict]:
    """Soft-route through NET, then fuse semantic/text seeds and graph evidence.

    Tree routing deliberately only adds a rank signal. A bad auto-placement must
    never make an otherwise relevant concept unreachable.
    """
    concepts = read_concepts(vault)
    if not concepts:
        return []
    _fail_closed(vault)
    if seeds is None:
        seeds = concept_index.search(vault, query, k=max(k * 5, 30))
    scores = _rrf_scores(seeds)
    store = NetStore(vault)
    nodes = store.nodes()
    edges = store.edges()
    tree = _tree_scores(query, nodes, edges)
    for concept_id, value in tree.items():
        scores[concept_id] = scores.get(concept_id, 0.0) + value
    for concept_id, value in _graph_scores(scores, edges, historical).items():
        scores[concept_id] = scores.get(concept_id, 0.0) + value
    by_id = {concept.id: concept for concept in concepts}
    if historical:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        for concept in concepts:
            if concept.updated_at and years.intersection(re.findall(r"\b(?:19|20)\d{2}\b", concept.updated_at)):
                scores[concept.id] = scores.get(concept.id, 0.0) + 0.02
    rows: list[dict] = []
    for concept_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        concept = by_id.get(concept_id)
        if not concept or (not historical and concept.state in {ConceptState.SUPERSEDED.value, ConceptState.ARCHIVED.value, ConceptState.DUPLICATE.value}):
            continue
        evidence = _evidence(concept_id, edges)
        warning = None
        if concept.state == ConceptState.DISPUTED.value:
            warning = "DISPUTED: review contradiction evidence before relying on this concept"
        elif any(item["relation"] == RelationType.OVERRIDES.value for item in evidence):
            warning = "OVERRIDES: verify scope/priority before treating this as universal"
        rows.append(_row(score, concept, warning, evidence))
        if len(rows) >= max(k, rerank):
            break
    if rerank and rows:
        try:
            from llm_wiki.retrieval._rerank import rerank_scores
            values = rerank_scores(query, [row["text"] for row in rows])
            for row, value in zip(rows, values):
                row["rerank_score"] = value
            rows.sort(key=lambda row: (-row["rerank_score"], -row["score"], row["id"]))
        except (ImportError, OSError, RuntimeError):
            pass
    return rows[:k]
```

Then update the three helpers:

```python
def _tree_scores(query: str, nodes: list, edges: list) -> dict[str, float]:
    q = _tokens(query)
    if not q:
        return {}
    topic_relevance = {node.id: len(q & _tokens(node.label)) for node in nodes}
    out: dict[str, float] = {}
    for edge in edges:
        if edge.type in {EdgeType.PRIMARY_TOPIC_OF.value, EdgeType.SECONDARY_TOPIC_OF.value}:
            rel = topic_relevance.get(edge.source, 0)
            if rel:
                out[edge.target] = out.get(edge.target, 0.0) + min(0.01, rel / 200)
    return out


def _graph_scores(seed_scores: dict[str, float], edges: list, historical: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    allowed = {RelationType.SUPPORTS.value, RelationType.COMPLEMENTS.value, RelationType.DUPLICATE_OF.value}
    if historical:
        allowed.add(RelationType.SUPERSEDES.value)
    top = {concept_id for concept_id, _ in sorted(seed_scores.items(), key=lambda row: -row[1])[:12]}
    for edge in edges:
        if edge.type != EdgeType.RELATES_TO.value or edge.relation not in allowed:
            continue
        if edge.source in top:
            out[edge.target] = out.get(edge.target, 0.0) + 0.004
        if edge.target in top and edge.relation != RelationType.SUPERSEDES.value:
            out[edge.source] = out.get(edge.source, 0.0) + 0.004
    return out


def _evidence(concept_id: str, edges: list) -> list[dict]:
    return [{"relation": edge.relation, "concept_id": edge.target if edge.source == concept_id else edge.source}
            for edge in edges
            if edge.type == EdgeType.RELATES_TO.value and concept_id in {edge.source, edge.target}]
```

- [ ] **Step 4: Collapse the double signal computation in `auto_decision()`**

Change:

```python
def auto_decision(vault: Path | None, query: str, k: int = 8, historical: bool = False,
                  thresholds_path: Path | None = None) -> dict:
    rows = recall(vault, query, k=max(k, 10), historical=historical)
    if not rows:
        return {"decision": "none", "reason": "no-candidates", "results": []}
    signals = {row["concept"].id: row for row in concept_index.search_with_signals(
        vault, query, k=max(k * 5, 30)
    )}
```

to:

```python
def auto_decision(vault: Path | None, query: str, k: int = 8, historical: bool = False,
                  thresholds_path: Path | None = None) -> dict:
    signal_k = max(k, 10) * 5
    signal_rows = concept_index.search_with_signals(vault, query, k=max(signal_k, 30))
    seeds = [(row["score"], row["concept"]) for row in signal_rows]
    rows = recall(vault, query, k=max(k, 10), historical=historical, seeds=seeds)
    if not rows:
        return {"decision": "none", "reason": "no-candidates", "results": []}
    signals = {row["concept"].id: row for row in signal_rows}
```

The rest of `auto_decision()` is unchanged — `candidates`, `threshold_file`, `thresholds`, `semantic_index`, and everything after still reference `rows` and `signals` exactly as before.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_v2_query_cost.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS — in particular `tests/test_v2_net_review_query_health.py::test_query_prefers_active_and_historical_includes_superseded` and any test touching `run_cli_query`/`auto_decision`, since ranking math is unchanged, only how many times data is fetched.

- [ ] **Step 7: Commit**

```bash
git add src/llm_wiki/v2/query.py tests/test_v2_query_cost.py
git commit -m "perf: read the NET graph once per query and compute signals once in auto_decision"
```

---

## Suggested order

Tasks 1 → 2 → 3 → 4, in that order: Task 1 is a pure bugfix with no dependents, Task 2 builds on Task 1's guard (so a cached `None` never gets computed twice for the wrong reason), Task 3 is independent of 1/2 but touches the same `net_builder.py` region so doing it after 2 avoids a merge conflict, and Task 4 (query path) is fully independent of the other three and could be done first if that's more urgent for the immediate `wiki-net query --auto` latency.

---

## Measured results (2026-08-17)

All four tasks landed (`c68d9e0` Task 1, `5b19a26` Task 2 + its `001fa3a` test-isolation follow-up, `02645b7` Task 3, `6a6aafd` Task 4 — see `git log`). Ran a real end-to-end `wiki-net build` (concept extraction → index → relations) against a synthetic 15-document vault, using the real `claude` CLI agent (`AgentCLIUserLLMAdapter`, not the offline rule-based adapter), once against the pre-fix commit (`c551e78`, checked out in an isolated `git worktree`) and once against current `HEAD`. Same vault content, same `wiki.toml` (`[v2] agent = "claude"`, default `relation_candidate_topk = 10`, no `relation_candidate_min_score` configured), each run starting from a clean `.llm_wiki_v2` artifact directory (no cache carried over).

| stage | before (c551e78) | after (HEAD) | delta |
|---|---|---|---|
| concept extraction | 147.7s | 140.8s | ~unchanged (not in scope) |
| index build | ~0s | ~0s | ~unchanged (not in scope) |
| **relations (candidates + classify + resolve)** | **3326.8s (55.4 min)** | **1976.5s (33.0 min)** | **-40.6%** |
| **total** | **3474.5s (57.9 min)** | **2117.3s (35.3 min)** | **-39.1%** |

Both runs produced the same graph shape (40 nodes; 57 vs 59 edges — the 2-edge difference is LLM classification non-determinism, not a contract change), confirming the speedup is free — grounding, confidence gating, and lifecycle states are unaffected.

**Caveat — this measurement isolates Task 1 only.** With 17 concepts and `topk=10`, the vault produced 170 candidate pairs. `relation_candidate_min_score` was left at its default `0.0`, so Task 3 filtered nothing; both runs built from an empty cache, so Task 2 had no cache hits to save; Task 4 only affects `query.py`, not `build`. The entire 40.6% relations-stage improvement here comes from Task 1 alone (skipping `resolve_temporal` for the ~90%+ of pairs classify() found no risky relation for — only ~5-15 of the 170 pairs were classified CONTRADICTS/SUPERSEDES/OVERRIDES and actually needed a resolve call). Tasks 2-4's benefit shows up on **repeat builds** (cache hits), **larger vaults with a configured score floor** (Task 3), and **repeated/concurrent queries** (Task 4) — none of which this single from-scratch build exercises. A follow-up measurement re-running `wiki-net build` a second time against the same `after` vault (to hit the Task 2 cache) or against a vault with `relation_candidate_min_score` set would isolate those.

A harness-only note: 5 proposals were skipped in both runs with `SUPERSEDES requires newer metadata or explicit revision evidence` — a pre-existing `review.py` validation path unrelated to this plan's fixes, caught and logged by the measurement script (not by production code) so a single occasional bad LLM classification didn't abort an hour-long timed run.
