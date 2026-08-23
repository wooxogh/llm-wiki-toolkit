# v2 NET Build Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the wall-clock cost of a *first-time* `wiki-net build` (no cache to hit yet — the scenario `perf(v2)/relation-query-latency` does not help) by running independent relation-classification LLM calls concurrently, without introducing data loss or correctness regressions.

**Architecture:** Three tasks, done in this order because each is a prerequisite for the next being *safe*:
1. Add retry/backoff to the LLM adapter's external CLI calls. Concurrency multiplies how many in-flight calls can hit a transient provider error at once; without retry, going concurrent would raise the failure rate, not just the throughput.
2. Separate "compute a relation proposal" (pure, no side effects) from "commit a relation proposal to `NetStore`" (mutates JSONL files via a non-atomic read-modify-write with no locking) so that, however proposals get computed, they are always committed from a single thread. This is a prerequisite, not an optimization — `NetStore.append_proposal`/`upsert_edge` read the whole file, mutate in memory, and overwrite the whole file; calling them concurrently from multiple threads today would silently lose proposals/edges to last-writer-wins.
3. Introduce a bounded thread pool for the now-safe "compute" step, with a `[v2] relation_concurrency` config knob defaulting to `1` (serial — identical to current behavior unless an operator opts in).

**Tech Stack:** Python 3 standard library `concurrent.futures.ThreadPoolExecutor`, pytest, existing `llm_wiki.v2` conventions.

**Spec:** No separate spec doc — this plan is the spec, following directly from the investigation in this conversation (see file/line citations per task). Builds on top of `perf(v2)/relation-query-latency` (relation caching, candidate score filter, query-path fixes) — that branch is a prerequisite, not duplicated here.

## Global Constraints

- Default behavior (no new config set) must be byte-for-byte identical to today: same call order, same progress events, same committed proposals.
- No new third-party dependencies — `concurrent.futures` is stdlib.
- All `NetStore` mutation (`submit_relation_proposal` and anything it calls) must run on the thread that is iterating `_discover_relations`'s main loop — never inside a worker thread.
- Run `pytest tests/test_v2_*.py -q` at the end of every task — all must stay green.

---

### Task 1: Retry/backoff for the LLM adapter's external CLI calls

**Files:**
- Modify: `src/llm_wiki/v2/llm_adapter.py`
- Test: `tests/test_v2_llm_adapter_retry.py` (new)

**Interfaces:**
- Consumes: nothing new
- Produces: `_retry(call: Callable[[], dict], sleep=time.sleep) -> dict` (module-private helper); `CommandUserLLMAdapter._call` and `AgentCLIUserLLMAdapter._call` behavior unchanged on success, now retry up to `MAX_RETRIES` (module constant, default `3`, env override `WIKI_V2_LLM_MAX_RETRIES`) with exponential backoff (`RETRY_BASE_DELAY` seconds, default `2.0`, env override `WIKI_V2_LLM_RETRY_BASE_DELAY`) on `RuntimeError` before giving up and re-raising.

**Context:** `CommandUserLLMAdapter._call` (`llm_adapter.py:57-69`) and `AgentCLIUserLLMAdapter._call_codex`/`_call_claude` (`llm_adapter.py:141-165`) raise `RuntimeError` immediately on a non-zero exit code or unparseable JSON, with no retry. At the call volume `net_builder._discover_relations` generates (every candidate pair), and once Task 3 makes several of those calls land on the provider at the same moment, a transient hiccup (auth refresh, rate limit, brief network blip) becomes far more likely to hit *some* in-flight call. Today that aborts the whole `wiki-net build`. This task adds a bounded retry with exponential backoff around the external-process call in both adapters, so a transient failure costs a few extra seconds instead of the whole run.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_llm_adapter_retry.py
from types import SimpleNamespace

from llm_wiki.v2 import llm_adapter


def test_command_adapter_retries_a_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_adapter, "MAX_RETRIES", 3)
    monkeypatch.setattr(llm_adapter.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def flaky_run(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return SimpleNamespace(returncode=1, stdout="", stderr="rate limited")
        return SimpleNamespace(returncode=0, stdout='{"concepts": []}', stderr="")

    monkeypatch.setattr(llm_adapter.subprocess, "run", flaky_run)
    adapter = llm_adapter.CommandUserLLMAdapter("bridge")
    result = adapter._call("extract_concepts", {"chunk": "x"})
    assert result == {"concepts": []}
    assert attempts["count"] == 3


def test_command_adapter_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_adapter, "MAX_RETRIES", 2)
    monkeypatch.setattr(llm_adapter.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def always_fails(*args, **kwargs):
        attempts["count"] += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="down")

    monkeypatch.setattr(llm_adapter.subprocess, "run", always_fails)
    adapter = llm_adapter.CommandUserLLMAdapter("bridge")
    try:
        adapter._call("extract_concepts", {"chunk": "x"})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert attempts["count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v2_llm_adapter_retry.py -v`
Expected: FAIL — `llm_adapter` has no `time` attribute yet (not imported) and `_call` does not retry (`attempts["count"] == 1`).

- [ ] **Step 3: Add the retry helper and constants**

In `src/llm_wiki/v2/llm_adapter.py`, add near the top (after the existing imports):

```python
import time
from typing import Callable
```

(`time` is new; `Callable` may already be imported via `Protocol`'s neighbors — check the existing `from typing import Protocol` line and extend it to `from typing import Callable, Protocol` instead of adding a second `typing` import line.)

Add module-level constants next to the class definitions:

```python
MAX_RETRIES = int(os.environ.get("WIKI_V2_LLM_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("WIKI_V2_LLM_RETRY_BASE_DELAY", "2.0"))


def _retry(call: Callable[[], dict], sleep=time.sleep) -> dict:
    last_exc: RuntimeError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except RuntimeError as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise last_exc
```

- [ ] **Step 4: Wrap `CommandUserLLMAdapter._call`**

Rename the existing method body to `_call_once` and add a thin retrying wrapper:

```python
    def _call(self, task: str, payload: dict) -> dict:
        return _retry(lambda: self._call_once(task, payload))

    def _call_once(self, task: str, payload: dict) -> dict:
        request = json.dumps({"task": task, "payload": payload}, ensure_ascii=False)
        result = subprocess.run(shlex.split(self.command), input=request, text=True,
                                encoding="utf-8", capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"User LLM command failed ({result.returncode}): {result.stderr.strip()}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("User LLM command did not return JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("User LLM command must return a JSON object")
        return value
```

- [ ] **Step 5: Wrap `AgentCLIUserLLMAdapter._call`**

Rename its existing body to `_call_once` and add the same wrapper:

```python
    def _call(self, task: str, payload: dict) -> dict:
        return _retry(lambda: self._call_once(task, payload))

    def _call_once(self, task: str, payload: dict) -> dict:
        executable = shutil.which(self.agent)
        if not executable:
            raise RuntimeError(
                f"v2 agent '{self.agent}' is configured but its CLI is not installed or not on PATH"
            )
        request = {"task": task, "payload": payload}
        prompt = (
            "You are the semantic reasoning adapter for llm-wiki v2. "
            "Do not inspect files, run tools, or mutate any state. Evaluate only the JSON request below. "
            "Return only one JSON object matching payload.json_schema exactly.\n\n"
            + json.dumps(request, ensure_ascii=False)
        )
        schema = payload.get("json_schema", {"type": "object"})
        if self.agent == "codex":
            return self._call_codex(executable, prompt, schema)
        return self._call_claude(executable, prompt, schema)
```

(Only the method name changes from `_call` to `_call_once` — the body is identical to today's `_call`. The new `_call` above it is the only addition.)

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_v2_llm_adapter_retry.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS — `tests/test_v2_hardening.py::test_command_adapter_sends_versioned_atomic_contract` in particular, since it monkeypatches `subprocess.run` for a single successful call and must still see exactly one call (no accidental retry loop on success).

- [ ] **Step 8: Commit**

```bash
git add src/llm_wiki/v2/llm_adapter.py tests/test_v2_llm_adapter_retry.py
git commit -m "feat: retry transient LLM CLI failures with exponential backoff"
```

---

### Task 2: Separate relation-proposal computation from `NetStore` commits

**Files:**
- Modify: `src/llm_wiki/v2/net_builder.py`
- Test: `tests/test_v2_relation_commit_seam.py` (new)

**Interfaces:**
- Consumes: `classify(adapter, source, target, vault=None)`, `resolve(adapter, source, target, relation, vault=None)` (existing, from `perf(v2)/relation-query-latency`)
- Produces: `_classify_pair(adapter, vault, source, target) -> tuple[Concept, Concept, list[RelationProposal]]` (pure — no `NetStore` access)
- Produces: `_compute_relation_proposals(pairs, adapter, vault) -> Iterator[tuple[Concept, Concept, list[RelationProposal]]]` (sequential in this task; Task 3 adds a concurrent branch behind a parameter)
- `_discover_relations`'s public behavior (signature, progress events, committed proposals) is unchanged — this task only changes its internal shape.

**Context:** `NetStore.append_proposal` (`net_store.py:100-103`) and `upsert_edge` read the whole JSONL file, mutate the list in Python, and overwrite the whole file — no lock, no compare-and-swap. `submit_relation_proposal` (`review.py:13-38`, called from `_discover_relations`) calls these. Today `_discover_relations` computes and commits in the same loop iteration, so there is exactly one thread ever touching the store — safe by accident, not by design. Before any concurrency can be introduced safely, "compute a proposal" must be extracted as a pure function with no store access, and the loop that calls `submit_relation_proposal` must be the only place any thread ever calls it.

This task also fixes a latent quirk while it's in the neighborhood: today, when `dirty_ids` skips a pair (`net_builder.py:168-169`), the `continue` skips the `progress("relations", ...)` call too, so if the *last* pair in the list happens to be skipped, the progress bar never reports `done == total`. Pre-filtering `pairs` before the loop (instead of `continue`-ing inside it) means every remaining pair gets exactly one progress tick, so the last one always closes out the bar.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_relation_commit_seam.py
from pathlib import Path

from llm_wiki.v2 import net_builder
from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType


def _concept(cid: str, text: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=text,
        summary=text, source_quote=text, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(text),
    )


class _StubAdapter:
    model_identity = "stub"

    def __init__(self, relation_by_pair):
        self.relation_by_pair = relation_by_pair

    def classify_relation(self, source, target):
        return self.relation_by_pair.get((source.id, target.id))

    def resolve_temporal(self, source, target):
        return None


def test_classify_pair_never_touches_a_store(tmp_path: Path):
    source, target = _concept("concept:a", "Frontend uses React"), _concept("concept:b", "Frontend used Vue")
    proposal = RelationProposal(
        id="proposal:a:SUPPORTS:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPPORTS.value, confidence=0.9, evidence=source.source_quote,
    )
    adapter = _StubAdapter({(source.id, target.id): proposal})
    result_source, result_target, proposals = net_builder._classify_pair(adapter, tmp_path, source, target)
    assert result_source is source and result_target is target
    assert [p.relation for p in proposals] == [RelationType.SUPPORTS.value]


def test_compute_relation_proposals_covers_every_pair(tmp_path: Path):
    a, b, c = _concept("concept:a", "A"), _concept("concept:b", "B"), _concept("concept:c", "C")
    adapter = _StubAdapter({})
    results = list(net_builder._compute_relation_proposals([(a, b), (b, c)], adapter, tmp_path))
    seen_pairs = {(source.id, target.id) for source, target, _ in results}
    assert seen_pairs == {("concept:a", "concept:b"), ("concept:b", "concept:c")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v2_relation_commit_seam.py -v`
Expected: FAIL — `net_builder` has no `_classify_pair`/`_compute_relation_proposals` attributes yet.

- [ ] **Step 3: Add the pure compute functions and rewire `_discover_relations`**

In `src/llm_wiki/v2/net_builder.py`, replace:

```python
def _discover_relations(store: NetStore, concepts, adapter, cfg,
                        progress: NetProgress | None = None,
                        dirty_ids: set[str] | None = None) -> None:
    terminal = {proposal.id for proposal in store.proposals() if proposal.status in {"APPROVED", "REJECTED"}}
    pairs = _candidate_pairs(store.vault, concepts, cfg.v2_relation_candidate_topk,
                             cfg.v2_relation_candidate_min_score, progress)
    for index, (source, target) in enumerate(pairs, start=1):
        if dirty_ids is not None and source.id not in dirty_ids and target.id not in dirty_ids:
            continue
        relation = classify(adapter, source, target, vault=store.vault)
        temporal = resolve(adapter, source, target, relation, vault=store.vault)
        for proposal in (relation, temporal):
            if proposal is not None and proposal.id not in terminal:
                submit_relation_proposal(store, proposal, cfg.v2_safe_relation_min_confidence,
                                         cfg.v2_require_user_approval)
        if progress:
            progress("relations", index, len(pairs), f"{source.id} -> {target.id}")
```

with:

```python
def _discover_relations(store: NetStore, concepts, adapter, cfg,
                        progress: NetProgress | None = None,
                        dirty_ids: set[str] | None = None) -> None:
    terminal = {proposal.id for proposal in store.proposals() if proposal.status in {"APPROVED", "REJECTED"}}
    pairs = _candidate_pairs(store.vault, concepts, cfg.v2_relation_candidate_topk,
                             cfg.v2_relation_candidate_min_score, progress)
    if dirty_ids is not None:
        pairs = [(source, target) for source, target in pairs
                if source.id in dirty_ids or target.id in dirty_ids]
    total = len(pairs)
    # Every submit_relation_proposal() call below happens on THIS thread, no
    # matter how _compute_relation_proposals computes its results (serially
    # today, optionally concurrently once a worker count is threaded in) —
    # NetStore's append/upsert methods do an unlocked read-modify-write and
    # are not safe to call from more than one thread.
    for index, (source, target, proposals) in enumerate(
        _compute_relation_proposals(pairs, adapter, store.vault), start=1
    ):
        for proposal in proposals:
            if proposal.id not in terminal:
                submit_relation_proposal(store, proposal, cfg.v2_safe_relation_min_confidence,
                                         cfg.v2_require_user_approval)
        if progress:
            progress("relations", index, total, f"{source.id} -> {target.id}")


def _classify_pair(adapter, vault, source, target):
    relation = classify(adapter, source, target, vault=vault)
    temporal = resolve(adapter, source, target, relation, vault=vault)
    return source, target, [p for p in (relation, temporal) if p is not None]


def _compute_relation_proposals(pairs, adapter, vault):
    for source, target in pairs:
        yield _classify_pair(adapter, vault, source, target)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v2_relation_commit_seam.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS — in particular `tests/test_v2_net_review_query_health.py::test_net_build_reports_each_long_running_phase` (progress phases still fire) and `test_safe_relation_commits_but_risky_relation_waits_for_review` (commit behavior unchanged) and the incremental-build tests in `tests/test_v2_completion.py` that exercise `changed_only`/`dirty_ids`.

- [ ] **Step 6: Commit**

```bash
git add src/llm_wiki/v2/net_builder.py tests/test_v2_relation_commit_seam.py
git commit -m "refactor: separate relation-proposal computation from NetStore commits"
```

---

### Task 3: Bounded thread pool for relation-proposal computation

**Files:**
- Modify: `src/llm_wiki/v2/net_builder.py`
- Modify: `src/llm_wiki/config.py`
- Modify: `wiki.toml.example`
- Test: `tests/test_v2_relation_concurrency.py` (new)

**Interfaces:**
- Consumes: `_classify_pair` (Task 2, unchanged)
- Produces: `_compute_relation_proposals(pairs, adapter, vault, workers: int = 1) -> Iterator[...]` (new `workers` parameter, default `1` preserves Task 2's sequential behavior exactly)
- Produces: `Config.v2_relation_concurrency: int` (new field, default `1`)

**Context:** With Task 2's seam in place, computing each pair's proposals concurrently is now safe — every `submit_relation_proposal` call still happens on the single thread that is iterating `_compute_relation_proposals`'s results; only the (side-effect-free) `_classify_pair` calls run on worker threads. `workers=1` takes the exact same code path as before (no `ThreadPoolExecutor` involved), so the default behavior for every existing test and every vault that hasn't opted in is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_relation_concurrency.py
import threading
from pathlib import Path

from llm_wiki.v2 import net_builder
from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType


def _concept(cid: str, text: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=text,
        summary=text, source_quote=text, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(text),
    )


class _ThreadRecordingAdapter:
    model_identity = "stub"

    def __init__(self):
        self.threads: set[int] = set()
        self.lock = threading.Lock()

    def classify_relation(self, source, target):
        with self.lock:
            self.threads.add(threading.get_ident())
        return RelationProposal(
            id=f"proposal:{source.id}:SUPPORTS:{target.id}", source_concept_id=source.id,
            target_concept_id=target.id, relation=RelationType.SUPPORTS.value,
            confidence=0.9, evidence=source.source_quote,
        )

    def resolve_temporal(self, source, target):
        return None


def test_concurrent_computation_uses_more_than_one_worker_thread(tmp_path: Path):
    concepts = [_concept(f"concept:{i}", f"text {i}") for i in range(8)]
    pairs = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
    adapter = _ThreadRecordingAdapter()
    results = list(net_builder._compute_relation_proposals(pairs, adapter, tmp_path, workers=4))
    assert len(results) == len(pairs)
    assert len(adapter.threads) > 1


def test_default_workers_stays_single_threaded(tmp_path: Path):
    concepts = [_concept(f"concept:{i}", f"text {i}") for i in range(4)]
    pairs = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
    adapter = _ThreadRecordingAdapter()
    results = list(net_builder._compute_relation_proposals(pairs, adapter, tmp_path))
    assert len(results) == len(pairs)
    assert adapter.threads == {threading.get_ident()}


def test_concurrent_build_commits_every_expected_proposal(tmp_path: Path):
    from llm_wiki.v2.concept_store import build_concepts
    from llm_wiki.v2.net_builder import build_net

    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\nrelation_concurrency = 4\n'
        'safe_relation_min_confidence = 0.0\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    for i in range(6):
        (tmp_path / "domain" / f"doc{i}.md").write_text(
            f"---\nid: doc{i}\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: doc {i}\n---\n"
            f"# Doc {i}\n\nFrontend uses React in service {i}.",
            encoding="utf-8")
    build_concepts(tmp_path)
    store = build_net(tmp_path, adapter=_ThreadRecordingAdapter())
    proposals = store.proposals()
    # A lost update from the read-modify-write race this plan guards against
    # would show up here as duplicate/overwritten ids or a short list.
    assert len(proposals) > 0
    assert len(proposals) == len({proposal.id for proposal in proposals})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v2_relation_concurrency.py -v`
Expected: FAIL — `_compute_relation_proposals` does not accept a `workers` keyword yet (`TypeError: _compute_relation_proposals() got an unexpected keyword argument 'workers'`), and `build_net` does not accept `relation_concurrency` from `wiki.toml` yet (`ConfigError` or the config field simply doesn't exist).

- [ ] **Step 3: Add the config field**

In `src/llm_wiki/config.py`, next to `v2_relation_candidate_min_score`:

```python
    v2_relation_candidate_min_score: float
    v2_relation_concurrency: int
```

```python
    v2_relation_concurrency = v2.get("relation_concurrency", 1)
    if not isinstance(v2_relation_concurrency, int) or v2_relation_concurrency <= 0:
        raise ConfigError("[v2] relation_concurrency must be a positive integer")
```

```python
        v2_relation_candidate_min_score=float(v2_relation_candidate_min_score),
        v2_relation_concurrency=v2_relation_concurrency,
```

- [ ] **Step 4: Add the thread pool branch**

In `src/llm_wiki/v2/net_builder.py`, add the import:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Change `_compute_relation_proposals` and its call site:

```python
def _compute_relation_proposals(pairs, adapter, vault, workers: int = 1):
    if workers <= 1:
        for source, target in pairs:
            yield _classify_pair(adapter, vault, source, target)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_classify_pair, adapter, vault, source, target) for source, target in pairs]
        for future in as_completed(futures):
            yield future.result()
```

And in `_discover_relations`, pass the config value through:

```python
    for index, (source, target, proposals) in enumerate(
        _compute_relation_proposals(pairs, adapter, store.vault, cfg.v2_relation_concurrency), start=1
    ):
```

- [ ] **Step 5: Add the knob to the example config**

In `wiki.toml.example`, next to `relation_candidate_min_score`:

```toml
relation_concurrency = 1   # raise to classify multiple candidate pairs concurrently; commits stay single-threaded
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_v2_relation_concurrency.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full v2 suite for regressions**

Run: `pytest tests/test_v2_*.py -q`
Expected: all PASS — `workers` defaulting to `1` in every existing call site (`_discover_relations` always passes `cfg.v2_relation_concurrency`, which defaults to `1` unless a vault's `wiki.toml` sets it) means every prior test's call order and commit set is unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/llm_wiki/v2/net_builder.py src/llm_wiki/config.py wiki.toml.example \
        tests/test_v2_relation_concurrency.py
git commit -m "feat: bounded thread pool for relation-proposal computation, default 1 (serial)"
```

---

## What this plan deliberately does NOT do

- **Pick a "correct" concurrency number.** `relation_concurrency` ships defaulting to `1` (no change) precisely because the right value depends on the provider's own rate limits and the local machine's capacity to spawn `codex exec`/`claude -p` processes — both of which must be measured against the target vault and provider account, not guessed. Once this lands, the next step is an operator running a real build at `relation_concurrency = 2, 4, 8` and comparing wall-clock and error/retry counts to find the knee of the curve.
- **Extend concurrency to concept extraction** (`concepts_from_chunks`/`extract()`). That loop has the same "safe to parallelize the compute, must serialize the write" shape (the cache write in `concept_extraction.py:28-29` is a plain file write, not a read-modify-write over a shared list, so its race is narrower — a lost cache write costs a re-computation, not lost data — but it deserves its own look rather than folding it in here).
- **Touch the query path's `_fail_closed()` full-vault re-verification cost.** That's a query-latency issue, unrelated to build-time concurrency.
