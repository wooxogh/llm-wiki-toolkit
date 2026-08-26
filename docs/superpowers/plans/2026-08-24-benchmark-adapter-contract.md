# Benchmark Adapter Contract Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the benchmark adapter contract so it normalizes the released LongMemEval, HoH, VitaminC, RGB, and FactLens formats, and so a run cannot report a metric it did not measure.

**Architecture:** Provenance (`split`, version) moves from the record contract to configuration and to a computed content digest. Container reading separates from field mapping, so each adapter declares a reader (`read_json_array`, `read_jsonl`, `read_csv`, `read_parquet`) and implements only `normalize`. Each case declares a task profile, and profiles declare capabilities; the runner scores per declared capability instead of inferring applicability from whether a field happens to be populated.

**Tech Stack:** Python 3.11+, stdlib only (`json`, `csv`, `hashlib`, `ast`), PyYAML for configuration, pytest for tests. `pyarrow` is an optional extra used solely by the HoH Parquet reader.

**Spec:** `docs/superpowers/specs/2026-08-24-benchmark-adapter-contract-design.md`

## Global Constraints

- Python floor: `requires-python = ">=3.11"`. Target both 3.11 and 3.12 (CI matrix).
- No network access in any test. No ML stack import (`torch`, `sentence_transformers`) anywhere in the benchmark package.
- No dataset data committed to the repository. Fixtures carry released *shapes* with synthetic *content*.
- `pyarrow` is an optional extra `benchmarks[hoh]`. Importing it at module scope is forbidden; a missing `pyarrow` must produce an actionable message, not an `ImportError` traceback.
- Seven registered suites: `longmemeval`, `hoh`, `vitaminc`, `factlens`, `rgb_base`, `rgb_integration`, `rgb_counterfactual`.
- All JSON artifacts stay deterministic: `sort_keys=True`, `ensure_ascii=False`, `allow_nan=False`, `newline="\n"` (existing `reports._json` contract).
- `wiki-eval`, `eval_gold.json`, and `eval_baseline.json` are not touched.
- Every error message names the source file and the record number.

---

## File Structure

**Create:**
- `benchmarks/src/llm_wiki_bench/profiles.py` — capability vocabulary, profile table, per-capability required-field rules.
- `benchmarks/src/llm_wiki_bench/readers.py` — the four container readers plus `file_digest`.
- `benchmarks/src/llm_wiki_bench/adapters/rgb_base.py`, `rgb_integration.py`, `rgb_counterfactual.py` — replace `rgb.py`.
- `benchmarks/tests/test_profiles.py`, `test_readers.py`, `test_conformance.py`.

**Modify:**
- `schema.py` — `BenchmarkCase` gains `profile`, `fine_evidence_ids`, `expects_abstention`; drops `task`. `Prediction` gains `sub_claim_labels`.
- `adapters/base.py` — reader declaration, `LoadResult`, no per-record provenance.
- `adapters/{longmemeval,hoh,vitaminc,factlens}.py` — rewritten to released formats.
- `metrics.py` — abstention, distractor rejection, multi-slot answer, sub-claim metrics.
- `runner.py` — capability-driven scoring, config shape, digest/manifest.
- `__main__.py` — `validate` loads data; new `conformance` subcommand.
- `registry.py` — seven suites.
- `benchmarks/configs/suite.example.yaml`, `benchmarks/README.md`, `benchmarks/pyproject.toml`.
- `benchmarks/fixtures/*` — replaced with released-shape fixtures.

**Delete:** `benchmarks/src/llm_wiki_bench/adapters/rgb.py`, `benchmarks/fixtures/rgb.jsonl`.

---

### Task 1: Capability vocabulary and profile table

**Files:**
- Create: `benchmarks/src/llm_wiki_bench/profiles.py`
- Test: `benchmarks/tests/test_profiles.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Profile` (frozen dataclass with `name: str`, `capabilities: frozenset[str]`), `PROFILES: dict[str, Profile]`, `CAPABILITIES: frozenset[str]`, `get_profile(name: str) -> Profile`, `capability_requirements(capability: str) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_profiles.py
import pytest

from llm_wiki_bench.profiles import (
    CAPABILITIES,
    PROFILES,
    capability_requirements,
    get_profile,
)


def test_every_profile_declares_known_capabilities():
    for profile in PROFILES.values():
        assert profile.capabilities
        assert profile.capabilities <= CAPABILITIES


def test_get_profile_returns_declared_capabilities():
    assert get_profile("grounded_verification").capabilities == frozenset({"label"})


def test_get_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown profile: nope"):
        get_profile("nope")


def test_every_capability_declares_its_required_case_fields():
    for capability in CAPABILITIES:
        assert capability_requirements(capability)


def test_capability_requirements_rejects_unknown_capability():
    with pytest.raises(ValueError, match="unknown capability: nope"):
        capability_requirements("nope")


def test_expected_profiles_are_registered():
    assert set(PROFILES) == {
        "memory_qa",
        "retrieval_qa",
        "multi_slot_retrieval_qa",
        "counterfactual_qa",
        "temporal_discrimination",
        "grounded_verification",
        "claim_decomposition",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_profiles.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_wiki_bench.profiles'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/profiles.py
"""Task profiles declare which capabilities a suite is scored on.

A profile is the suite's contract with the runner. The runner scores exactly
the declared capabilities, so a metric missing from a report is missing because
the profile does not support it, never because a field happened to be empty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    capabilities: frozenset[str]


# Capability -> the BenchmarkCase attributes or label keys the capability needs.
# "labels.x" denotes a required key inside BenchmarkCase.labels.
_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "retrieval": ("evidence_ids",),
    "fine_retrieval": ("fine_evidence_ids",),
    "answer": ("labels.answers",),
    "multi_slot_answer": ("labels.answer_slots",),
    "citations": ("evidence_ids",),
    "abstention": ("expects_abstention",),
    "label": ("labels.label",),
    "distractor_rejection": ("labels.distractor_answers",),
    "sub_claim_labels": ("labels.sub_claims", "labels.sub_claim_labels"),
}

CAPABILITIES = frozenset(_REQUIREMENTS)

PROFILES: dict[str, Profile] = {
    profile.name: profile
    for profile in (
        Profile("memory_qa", frozenset({"retrieval", "fine_retrieval", "answer", "abstention"})),
        Profile("retrieval_qa", frozenset({"retrieval", "answer", "citations"})),
        Profile("multi_slot_retrieval_qa", frozenset({"retrieval", "multi_slot_answer", "citations"})),
        Profile("counterfactual_qa", frozenset({"retrieval", "answer", "distractor_rejection"})),
        Profile("temporal_discrimination", frozenset({"answer", "distractor_rejection"})),
        Profile("grounded_verification", frozenset({"label"})),
        Profile("claim_decomposition", frozenset({"sub_claim_labels"})),
    )
}


def get_profile(name: str) -> Profile:
    """Return a registered profile or fail clearly."""
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown profile: {name}") from error


def capability_requirements(capability: str) -> tuple[str, ...]:
    """Return the case fields a capability requires to be scoreable."""
    try:
        return _REQUIREMENTS[capability]
    except KeyError as error:
        raise ValueError(f"unknown capability: {capability}") from error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_profiles.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/profiles.py benchmarks/tests/test_profiles.py
git commit -m "feat(benchmarks): declare task profiles and capabilities"
```

---

### Task 2: Container readers and content digest

**Files:**
- Create: `benchmarks/src/llm_wiki_bench/readers.py`
- Test: `benchmarks/tests/test_readers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `read_json_array(path: Path) -> Iterator[tuple[int, dict]]`, `read_jsonl(path)`, `read_csv(path)`, `read_parquet(path)` (same signature), `file_digest(path: Path) -> str` returning `"sha256:<hex>"`, and `READERS: dict[str, Callable]` keyed by `"json_array" | "jsonl" | "csv" | "parquet"`.

Record numbers are 1-based and count *records*, not lines.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_readers.py
import json

import pytest

from llm_wiki_bench.readers import (
    READERS,
    file_digest,
    read_csv,
    read_json_array,
    read_jsonl,
    read_parquet,
)


def test_read_json_array_yields_numbered_records(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert list(read_json_array(path)) == [(1, {"a": 1}), (2, {"a": 2})]


def test_read_json_array_rejects_a_top_level_object(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array"):
        list(read_json_array(path))


def test_read_jsonl_skips_blank_lines_without_consuming_a_record_number(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [(1, {"a": 1}), (2, {"a": 2})]


def test_read_jsonl_reports_the_offending_record_number(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="record 2: invalid JSON"):
        list(read_jsonl(path))


def test_read_jsonl_accepts_a_json_extension(tmp_path):
    """RGB ships JSON Lines under a .json extension."""
    path = tmp_path / "en.json"
    path.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")
    assert [record for _, record in read_jsonl(path)] == [{"id": 0}, {"id": 1}]


def test_read_csv_yields_dict_rows(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("ind,claim\n0,hello\n", encoding="utf-8")
    assert list(read_csv(path)) == [(1, {"ind": "0", "claim": "hello"})]


def test_file_digest_is_stable_and_prefixed(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    digest = file_digest(path)
    assert digest.startswith("sha256:")
    assert digest == file_digest(path)
    assert digest == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_readers_registry_exposes_every_container():
    assert set(READERS) == {"json_array", "jsonl", "csv", "parquet"}


def test_read_parquet_without_pyarrow_explains_the_extra(tmp_path, monkeypatch):
    # A None entry in sys.modules makes `import pyarrow.parquet` raise
    # ImportError ("halted; None in sys.modules"), which is how this simulates
    # the extra being absent without uninstalling anything.
    monkeypatch.setitem(__import__("sys").modules, "pyarrow.parquet", None)
    path = tmp_path / "a.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"benchmarks\[hoh\]"):
        list(read_parquet(path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_readers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_wiki_bench.readers'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/readers.py
"""Container readers, separated from field mapping.

Each reader yields ``(record_number, record)`` with 1-based record numbers, so
an adapter reports a malformed record by its position in the source regardless
of the container it came from.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def read_json_array(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read a top-level JSON array of objects (LongMemEval)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: unable to read JSON: {error}") from error
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array")
    for record_number, record in enumerate(payload, start=1):
        yield record_number, _require_object(record, path, record_number)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read one JSON object per line, ignoring blank lines (VitaminC, RGB)."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        record_number = 0
        for line in handle:
            if not line.strip():
                continue
            record_number += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}: record {record_number}: invalid JSON") from error
            yield record_number, _require_object(record, source, record_number)


def read_csv(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read a header-bearing CSV into string-valued dicts (FactLens)."""
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        for record_number, row in enumerate(csv.DictReader(handle), start=1):
            yield record_number, dict(row)


def read_parquet(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read a Parquet table row-wise (HoH); requires the optional extra."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ValueError(
            f"{path}: reading Parquet needs pyarrow; install the optional extra "
            "with: pip install -e 'benchmarks[hoh]'"
        ) from error
    try:
        table = parquet.read_table(str(path))
    except Exception as error:  # pyarrow raises its own error hierarchy
        raise ValueError(f"{path}: unable to read Parquet: {error}") from error
    for record_number, record in enumerate(table.to_pylist(), start=1):
        yield record_number, _require_object(record, Path(path), record_number)


def file_digest(path: Path) -> str:
    """Return ``sha256:<hex>`` over the source bytes actually scored."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


READERS: dict[str, Callable[[Path], Iterator[tuple[int, dict[str, Any]]]]] = {
    "json_array": read_json_array,
    "jsonl": read_jsonl,
    "csv": read_csv,
    "parquet": read_parquet,
}


def _require_object(record: Any, path: Path, record_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{path}: record {record_number}: expected a JSON object")
    return record
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_readers.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/readers.py benchmarks/tests/test_readers.py
git commit -m "feat(benchmarks): read each released container format"
```

---

### Task 3: Case schema carries profile, not per-record provenance

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/schema.py`
- Test: `benchmarks/tests/test_schema.py`

**Interfaces:**
- Consumes: `profiles.get_profile`, `profiles.capability_requirements`.
- Produces: `BenchmarkCase(id, dataset, split, profile, prompt, labels, context=(), evidence_ids=(), fine_evidence_ids=(), expects_abstention=False, metadata={})`; `Prediction(case_id, answer=None, label=None, ranked_evidence_ids=(), cited_evidence_ids=(), sub_claim_labels=(), abstained=False, latency_ms=None)`; `DATASETS` frozenset of the seven suite names; `validate_case`, `validate_prediction`.

`task` is removed. `profile` replaces it, and validation enforces that every capability the profile declares has its data present.

- [ ] **Step 1: Write the failing test**

Append to `benchmarks/tests/test_schema.py`:

```python
def test_case_requires_a_known_profile():
    with pytest.raises(ValueError, match="unknown profile: nope"):
        BenchmarkCase(
            id="c1",
            dataset="vitaminc",
            split="test",
            profile="nope",
            prompt="claim",
            labels={"label": "entailment"},
        )


def test_declared_capability_without_data_is_an_error_not_a_silent_skip():
    """grounded_verification declares `label`; omitting it must fail."""
    with pytest.raises(ValueError, match="grounded_verification requires labels.label"):
        BenchmarkCase(
            id="c1",
            dataset="vitaminc",
            split="test",
            profile="grounded_verification",
            prompt="claim",
            labels={},
        )


def test_retrieval_profile_requires_non_empty_evidence_ids():
    with pytest.raises(ValueError, match="retrieval_qa requires evidence_ids"):
        BenchmarkCase(
            id="c1",
            dataset="rgb_base",
            split="en",
            profile="retrieval_qa",
            prompt="q",
            labels={"answers": ("a",)},
            evidence_ids=(),
        )


def test_memory_qa_case_accepts_two_evidence_granularities():
    case = BenchmarkCase(
        id="q1",
        dataset="longmemeval",
        split="oracle",
        profile="memory_qa",
        prompt="where is the key?",
        labels={"answers": ("the blue vase",)},
        evidence_ids=("s1",),
        fine_evidence_ids=("s1:2",),
        expects_abstention=False,
    )
    assert case.fine_evidence_ids == ("s1:2",)
    assert case.expects_abstention is False


def test_case_no_longer_accepts_a_task_field():
    with pytest.raises(TypeError):
        BenchmarkCase(
            id="c1",
            dataset="vitaminc",
            split="test",
            profile="grounded_verification",
            task="verification",
            prompt="claim",
            labels={"label": "entailment"},
        )


def test_prediction_carries_sub_claim_labels():
    prediction = Prediction(case_id="c1", sub_claim_labels=("true", "false"))
    assert prediction.sub_claim_labels == ("true", "false")


def test_expects_abstention_must_be_boolean():
    with pytest.raises(ValueError, match="expects_abstention must be a boolean"):
        BenchmarkCase(
            id="q1",
            dataset="longmemeval",
            split="oracle",
            profile="memory_qa",
            prompt="q",
            labels={"answers": ("a",)},
            evidence_ids=("s1",),
            fine_evidence_ids=("s1:0",),
            expects_abstention="yes",
        )


def test_seven_suites_are_registered():
    assert DATASETS == frozenset(
        {
            "longmemeval",
            "hoh",
            "vitaminc",
            "factlens",
            "rgb_base",
            "rgb_integration",
            "rgb_counterfactual",
        }
    )
```

Update the existing imports at the top of `test_schema.py` to include `DATASETS`, and update every pre-existing case construction in that file to pass `profile=` instead of `task=`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_schema.py -q`
Expected: FAIL — `TypeError: BenchmarkCase.__init__() got an unexpected keyword argument 'profile'`

- [ ] **Step 3: Write minimal implementation**

In `schema.py`, replace `DATASETS`, the `BenchmarkCase` dataclass, `Prediction`, and `validate_case`:

```python
from .profiles import capability_requirements, get_profile

DATASETS = frozenset(
    {
        "longmemeval",
        "hoh",
        "vitaminc",
        "factlens",
        "rgb_base",
        "rgb_integration",
        "rgb_counterfactual",
    }
)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    dataset: str
    split: str
    profile: str
    prompt: str
    labels: Mapping[str, Any]
    context: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    fine_evidence_ids: tuple[str, ...] = ()
    expects_abstention: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, dict):
            raise ValueError("labels must be a dictionary")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        object.__setattr__(self, "context", _normalize_strings(self.context, "context"))
        object.__setattr__(self, "evidence_ids", _normalize_strings(self.evidence_ids, "evidence_ids"))
        object.__setattr__(
            self, "fine_evidence_ids", _normalize_strings(self.fine_evidence_ids, "fine_evidence_ids")
        )
        object.__setattr__(self, "labels", _freeze(self.labels))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        validate_case(self)
```

Add `sub_claim_labels: tuple[str, ...] = ()` to `Prediction`, normalized in `__post_init__` with
`_normalize_strings(self.sub_claim_labels, "sub_claim_labels")` — note this field allows duplicates, since two sub-claims may share a label, so do not add a uniqueness check for it.

Replace `validate_case`:

```python
def validate_case(case: BenchmarkCase) -> None:
    _require_nonblank(case.id, "id")
    _require_nonblank(case.dataset, "dataset")
    if case.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(sorted(DATASETS))}")
    _require_nonblank(case.split, "split")
    _require_nonblank(case.prompt, "prompt")
    if not isinstance(case.expects_abstention, bool):
        raise ValueError("expects_abstention must be a boolean")
    if len(case.evidence_ids) != len(set(case.evidence_ids)):
        raise ValueError("evidence_ids must not contain duplicates")
    if len(case.fine_evidence_ids) != len(set(case.fine_evidence_ids)):
        raise ValueError("fine_evidence_ids must not contain duplicates")
    _require_declared_capabilities(case)


def _require_declared_capabilities(case: BenchmarkCase) -> None:
    """Fail when a profile declares a capability the case has no data for.

    Scoring must never quietly drop a metric because a field was empty; that is
    how a partial run comes to look like a complete one.
    """
    profile = get_profile(case.profile)
    for capability in sorted(profile.capabilities):
        for requirement in capability_requirements(capability):
            if not _has_requirement(case, requirement):
                raise ValueError(f"{profile.name} requires {requirement} on case {case.id}")


def _has_requirement(case: BenchmarkCase, requirement: str) -> bool:
    if requirement.startswith("labels."):
        value = case.labels.get(requirement.removeprefix("labels."))
        return value is not None and (not isinstance(value, (str, tuple, list)) or len(value) > 0)
    value = getattr(case, requirement)
    if isinstance(value, bool):
        return True
    return bool(value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_schema.py -q`
Expected: PASS. Adapter/runner tests still fail at this point; Tasks 4–12 repair them.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/schema.py benchmarks/tests/test_schema.py
git commit -m "refactor(benchmarks): declare profile capabilities on cases"
```

---

### Task 4: Adapter base declares a reader and returns provenance

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/adapters/base.py`
- Test: `benchmarks/tests/test_adapters.py`

**Interfaces:**
- Consumes: `readers.READERS`, `readers.file_digest`, `schema.BenchmarkCase`.
- Produces: `LoadResult(cases: tuple[BenchmarkCase, ...], content_digest: str, record_count: int)`; `BenchmarkAdapter` with class attributes `name: str`, `profile: str`, `container: str`, `evidence_id_origin: str`, `required: bool = True`, method `load(self, path: Path, *, split: str) -> LoadResult`, and abstract `normalize(self, record, path, record_number, split) -> dict`.

`normalize` returns `BenchmarkCase` keyword arguments excluding `dataset`, `split`, and `profile`, which `load` supplies. There is no per-record `source_version` and no per-record `split` filtering.

- [ ] **Step 1: Write the failing test**

Replace the contents of `benchmarks/tests/test_adapters.py` with a base-contract test (per-suite tests arrive in Tasks 5–9):

```python
import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.base import BenchmarkAdapter, LoadResult


class _Stub(BenchmarkAdapter):
    name = "vitaminc"
    profile = "grounded_verification"
    container = "jsonl"
    evidence_id_origin = "upstream"

    def normalize(self, record, path, record_number, split):
        return {
            "id": str(record["unique_id"]),
            "prompt": record["claim"],
            "labels": {"label": "entailment"},
            "metadata": {},
        }


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"unique_id": "u1", "claim": "c1"}) + "\n"
        + json.dumps({"unique_id": "u2", "claim": "c2"}) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_returns_cases_with_the_configured_split(tmp_path):
    result = _Stub().load(_source(tmp_path), split="test")
    assert isinstance(result, LoadResult)
    assert [case.id for case in result.cases] == ["u1", "u2"]
    assert {case.split for case in result.cases} == {"test"}
    assert {case.profile for case in result.cases} == {"grounded_verification"}


def test_load_reports_digest_and_record_count(tmp_path):
    result = _Stub().load(_source(tmp_path), split="test")
    assert result.record_count == 2
    assert result.content_digest.startswith("sha256:")


def test_load_records_provenance_in_case_metadata(tmp_path):
    source = _source(tmp_path)
    case = _Stub().load(source, split="test").cases[0]
    assert case.metadata["source_path"] == str(source)
    assert case.metadata["source_record"] == 1
    assert "source_version" not in case.metadata


def test_load_rejects_a_blank_split(tmp_path):
    with pytest.raises(ValueError, match="split must be a non-blank string"):
        _Stub().load(_source(tmp_path), split="  ")


def test_load_rejects_an_unknown_container():
    class _Bad(_Stub):
        container = "xml"

    with pytest.raises(ValueError, match="unknown container: xml"):
        _Bad().load(Path("unused.jsonl"), split="test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapters.py -q`
Expected: FAIL — `ImportError: cannot import name 'LoadResult'`

- [ ] **Step 3: Write minimal implementation**

Replace `benchmarks/src/llm_wiki_bench/adapters/base.py`:

```python
"""Adapter base: declare a container, map fields, return provenance.

Provenance is derived, never asserted by the record. No released dataset
carries a per-record version or split, so `load` takes the split from
configuration and computes the digest from the bytes it actually read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_wiki_bench.readers import READERS, file_digest
from llm_wiki_bench.schema import BenchmarkCase


@dataclass(frozen=True)
class LoadResult:
    cases: tuple[BenchmarkCase, ...]
    content_digest: str
    record_count: int


class BenchmarkAdapter(ABC):
    """Translate one released source format into normalized cases."""

    name: str
    profile: str
    container: str
    evidence_id_origin: str
    required: bool = True

    def load(self, path: Path, *, split: str) -> LoadResult:
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split must be a non-blank string")
        try:
            reader = READERS[self.container]
        except KeyError as error:
            raise ValueError(f"unknown container: {self.container}") from error
        source = Path(path)
        cases: list[BenchmarkCase] = []
        record_count = 0
        for record_number, record in reader(source):
            record_count += 1
            values = self.normalize(record, source, record_number, split)
            metadata = dict(values.pop("metadata", {}))
            metadata.update({"source_path": str(source), "source_record": record_number})
            cases.append(
                BenchmarkCase(
                    dataset=self.name,
                    split=split,
                    profile=self.profile,
                    metadata=metadata,
                    **values,
                )
            )
        return LoadResult(tuple(cases), file_digest(source), record_count)

    @abstractmethod
    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        """Return BenchmarkCase fields other than dataset, split, and profile."""

    @staticmethod
    def _required(record: dict[str, Any], key: str, path: Path, record_number: int) -> Any:
        if key not in record:
            raise ValueError(f"{path}: record {record_number}: missing {key}")
        return record[key]

    @staticmethod
    def _metadata(record: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
        return {"source_fields": {key: value for key, value in record.items() if key not in consumed}}
```

Note: `load` computes the digest after reading so a truncated read fails first. `_Bad` in the test never reaches file access because the container lookup happens before reading.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_adapters.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/adapters/base.py benchmarks/tests/test_adapters.py
git commit -m "refactor(benchmarks): derive adapter provenance instead of asserting it"
```

---

### Task 5: LongMemEval adapter against the released JSON array

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/adapters/longmemeval.py`
- Create: `benchmarks/fixtures/longmemeval.json`
- Delete: `benchmarks/fixtures/longmemeval.jsonl`
- Test: `benchmarks/tests/test_adapter_longmemeval.py`

**Interfaces:**
- Consumes: `BenchmarkAdapter`, `LoadResult`.
- Produces: `LongMemEvalAdapter` with `name="longmemeval"`, `profile="memory_qa"`, `container="json_array"`, `evidence_id_origin="upstream"`.

Released fields: `question_id`, `question_type`, `question`, `answer` (a single string), `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (list of sessions; each session is a list of turns `{"role", "content"}` with an optional `"has_answer": true`), `answer_session_ids`. A `question_id` ending in `_abs` is an abstention question.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_longmemeval.py
import json

import pytest

from llm_wiki_bench.adapters.longmemeval import LongMemEvalAdapter

RECORD = {
    "question_id": "gpt4_2655b836",
    "question_type": "temporal-reasoning",
    "question": "What was the first issue with my new car?",
    "answer": "GPS system not functioning correctly",
    "question_date": "2023/05/20 (Sat) 14:31",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_dates": ["2023/05/01", "2023/05/10"],
    "haystack_sessions": [
        [
            {"role": "user", "content": "picked up the car"},
            {"role": "assistant", "content": "the GPS was wrong", "has_answer": True},
        ],
        [{"role": "user", "content": "unrelated chat"}],
    ],
    "answer_session_ids": ["s1"],
}


def _write(tmp_path, records):
    path = tmp_path / "longmemeval_oracle.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_normalizes_the_released_record(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.id == "gpt4_2655b836"
    assert case.dataset == "longmemeval"
    assert case.profile == "memory_qa"
    assert case.prompt == "What was the first issue with my new car?"
    assert case.labels["answers"] == ("GPS system not functioning correctly",)


def test_session_and_turn_evidence_are_both_retained(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.evidence_ids == ("s1",)
    assert case.fine_evidence_ids == ("s1:1",)


def test_context_flattens_sessions_in_order(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.context == (
        "user: picked up the car",
        "assistant: the GPS was wrong",
        "user: unrelated chat",
    )


def test_question_type_and_date_are_kept_as_metadata(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.metadata["question_type"] == "temporal-reasoning"
    assert case.metadata["question_date"] == "2023/05/20 (Sat) 14:31"


def test_abs_suffix_marks_an_abstention_question(tmp_path):
    record = dict(RECORD, question_id="gpt4_2655b836_abs")
    case = LongMemEvalAdapter().load(_write(tmp_path, [record]), split="oracle").cases[0]
    assert case.expects_abstention is True


def test_a_non_abs_question_does_not_expect_abstention(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.expects_abstention is False


def test_session_id_and_session_count_must_agree(tmp_path):
    record = dict(RECORD, haystack_session_ids=["s1"])
    with pytest.raises(ValueError, match="record 1: haystack_session_ids and haystack_sessions differ in length"):
        LongMemEvalAdapter().load(_write(tmp_path, [record]), split="oracle")


def test_missing_released_field_names_the_record(tmp_path):
    record = {key: value for key, value in RECORD.items() if key != "answer"}
    with pytest.raises(ValueError, match="record 1: missing answer"):
        LongMemEvalAdapter().load(_write(tmp_path, [record]), split="oracle")


def test_the_committed_fixture_matches_the_released_shape(tmp_path):
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "longmemeval.json"
    result = LongMemEvalAdapter().load(fixture, split="oracle")
    assert result.record_count == len(json.loads(fixture.read_text(encoding="utf-8")))
    assert result.cases
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_longmemeval.py -q`
Expected: FAIL — `TypeError: normalize() takes 4 positional arguments but 5 were given`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/longmemeval.py
"""LongMemEval adapter for the released JSON array.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Files: longmemeval_s_cleaned.json, longmemeval_m_cleaned.json,
       longmemeval_oracle.json  (MIT)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "answer_session_ids",
}


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"
    profile = "memory_qa"
    container = "json_array"
    evidence_id_origin = "upstream"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        question_id = str(self._required(record, "question_id", path, record_number))
        session_ids = self._required(record, "haystack_session_ids", path, record_number)
        sessions = self._required(record, "haystack_sessions", path, record_number)
        if not isinstance(session_ids, list) or not isinstance(sessions, list):
            raise ValueError(
                f"{path}: record {record_number}: haystack_session_ids and haystack_sessions must be lists"
            )
        if len(session_ids) != len(sessions):
            raise ValueError(
                f"{path}: record {record_number}: haystack_session_ids and haystack_sessions differ in length"
            )

        context: list[str] = []
        fine_evidence_ids: list[str] = []
        for session_id, turns in zip(session_ids, sessions):
            if not isinstance(turns, list):
                raise ValueError(f"{path}: record {record_number}: session {session_id} is not a list of turns")
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"{path}: record {record_number}: session {session_id} turn {turn_index} is not an object"
                    )
                role = self._required(turn, "role", path, record_number)
                content = self._required(turn, "content", path, record_number)
                context.append(f"{role}: {content}")
                if turn.get("has_answer") is True:
                    fine_evidence_ids.append(f"{session_id}:{turn_index}")

        metadata = self._metadata(record, _CONSUMED)
        metadata.update(
            {
                "question_type": record.get("question_type"),
                "question_date": record.get("question_date"),
            }
        )
        return {
            "id": question_id,
            "prompt": self._required(record, "question", path, record_number),
            "context": tuple(context),
            "evidence_ids": tuple(self._required(record, "answer_session_ids", path, record_number)),
            "fine_evidence_ids": tuple(fine_evidence_ids),
            "labels": {"answers": (self._required(record, "answer", path, record_number),)},
            "expects_abstention": question_id.endswith("_abs"),
            "metadata": metadata,
        }
```

- [ ] **Step 4: Write the fixture**

```bash
cat > benchmarks/fixtures/longmemeval.json <<'JSON'
[
  {
    "question_id": "fixture_single_session",
    "question_type": "single-session-user",
    "question": "Where did I leave the spare key?",
    "answer": "in the blue vase",
    "question_date": "2024/03/02 (Sat) 10:00",
    "haystack_session_ids": ["fixture-s1", "fixture-s2"],
    "haystack_dates": ["2024/02/01", "2024/02/20"],
    "haystack_sessions": [
      [
        {"role": "user", "content": "I put the spare key in the blue vase.", "has_answer": true},
        {"role": "assistant", "content": "Noted."}
      ],
      [
        {"role": "user", "content": "The weather was pleasant today."},
        {"role": "assistant", "content": "Glad to hear it."}
      ]
    ],
    "answer_session_ids": ["fixture-s1"]
  },
  {
    "question_id": "fixture_unanswerable_abs",
    "question_type": "single-session-user",
    "question": "Which airline did I fly to Lisbon?",
    "answer": "no information available",
    "question_date": "2024/03/03 (Sun) 09:15",
    "haystack_session_ids": ["fixture-s3"],
    "haystack_dates": ["2024/02/25"],
    "haystack_sessions": [
      [
        {"role": "user", "content": "I renewed my passport."},
        {"role": "assistant", "content": "Good timing."}
      ]
    ],
    "answer_session_ids": ["fixture-s3"]
  }
]
JSON
rm benchmarks/fixtures/longmemeval.jsonl
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_adapter_longmemeval.py -q`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/adapters/longmemeval.py benchmarks/fixtures/longmemeval.json benchmarks/tests/test_adapter_longmemeval.py
git rm --cached benchmarks/fixtures/longmemeval.jsonl 2>/dev/null || true
git add -A benchmarks/fixtures
git commit -m "feat(benchmarks): normalize released LongMemEval records"
```

---

### Task 6: VitaminC adapter with the released label spelling

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/adapters/vitaminc.py`, `benchmarks/fixtures/vitaminc.jsonl`
- Test: `benchmarks/tests/test_adapter_vitaminc.py`

**Interfaces:**
- Produces: `VitaminCAdapter` with `name="vitaminc"`, `profile="grounded_verification"`, `container="jsonl"`, `evidence_id_origin="upstream"`.

Released fields: `unique_id`, `case_id`, `wiki_revision_id`, `label` (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` — **spaces, not underscores**), `claim`, `evidence`, `page`, `revision_type`, `FEVER_id`, `big_bench_canary`. There is no retrievable corpus, so the profile declares `label` only.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_vitaminc.py
import json

import pytest

from llm_wiki_bench.adapters.vitaminc import VitaminCAdapter

RECORD = {
    "unique_id": "5ed4de07c9e77c000848a180_1",
    "case_id": "5ed4de07c9e77c000848a180",
    "wiki_revision_id": "927477259",
    "label": "NOT ENOUGH INFO",
    "claim": "Westlife made under 23.5 million sales in the UK .",
    "evidence": "According to the British Phonographic Industry , Westlife ...",
    "page": "Westlife",
    "revision_type": "real",
    "FEVER_id": "",
    "big_bench_canary": "26b5c67b",
}


def _write(tmp_path, records):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "released,expected",
    [
        ("SUPPORTS", "entailment"),
        ("REFUTES", "contradiction"),
        ("NOT ENOUGH INFO", "neutral"),
    ],
)
def test_released_label_spellings_map(tmp_path, released, expected):
    record = dict(RECORD, label=released)
    case = VitaminCAdapter().load(_write(tmp_path, [record]), split="test").cases[0]
    assert case.labels["label"] == expected


def test_underscored_label_is_not_accepted(tmp_path):
    """The released spelling has spaces; the old adapter only took underscores."""
    record = dict(RECORD, label="NOT_ENOUGH_INFO")
    with pytest.raises(ValueError, match="unsupported VitaminC label 'NOT_ENOUGH_INFO'"):
        VitaminCAdapter().load(_write(tmp_path, [record]), split="test")


def test_evidence_is_context_and_no_retrieval_ids_are_invented(tmp_path):
    case = VitaminCAdapter().load(_write(tmp_path, [RECORD]), split="test").cases[0]
    assert case.id == "5ed4de07c9e77c000848a180_1"
    assert case.prompt == RECORD["claim"]
    assert case.context == (RECORD["evidence"],)
    assert case.evidence_ids == ()


def test_page_and_revision_are_kept_as_metadata(tmp_path):
    case = VitaminCAdapter().load(_write(tmp_path, [RECORD]), split="test").cases[0]
    assert case.metadata["page"] == "Westlife"
    assert case.metadata["wiki_revision_id"] == "927477259"
    assert case.metadata["revision_type"] == "real"


def test_the_committed_fixture_matches_the_released_shape():
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "vitaminc.jsonl"
    result = VitaminCAdapter().load(fixture, split="test")
    assert result.record_count == 3
    assert {case.labels["label"] for case in result.cases} == {
        "entailment",
        "contradiction",
        "neutral",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_vitaminc.py -q`
Expected: FAIL — the current adapter rejects `"NOT ENOUGH INFO"`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/vitaminc.py
"""VitaminC adapter for the released JSONL.

Source: https://huggingface.co/datasets/tals/vitaminc  (CC-BY-SA-3.0)
Files: train.jsonl, dev.jsonl, test.jsonl

Evidence is supplied inline and there is no retrievable corpus, so this suite
scores a label and nothing else. The released label spelling uses spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_LABELS = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral",
}

_CONSUMED = {"unique_id", "claim", "evidence", "label"}


class VitaminCAdapter(BenchmarkAdapter):
    name = "vitaminc"
    profile = "grounded_verification"
    container = "jsonl"
    evidence_id_origin = "upstream"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        label = self._required(record, "label", path, record_number)
        if label not in _LABELS:
            raise ValueError(
                f"{path}: record {record_number}: unsupported VitaminC label {label!r}; "
                f"expected one of {sorted(_LABELS)}"
            )
        return {
            "id": str(self._required(record, "unique_id", path, record_number)),
            "prompt": self._required(record, "claim", path, record_number),
            "context": (self._required(record, "evidence", path, record_number),),
            "labels": {"label": _LABELS[label]},
            "metadata": {
                "case_id": record.get("case_id"),
                "page": record.get("page"),
                "revision_type": record.get("revision_type"),
                "wiki_revision_id": record.get("wiki_revision_id"),
            },
        }
```

- [ ] **Step 4: Write the fixture**

```bash
cat > benchmarks/fixtures/vitaminc.jsonl <<'JSONL'
{"unique_id":"fixture000_1","case_id":"fixture000","wiki_revision_id":"1","label":"SUPPORTS","claim":"The tower is 120 metres tall.","evidence":"The tower stands 120 metres tall.","page":"Fixture Tower","revision_type":"real","FEVER_id":"","big_bench_canary":"fixture"}
{"unique_id":"fixture000_2","case_id":"fixture000","wiki_revision_id":"2","label":"REFUTES","claim":"The tower is 200 metres tall.","evidence":"The tower stands 120 metres tall.","page":"Fixture Tower","revision_type":"real","FEVER_id":"","big_bench_canary":"fixture"}
{"unique_id":"fixture001_1","case_id":"fixture001","wiki_revision_id":"3","label":"NOT ENOUGH INFO","claim":"The tower was repainted in 1998.","evidence":"The tower stands 120 metres tall.","page":"Fixture Tower","revision_type":"synthetic","FEVER_id":"","big_bench_canary":"fixture"}
JSONL
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_adapter_vitaminc.py -q`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/adapters/vitaminc.py benchmarks/fixtures/vitaminc.jsonl benchmarks/tests/test_adapter_vitaminc.py
git commit -m "fix(benchmarks): accept the released VitaminC label spelling"
```

---

### Task 7: RGB base adapter (`en`, `en_refine`)

**Files:**
- Create: `benchmarks/src/llm_wiki_bench/adapters/rgb_base.py`, `benchmarks/fixtures/rgb_base.jsonl`
- Delete: `benchmarks/src/llm_wiki_bench/adapters/rgb.py`, `benchmarks/fixtures/rgb.jsonl`
- Test: `benchmarks/tests/test_adapter_rgb_base.py`

**Interfaces:**
- Produces: `RGBBaseAdapter` with `name="rgb_base"`, `profile="retrieval_qa"`, `container="jsonl"`, `evidence_id_origin="synthesized"`; and the shared helper `synthesize_pool_ids(case_id: str, positive: list[str], negative: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]` returning `(context, evidence_ids, all_ids)`, exported from `rgb_base` and reused by Task 9.

Released fields: `id` (integer), `query`, `answer` as `list[list[str]]` (answer slots, each a list of accepted aliases), `positive` as `list[str]`, `negative` as `list[str]`. Files are JSON Lines despite the `.json` extension. Documents have no upstream identifiers, so identifiers are synthesized as `{case_id}:positive:{index}` and `{case_id}:negative:{index}`, and `context` is emitted in that same order.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_rgb_base.py
import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.rgb_base import RGBBaseAdapter

RECORD = {
    "id": 0,
    "query": "Super Bowl 2021 location",
    "answer": [["Tampa, Florida", "Tampa", "Raymond James Stadium"]],
    "positive": ["The game was played in Tampa, Florida.", "Held at Raymond James Stadium."],
    "negative": ["Super Bowl LVIII will be in Las Vegas.", "Ticket packages now available."],
}


def _write(tmp_path, records, name="en.json"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_normalizes_the_released_record(tmp_path):
    case = RGBBaseAdapter().load(_write(tmp_path, [RECORD]), split="en").cases[0]
    assert case.id == "0"
    assert case.dataset == "rgb_base"
    assert case.profile == "retrieval_qa"
    assert case.prompt == "Super Bowl 2021 location"


def test_answer_slot_aliases_flatten_into_accepted_answers(tmp_path):
    case = RGBBaseAdapter().load(_write(tmp_path, [RECORD]), split="en").cases[0]
    assert case.labels["answers"] == ("Tampa, Florida", "Tampa", "Raymond James Stadium")


def test_a_bare_string_answer_is_also_accepted(tmp_path):
    record = dict(RECORD, answer="Tampa, Florida")
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.labels["answers"] == ("Tampa, Florida",)


def test_a_flat_list_answer_is_also_accepted(tmp_path):
    record = dict(RECORD, answer=["Tampa, Florida", "Tampa"])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.labels["answers"] == ("Tampa, Florida", "Tampa")


def test_document_ids_are_synthesized_and_context_matches_their_order(tmp_path):
    case = RGBBaseAdapter().load(_write(tmp_path, [RECORD]), split="en").cases[0]
    assert case.evidence_ids == ("0:positive:0", "0:positive:1")
    assert case.metadata["candidate_ids"] == (
        "0:positive:0",
        "0:positive:1",
        "0:negative:0",
        "0:negative:1",
    )
    assert case.context == (
        "The game was played in Tampa, Florida.",
        "Held at Raymond James Stadium.",
        "Super Bowl LVIII will be in Las Vegas.",
        "Ticket packages now available.",
    )


def test_a_json_extension_is_read_as_json_lines(tmp_path):
    result = RGBBaseAdapter().load(_write(tmp_path, [RECORD, dict(RECORD, id=1)]), split="en")
    assert result.record_count == 2


def test_empty_positive_pool_is_rejected(tmp_path):
    record = dict(RECORD, positive=[])
    with pytest.raises(ValueError, match="record 1: positive must be a non-empty list"):
        RGBBaseAdapter().load(_write(tmp_path, [record]), split="en")


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_base.jsonl"
    result = RGBBaseAdapter().load(fixture, split="en")
    assert result.record_count == 2
    assert all(case.evidence_ids for case in result.cases)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_rgb_base.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_wiki_bench.adapters.rgb_base'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/rgb_base.py
"""RGB base adapter for en.json and en_refine.json.

Source: https://github.com/chen700564/RGB  (no license declaration)
en_refine is the corrected edition of en and shares its schema. Files are JSON
Lines despite the .json extension.

RGB supplies document pools, not an assembled context. The upstream harness
builds a context from them at a chosen noise_rate/passage_num; those are
parameters of the prediction step, so this adapter emits both pools with
synthesized identifiers and leaves assembly to whatever produced the
predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {"id", "query", "answer", "positive", "negative"}


def flatten_answers(value: Any, path: Path, record_number: int) -> tuple[str, ...]:
    """Accept a string, a list of strings, or slots of aliases.

    en and en_int use list[list[str]]; en_fact uses a bare string.
    """
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: record {record_number}: answer must be a string or a non-empty list")
    answers: list[str] = []
    for item in value:
        if isinstance(item, str):
            answers.append(item)
        elif isinstance(item, list) and all(isinstance(alias, str) for alias in item):
            answers.extend(item)
        else:
            raise ValueError(
                f"{path}: record {record_number}: answer entries must be strings or lists of strings"
            )
    if not answers:
        raise ValueError(f"{path}: record {record_number}: answer must contain at least one string")
    return tuple(answers)


def synthesize_pool_ids(
    case_id: str, positive: list[str], negative: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (context, positive_ids, all_candidate_ids) in a single stable order."""
    positive_ids = tuple(f"{case_id}:positive:{index}" for index in range(len(positive)))
    negative_ids = tuple(f"{case_id}:negative:{index}" for index in range(len(negative)))
    return tuple(positive) + tuple(negative), positive_ids, positive_ids + negative_ids


def require_documents(record: dict[str, Any], key: str, path: Path, record_number: int) -> list[str]:
    value = record.get(key)
    if key == "positive" and (not isinstance(value, list) or not value):
        raise ValueError(f"{path}: record {record_number}: positive must be a non-empty list")
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path}: record {record_number}: {key} must be a list of strings")
    return value


class RGBBaseAdapter(BenchmarkAdapter):
    name = "rgb_base"
    profile = "retrieval_qa"
    container = "jsonl"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        case_id = str(self._required(record, "id", path, record_number))
        positive = require_documents(record, "positive", path, record_number)
        negative = require_documents(record, "negative", path, record_number)
        context, positive_ids, candidate_ids = synthesize_pool_ids(case_id, positive, negative)
        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = candidate_ids
        return {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": context,
            "evidence_ids": positive_ids,
            "labels": {
                "answers": flatten_answers(
                    self._required(record, "answer", path, record_number), path, record_number
                )
            },
            "metadata": metadata,
        }
```

- [ ] **Step 4: Write the fixture and remove the old adapter**

```bash
cat > benchmarks/fixtures/rgb_base.jsonl <<'JSONL'
{"id":0,"query":"Which city hosted the fixture summit?","answer":[["Lisbon","Lisbon, Portugal"]],"positive":["The fixture summit was held in Lisbon.","Delegates gathered in Lisbon, Portugal."],"negative":["The next summit venue is undecided.","Ticketing opens later this year."]}
{"id":1,"query":"Who chaired the fixture committee?","answer":[["Ada Lovelace","A. Lovelace"]],"positive":["Ada Lovelace chaired the fixture committee."],"negative":["The committee met four times.","Minutes are published quarterly."]}
JSONL
git rm benchmarks/src/llm_wiki_bench/adapters/rgb.py benchmarks/fixtures/rgb.jsonl
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_adapter_rgb_base.py -q`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add -A benchmarks/src/llm_wiki_bench/adapters benchmarks/fixtures benchmarks/tests/test_adapter_rgb_base.py
git commit -m "feat(benchmarks): normalize RGB base records from document pools"
```

---

### Task 8: RGB integration adapter (`en_int`)

**Files:**
- Create: `benchmarks/src/llm_wiki_bench/adapters/rgb_integration.py`, `benchmarks/fixtures/rgb_integration.jsonl`
- Test: `benchmarks/tests/test_adapter_rgb_integration.py`

**Interfaces:**
- Consumes: `rgb_base.synthesize_pool_ids`, `rgb_base.require_documents`. It does **not** use `flatten_answers`, because slots must stay separate rather than flatten.
- Produces: `RGBIntegrationAdapter` with `name="rgb_integration"`, `profile="multi_slot_retrieval_qa"`, `container="jsonl"`, `evidence_id_origin="synthesized"`.

Released fields differ from the base file: `answer` is `list[list[str]]` with one entry **per sub-question slot**, `positive` is `list[list[str]]` grouped per sub-question, and the record carries `asnwer1` — an upstream typo, preserved verbatim — alongside `answer2`. The profile scores `multi_slot_answer`, so `labels["answer_slots"]` is a tuple of per-slot alias tuples rather than one flat answer list.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_rgb_integration.py
import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.rgb_integration import RGBIntegrationAdapter

RECORD = {
    "id": 0,
    "query": "When was the summit and who chaired it?",
    "answer": [["January 2 2022", "Jan 2, 2022"], ["Ada Lovelace", "A. Lovelace"]],
    "asnwer1": "January 2 2022",
    "answer2": "Ada Lovelace",
    "positive": [
        ["The summit opened on January 2 2022."],
        ["Ada Lovelace chaired the summit."],
    ],
    "negative": ["Unrelated conference news.", "Venue rumours."],
}


def _write(tmp_path, records):
    path = tmp_path / "en_int.json"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_each_answer_slot_is_kept_separate(tmp_path):
    case = RGBIntegrationAdapter().load(_write(tmp_path, [RECORD]), split="en_int").cases[0]
    assert case.profile == "multi_slot_retrieval_qa"
    assert case.labels["answer_slots"] == (
        ("January 2 2022", "Jan 2, 2022"),
        ("Ada Lovelace", "A. Lovelace"),
    )


def test_grouped_positive_documents_flatten_in_order(tmp_path):
    case = RGBIntegrationAdapter().load(_write(tmp_path, [RECORD]), split="en_int").cases[0]
    assert case.context[:2] == (
        "The summit opened on January 2 2022.",
        "Ada Lovelace chaired the summit.",
    )
    assert case.evidence_ids == ("0:positive:0", "0:positive:1")


def test_the_upstream_asnwer1_typo_is_preserved_as_metadata(tmp_path):
    case = RGBIntegrationAdapter().load(_write(tmp_path, [RECORD]), split="en_int").cases[0]
    assert case.metadata["source_fields"]["asnwer1"] == "January 2 2022"
    assert case.metadata["source_fields"]["answer2"] == "Ada Lovelace"


def test_a_flat_positive_list_is_also_accepted(tmp_path):
    record = dict(RECORD, positive=["One.", "Two."])
    case = RGBIntegrationAdapter().load(_write(tmp_path, [record]), split="en_int").cases[0]
    assert case.context[:2] == ("One.", "Two.")


def test_a_single_slot_answer_is_rejected(tmp_path):
    record = dict(RECORD, answer=[["only one slot"]])
    with pytest.raises(ValueError, match="record 1: answer must declare at least two slots"):
        RGBIntegrationAdapter().load(_write(tmp_path, [record]), split="en_int")


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_integration.jsonl"
    result = RGBIntegrationAdapter().load(fixture, split="en_int")
    assert result.record_count == 1
    assert len(result.cases[0].labels["answer_slots"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_rgb_integration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_wiki_bench.adapters.rgb_integration'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/rgb_integration.py
"""RGB information-integration adapter for en_int.json.

Source: https://github.com/chen700564/RGB  (no license declaration)

This variant does not share the base schema: answer carries one slot per
sub-question and positive is grouped per sub-question. The record also carries
an upstream field-name typo, `asnwer1`, which is preserved verbatim rather than
corrected, so the normalized record stays traceable to the release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter
from .rgb_base import require_documents, synthesize_pool_ids

_CONSUMED = {"id", "query", "answer", "positive", "negative"}


class RGBIntegrationAdapter(BenchmarkAdapter):
    name = "rgb_integration"
    profile = "multi_slot_retrieval_qa"
    container = "jsonl"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        case_id = str(self._required(record, "id", path, record_number))
        slots = self._answer_slots(record, path, record_number)
        positive = self._flatten_positive(record, path, record_number)
        negative = require_documents(record, "negative", path, record_number)
        context, positive_ids, candidate_ids = synthesize_pool_ids(case_id, positive, negative)
        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = candidate_ids
        return {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": context,
            "evidence_ids": positive_ids,
            "labels": {"answer_slots": slots},
            "metadata": metadata,
        }

    def _answer_slots(
        self, record: dict[str, Any], path: Path, record_number: int
    ) -> tuple[tuple[str, ...], ...]:
        value = self._required(record, "answer", path, record_number)
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"{path}: record {record_number}: answer must declare at least two slots")
        slots: list[tuple[str, ...]] = []
        for slot in value:
            if isinstance(slot, str):
                slots.append((slot,))
            elif isinstance(slot, list) and slot and all(isinstance(alias, str) for alias in slot):
                slots.append(tuple(slot))
            else:
                raise ValueError(
                    f"{path}: record {record_number}: each answer slot must be a string or a non-empty list of strings"
                )
        return tuple(slots)

    def _flatten_positive(self, record: dict[str, Any], path: Path, record_number: int) -> list[str]:
        value = record.get("positive")
        if not isinstance(value, list) or not value:
            raise ValueError(f"{path}: record {record_number}: positive must be a non-empty list")
        if all(isinstance(item, str) for item in value):
            return list(value)
        documents: list[str] = []
        for group in value:
            if not isinstance(group, list) or any(not isinstance(item, str) for item in group):
                raise ValueError(
                    f"{path}: record {record_number}: positive must be strings or lists of strings"
                )
            documents.extend(group)
        if not documents:
            raise ValueError(f"{path}: record {record_number}: positive must be a non-empty list")
        return documents
```

- [ ] **Step 4: Write the fixture**

```bash
cat > benchmarks/fixtures/rgb_integration.jsonl <<'JSONL'
{"id":0,"query":"When did the fixture summit open and who chaired it?","answer":[["January 2 2022","Jan 2, 2022"],["Ada Lovelace","A. Lovelace"]],"asnwer1":"January 2 2022","answer2":"Ada Lovelace","positive":[["The fixture summit opened on January 2 2022."],["Ada Lovelace chaired the fixture summit."]],"negative":["Unrelated conference news.","Venue rumours persist."]}
JSONL
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_adapter_rgb_integration.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/adapters/rgb_integration.py benchmarks/fixtures/rgb_integration.jsonl benchmarks/tests/test_adapter_rgb_integration.py
git commit -m "feat(benchmarks): normalize RGB integration slots and grouped documents"
```

---

### Task 9: RGB counterfactual adapter (`en_fact`)

**Files:**
- Create: `benchmarks/src/llm_wiki_bench/adapters/rgb_counterfactual.py`, `benchmarks/fixtures/rgb_counterfactual.jsonl`
- Test: `benchmarks/tests/test_adapter_rgb_counterfactual.py`

**Interfaces:**
- Consumes: `rgb_base.flatten_answers`, `rgb_base.require_documents`. It does **not** use `synthesize_pool_ids`, because it has three pools rather than two and builds the identifiers itself.
- Produces: `RGBCounterfactualAdapter` with `name="rgb_counterfactual"`, `profile="counterfactual_qa"`, `container="jsonl"`, `evidence_id_origin="synthesized"`.

Released fields: `answer` is a **bare string** here, plus `fakeanswer` (a string) and `positive_wrong` (documents supporting the fake answer). `labels["distractor_answers"]` carries the fake answer so the `distractor_rejection` capability can score whether the prediction reproduced it. `positive_wrong` documents get `{case_id}:positive_wrong:{index}` identifiers and are **not** counted as expected evidence.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_rgb_counterfactual.py
import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.rgb_counterfactual import RGBCounterfactualAdapter

RECORD = {
    "id": 0,
    "query": "Super Bowl 2021 location",
    "answer": "Tampa, Florida",
    "fakeanswer": "Glendale, Arizona",
    "positive": ["The game was played in Tampa, Florida."],
    "positive_wrong": ["The game was played in Glendale, Arizona."],
    "negative": ["Ticket packages now available."],
}


def _write(tmp_path, records):
    path = tmp_path / "en_fact.json"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_a_bare_string_answer_is_normalized(tmp_path):
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [RECORD]), split="en_fact").cases[0]
    assert case.profile == "counterfactual_qa"
    assert case.labels["answers"] == ("Tampa, Florida",)


def test_the_fake_answer_becomes_the_distractor(tmp_path):
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [RECORD]), split="en_fact").cases[0]
    assert case.labels["distractor_answers"] == ("Glendale, Arizona",)


def test_positive_wrong_documents_are_candidates_but_not_expected_evidence(tmp_path):
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [RECORD]), split="en_fact").cases[0]
    assert case.evidence_ids == ("0:positive:0",)
    assert case.metadata["candidate_ids"] == (
        "0:positive:0",
        "0:negative:0",
        "0:positive_wrong:0",
    )
    assert case.context == (
        "The game was played in Tampa, Florida.",
        "Ticket packages now available.",
        "The game was played in Glendale, Arizona.",
    )


def test_a_missing_fakeanswer_is_rejected(tmp_path):
    record = {key: value for key, value in RECORD.items() if key != "fakeanswer"}
    with pytest.raises(ValueError, match="record 1: missing fakeanswer"):
        RGBCounterfactualAdapter().load(_write(tmp_path, [record]), split="en_fact")


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_counterfactual.jsonl"
    result = RGBCounterfactualAdapter().load(fixture, split="en_fact")
    assert result.record_count == 1
    assert result.cases[0].labels["distractor_answers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_rgb_counterfactual.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_wiki_bench.adapters.rgb_counterfactual'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/rgb_counterfactual.py
"""RGB counterfactual-robustness adapter for en_fact.json.

Source: https://github.com/chen700564/RGB  (no license declaration)

This variant carries a bare-string `answer`, a `fakeanswer`, and
`positive_wrong` documents that support the fake answer. Reproducing the fake
answer is a distinct failure from being merely wrong, so the fake answer is
scored as a distractor rather than folded into the accepted answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter
from .rgb_base import flatten_answers, require_documents

_CONSUMED = {"id", "query", "answer", "fakeanswer", "positive", "positive_wrong", "negative"}


class RGBCounterfactualAdapter(BenchmarkAdapter):
    name = "rgb_counterfactual"
    profile = "counterfactual_qa"
    container = "jsonl"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        case_id = str(self._required(record, "id", path, record_number))
        positive = require_documents(record, "positive", path, record_number)
        negative = require_documents(record, "negative", path, record_number)
        positive_wrong = require_documents(record, "positive_wrong", path, record_number)

        positive_ids = tuple(f"{case_id}:positive:{index}" for index in range(len(positive)))
        negative_ids = tuple(f"{case_id}:negative:{index}" for index in range(len(negative)))
        wrong_ids = tuple(f"{case_id}:positive_wrong:{index}" for index in range(len(positive_wrong)))

        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = positive_ids + negative_ids + wrong_ids
        return {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": tuple(positive) + tuple(negative) + tuple(positive_wrong),
            "evidence_ids": positive_ids,
            "labels": {
                "answers": flatten_answers(
                    self._required(record, "answer", path, record_number), path, record_number
                ),
                "distractor_answers": flatten_answers(
                    self._required(record, "fakeanswer", path, record_number), path, record_number
                ),
            },
            "metadata": metadata,
        }
```

- [ ] **Step 4: Write the fixture**

```bash
cat > benchmarks/fixtures/rgb_counterfactual.jsonl <<'JSONL'
{"id":0,"query":"Where was the fixture summit held?","answer":"Lisbon","fakeanswer":"Madrid","positive":["The fixture summit was held in Lisbon."],"positive_wrong":["The fixture summit was held in Madrid."],"negative":["Ticketing details were published separately."]}
JSONL
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_adapter_rgb_counterfactual.py -q`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/adapters/rgb_counterfactual.py benchmarks/fixtures/rgb_counterfactual.jsonl benchmarks/tests/test_adapter_rgb_counterfactual.py
git commit -m "feat(benchmarks): score RGB counterfactual answers against fakeanswer"
```

---

### Task 10: HoH adapter over Parquet

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/adapters/hoh.py`, `benchmarks/pyproject.toml`
- Delete: `benchmarks/fixtures/hoh.jsonl`
- Test: `benchmarks/tests/test_adapter_hoh.py`

**Interfaces:**
- Produces: `HoHAdapter` with `name="hoh"`, `profile="temporal_discrimination"`, `container="parquet"`, `evidence_id_origin="synthesized"`.

Released fields (`russwest404/HoH-QAs`, `hoh_qas_240601_241201.parquet`, Apache-2.0): `question`, `answer`, `last_modified_time` (timestamp), `evidence` (the current statement), `outdated_infos` (list of `{"answer", "evidence", ...}`), `document` (`{"id", "title"}`). **There is no record identifier**, so the case id is `{document.id}:{record_number}`.

The Parquet fixture is generated in the test rather than committed, so no binary lands in the repository and the test skips cleanly without `pyarrow`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_hoh.py
import pytest

from llm_wiki_bench.adapters.hoh import HoHAdapter

pytest.importorskip("pyarrow", reason="HoH needs the optional 'hoh' extra")

RECORD = {
    "question": "Which yeast ferments gluconolactone?",
    "answer": "Maudiozyma bulderi",
    "last_modified_time": "2024-07-01T00:00:00",
    "evidence": 'The yeast "Maudiozyma bulderi" ferments gluconolactone.',
    "outdated_infos": [
        {"answer": "Saccharomyces bulderi", "evidence": 'The yeast "Saccharomyces bulderi" ferments gluconolactone.'}
    ],
    "document": {"id": "1000005", "title": "Glucono delta-lactone"},
}


def _write(tmp_path, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "hoh.parquet"
    pq.write_table(pa.Table.from_pylist(records), str(path))
    return path


def test_normalizes_the_released_record(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="240601_241201").cases[0]
    assert case.dataset == "hoh"
    assert case.profile == "temporal_discrimination"
    assert case.prompt == "Which yeast ferments gluconolactone?"
    assert case.labels["answers"] == ("Maudiozyma bulderi",)


def test_the_case_id_is_synthesized_from_document_and_record_number(tmp_path):
    """HoH has no record identifier of its own."""
    cases = HoHAdapter().load(_write(tmp_path, [RECORD, RECORD]), split="s").cases
    assert [case.id for case in cases] == ["1000005:1", "1000005:2"]


def test_outdated_answers_become_distractors(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert case.labels["distractor_answers"] == ("Saccharomyces bulderi",)


def test_current_evidence_precedes_outdated_evidence_in_context(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert case.context == (
        'The yeast "Maudiozyma bulderi" ferments gluconolactone.',
        'The yeast "Saccharomyces bulderi" ferments gluconolactone.',
    )


def test_no_retrieval_metrics_are_claimed(tmp_path):
    """temporal_discrimination does not declare `retrieval`."""
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert case.evidence_ids == ()


def test_last_modified_time_is_stringified_for_json_artifacts(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert isinstance(case.metadata["last_modified_time"], str)


def test_an_empty_outdated_infos_list_is_rejected(tmp_path):
    record = dict(RECORD, outdated_infos=[])
    with pytest.raises(ValueError, match="record 1: outdated_infos must be a non-empty list"):
        HoHAdapter().load(_write(tmp_path, [record]), split="s")


def test_a_missing_document_id_is_rejected(tmp_path):
    record = dict(RECORD, document={"title": "no id"})
    with pytest.raises(ValueError, match="record 1: document.id is required"):
        HoHAdapter().load(_write(tmp_path, [record]), split="s")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_hoh.py -q`
Expected: FAIL — the current adapter requires `passages`/`split`/`source_version`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/hoh.py
"""HoH adapter for the released Parquet table.

Source: https://huggingface.co/datasets/russwest404/HoH-QAs  (Apache-2.0)
File: hoh_qas_240601_241201.parquet

HoH measures whether a system distinguishes current evidence from the outdated
variants in `outdated_infos`; it is not multi-hop QA. Reproducing an outdated
answer is scored as a distractor failure rather than as being merely wrong.

The release carries no record identifier, so the case id combines
`document.id` with the record number. That makes identifiers stable only for a
fixed file, which is why the manifest records the source digest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {"question", "answer", "evidence", "outdated_infos", "document"}


class HoHAdapter(BenchmarkAdapter):
    name = "hoh"
    profile = "temporal_discrimination"
    container = "parquet"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        document = self._required(record, "document", path, record_number)
        if not isinstance(document, dict) or not str(document.get("id") or "").strip():
            raise ValueError(f"{path}: record {record_number}: document.id is required")
        outdated = self._required(record, "outdated_infos", path, record_number)
        if not isinstance(outdated, list) or not outdated:
            raise ValueError(f"{path}: record {record_number}: outdated_infos must be a non-empty list")

        distractors: list[str] = []
        outdated_evidence: list[str] = []
        for index, entry in enumerate(outdated):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: record {record_number}: outdated_infos[{index}] is not an object")
            distractors.append(str(self._required(entry, "answer", path, record_number)))
            outdated_evidence.append(str(self._required(entry, "evidence", path, record_number)))

        metadata = self._metadata(record, _CONSUMED)
        metadata.update(
            {
                "document_id": str(document["id"]),
                "document_title": document.get("title"),
                "last_modified_time": str(record.get("last_modified_time")),
                "outdated_count": len(outdated),
            }
        )
        metadata["source_fields"] = {
            key: str(value) for key, value in metadata.get("source_fields", {}).items()
        }
        return {
            "id": f"{document['id']}:{record_number}",
            "prompt": self._required(record, "question", path, record_number),
            "context": (self._required(record, "evidence", path, record_number), *outdated_evidence),
            "labels": {
                "answers": (self._required(record, "answer", path, record_number),),
                "distractor_answers": tuple(distractors),
            },
            "metadata": metadata,
        }
```

- [ ] **Step 4: Declare the optional extra and drop the stale fixture**

In `benchmarks/pyproject.toml`, under `[project.optional-dependencies]`, add `hoh = ["pyarrow>=15"]` next to the existing `dev` entry, and extend `dev` to `["pytest>=8", "pyarrow>=15"]` so the HoH test runs in CI rather than skipping.

```bash
git rm benchmarks/fixtures/hoh.jsonl
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pip install -e '.[dev]' && pytest tests/test_adapter_hoh.py -q`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add -A benchmarks/src/llm_wiki_bench/adapters/hoh.py benchmarks/pyproject.toml benchmarks/fixtures benchmarks/tests/test_adapter_hoh.py
git commit -m "feat(benchmarks): normalize HoH from released Parquet"
```

---

### Task 11: FactLens adapter over CSV

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/adapters/factlens.py`
- Create: `benchmarks/fixtures/factlens.csv`
- Delete: `benchmarks/fixtures/factlens.jsonl`
- Test: `benchmarks/tests/test_adapter_factlens.py`

**Interfaces:**
- Produces: `FactLensAdapter` with `name="factlens"`, `profile="claim_decomposition"`, `container="csv"`, `evidence_id_origin="upstream"`, `required=False`.

Released columns (`megagonlabs/factlens`, `benchmark/fact_lens_benchmark.csv`, BSD-3-Clause): `ind`, `claim`, `sub_claims`, `labels`, `aggregated_label`. `sub_claims` and `labels` are **Python-repr lists of strings**, not JSON, so they are parsed with `ast.literal_eval`. There is no evidence, so the profile declares `sub_claim_labels` only.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_adapter_factlens.py
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.factlens import FactLensAdapter

HEADER = "ind,claim,sub_claims,labels,aggregated_label\n"
ROW = (
    '0,"A represented Munich, while B represented London.",'
    '"[\'A represented Munich\', \'B represented London\']",'
    '"[\'true\', \'false\']",False\n'
)


def _write(tmp_path, body):
    path = tmp_path / "fact_lens_benchmark.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_python_repr_lists_are_parsed(tmp_path):
    case = FactLensAdapter().load(_write(tmp_path, ROW), split="benchmark").cases[0]
    assert case.profile == "claim_decomposition"
    assert case.labels["sub_claims"] == ("A represented Munich", "B represented London")
    assert case.labels["sub_claim_labels"] == ("true", "false")


def test_the_aggregated_label_is_normalized_to_a_boolean(tmp_path):
    case = FactLensAdapter().load(_write(tmp_path, ROW), split="benchmark").cases[0]
    assert case.labels["aggregated_label"] is False


def test_the_claim_is_the_prompt_and_no_evidence_is_invented(tmp_path):
    case = FactLensAdapter().load(_write(tmp_path, ROW), split="benchmark").cases[0]
    assert case.id == "0"
    assert case.prompt.startswith("A represented Munich")
    assert case.context == ()
    assert case.evidence_ids == ()


def test_sub_claims_and_labels_must_be_the_same_length(tmp_path):
    body = (
        '0,"claim","[\'one\', \'two\']","[\'true\']",True\n'
    )
    with pytest.raises(ValueError, match="record 1: sub_claims and labels differ in length"):
        FactLensAdapter().load(_write(tmp_path, body), split="benchmark")


def test_a_malformed_repr_list_names_the_record(tmp_path):
    body = '0,"claim","not a list","[\'true\']",True\n'
    with pytest.raises(ValueError, match="record 1: sub_claims is not a list of strings"):
        FactLensAdapter().load(_write(tmp_path, body), split="benchmark")


def test_factlens_stays_optional():
    assert FactLensAdapter().required is False


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "factlens.csv"
    result = FactLensAdapter().load(fixture, split="benchmark")
    assert result.record_count == 2
    assert all(case.labels["sub_claims"] for case in result.cases)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_adapter_factlens.py -q`
Expected: FAIL — the current adapter expects `verdict`/`source`/`source_id`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/adapters/factlens.py
"""Optional FactLens adapter for the released CSV.

Source: https://github.com/megagonlabs/factlens  (BSD-3-Clause)
File: benchmark/fact_lens_benchmark.csv

The benchmark carries no evidence: it measures decomposition of a complex claim
into sub-claims and the per-sub-claim verdicts. `sub_claims` and `labels` are
Python-repr lists rather than JSON, so they are parsed with ast.literal_eval.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {"ind", "claim", "sub_claims", "labels", "aggregated_label"}


class FactLensAdapter(BenchmarkAdapter):
    name = "factlens"
    profile = "claim_decomposition"
    container = "csv"
    evidence_id_origin = "upstream"
    required = False

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        sub_claims = self._repr_list(record, "sub_claims", path, record_number)
        labels = self._repr_list(record, "labels", path, record_number)
        if len(sub_claims) != len(labels):
            raise ValueError(f"{path}: record {record_number}: sub_claims and labels differ in length")
        if not sub_claims:
            raise ValueError(f"{path}: record {record_number}: sub_claims must not be empty")
        return {
            "id": str(self._required(record, "ind", path, record_number)),
            "prompt": self._required(record, "claim", path, record_number),
            "labels": {
                "sub_claims": sub_claims,
                "sub_claim_labels": labels,
                "aggregated_label": self._boolean(record, path, record_number),
            },
            "metadata": self._metadata(record, _CONSUMED),
        }

    def _repr_list(
        self, record: dict[str, Any], key: str, path: Path, record_number: int
    ) -> tuple[str, ...]:
        raw = self._required(record, key, path, record_number)
        try:
            value = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{path}: record {record_number}: {key} is not a list of strings") from error
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{path}: record {record_number}: {key} is not a list of strings")
        return tuple(value)

    def _boolean(self, record: dict[str, Any], path: Path, record_number: int) -> bool:
        raw = self._required(record, "aggregated_label", path, record_number)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
            return raw.strip().lower() == "true"
        raise ValueError(f"{path}: record {record_number}: aggregated_label must be True or False")
```

- [ ] **Step 4: Write the fixture**

```bash
cat > benchmarks/fixtures/factlens.csv <<'CSV'
ind,claim,sub_claims,labels,aggregated_label
0,"The tower is 120 metres tall and was opened in 1961.","['The tower is 120 metres tall', 'The tower was opened in 1961']","['true', 'true']",True
1,"The stadium holds 25,000 people and is the home of the fixture team.","['The stadium holds 25,000 people', 'The stadium is the home of the fixture team']","['false', 'true']",False
CSV
rm benchmarks/fixtures/factlens.jsonl
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_adapter_factlens.py -q`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add -A benchmarks/src/llm_wiki_bench/adapters/factlens.py benchmarks/fixtures benchmarks/tests/test_adapter_factlens.py
git commit -m "feat(benchmarks): normalize FactLens sub-claims from released CSV"
```

---

### Task 12: Registry lists seven suites

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/registry.py`, `benchmarks/src/llm_wiki_bench/adapters/__init__.py`
- Test: `benchmarks/tests/test_registry.py`

**Interfaces:**
- Produces: `get_adapter(name: str) -> BenchmarkAdapter`, `enabled_adapters(config: dict) -> list[BenchmarkAdapter]`, `REQUIRED_SUITES: tuple[str, ...]` (the six `required=True` suite names, sorted), `OPTIONAL_SUITES: tuple[str, ...]` (`("factlens",)`).

`REQUIRED_SUITES` moves out of `runner.py` so the registry is the single source of truth.

- [ ] **Step 1: Write the failing test**

Replace `benchmarks/tests/test_registry.py`:

```python
import pytest

from llm_wiki_bench.registry import (
    OPTIONAL_SUITES,
    REQUIRED_SUITES,
    enabled_adapters,
    get_adapter,
)


def test_every_suite_is_registered():
    for name in (*REQUIRED_SUITES, *OPTIONAL_SUITES):
        assert get_adapter(name).name == name


def test_required_suites_are_the_six_non_optional_ones():
    assert REQUIRED_SUITES == (
        "hoh",
        "longmemeval",
        "rgb_base",
        "rgb_counterfactual",
        "rgb_integration",
        "vitaminc",
    )


def test_factlens_is_the_only_optional_suite():
    assert OPTIONAL_SUITES == ("factlens",)
    assert get_adapter("factlens").required is False


def test_every_adapter_declares_a_profile_and_container():
    for name in (*REQUIRED_SUITES, *OPTIONAL_SUITES):
        adapter = get_adapter(name)
        assert adapter.profile
        assert adapter.container in {"json_array", "jsonl", "csv", "parquet"}
        assert adapter.evidence_id_origin in {"upstream", "synthesized"}


def test_unknown_suite_fails_clearly():
    with pytest.raises(ValueError, match="unknown benchmark adapter: rgb"):
        get_adapter("rgb")


def test_factlens_is_enabled_only_when_configured():
    without = enabled_adapters({"datasets": {}})
    assert [adapter.name for adapter in without] == list(REQUIRED_SUITES)
    with_factlens = enabled_adapters(
        {"datasets": {"factlens": {"path": "benchmarks/fixtures/factlens.csv"}}}
    )
    assert "factlens" in [adapter.name for adapter in with_factlens]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_registry.py -q`
Expected: FAIL — `ImportError: cannot import name 'REQUIRED_SUITES'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/src/llm_wiki_bench/registry.py
"""Named benchmark adapter discovery.

The registry is the single source of truth for which suites exist; the runner
no longer keeps its own list.
"""

from __future__ import annotations

from typing import Any

from .adapters import (
    BenchmarkAdapter,
    FactLensAdapter,
    HoHAdapter,
    LongMemEvalAdapter,
    RGBBaseAdapter,
    RGBCounterfactualAdapter,
    RGBIntegrationAdapter,
    VitaminCAdapter,
)

_ADAPTERS: dict[str, BenchmarkAdapter] = {
    adapter.name: adapter
    for adapter in (
        LongMemEvalAdapter(),
        HoHAdapter(),
        VitaminCAdapter(),
        RGBBaseAdapter(),
        RGBIntegrationAdapter(),
        RGBCounterfactualAdapter(),
        FactLensAdapter(),
    )
}

REQUIRED_SUITES: tuple[str, ...] = tuple(
    sorted(name for name, adapter in _ADAPTERS.items() if adapter.required)
)
OPTIONAL_SUITES: tuple[str, ...] = tuple(
    sorted(name for name, adapter in _ADAPTERS.items() if not adapter.required)
)


def get_adapter(name: str) -> BenchmarkAdapter:
    """Return a registered adapter or fail clearly for an unsupported suite."""
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark adapter: {name}") from error


def enabled_adapters(config: dict[str, Any]) -> list[BenchmarkAdapter]:
    """Return required suites plus each optional suite that has a configured path."""
    adapters = [get_adapter(name) for name in REQUIRED_SUITES]
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        return adapters
    for name in OPTIONAL_SUITES:
        details = datasets.get(name, {})
        if isinstance(details, dict) and details.get("path"):
            adapters.append(get_adapter(name))
    return adapters
```

Update `adapters/__init__.py` to import and export `RGBBaseAdapter`, `RGBIntegrationAdapter`, and `RGBCounterfactualAdapter` in place of `RGBAdapter`, keeping `__all__` sorted.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_registry.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/registry.py benchmarks/src/llm_wiki_bench/adapters/__init__.py benchmarks/tests/test_registry.py
git commit -m "refactor(benchmarks): make the registry the suite source of truth"
```

---

### Task 13: Metrics for the new capabilities

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/metrics.py`
- Test: `benchmarks/tests/test_metrics.py`

**Interfaces:**
- Consumes: existing `score_retrieval`, `score_answer`, `score_citations`, `score_label`, `_normalize_text`.
- Produces:
  - `score_abstention(expects_abstention: bool, abstained: bool) -> dict[str, float]` emitting per-case `abstention_tp`, `abstention_fp`, `abstention_fn` as `0.0`/`1.0`.
  - `abstention_summary(rows: list[dict]) -> dict[str, float | None]` returning corpus-level `abstention_precision`, `abstention_recall`, `abstention_f1`.
  - `score_distractor_rejection(distractor_answers: tuple[str, ...], answer: str | None) -> dict[str, float]` emitting `distractor_rejection`.
  - `score_multi_slot_answer(answer_slots: tuple[tuple[str, ...], ...], answer: str | None) -> dict[str, float]` emitting `slot_coverage` and `all_slots_matched`.
  - `score_sub_claims(expected: tuple[str, ...], predicted: tuple[str, ...]) -> dict[str, float | None]` emitting `sub_claim_accuracy` and `sub_claim_macro_f1`.

Precision, recall, and F1 cannot be averaged per case, so abstention follows the existing corpus-level pattern already used by `label_confusion_matrix`: per-case counters, summarized once.

- [ ] **Step 1: Write the failing test**

Append to `benchmarks/tests/test_metrics.py`:

```python
from llm_wiki_bench.metrics import (
    abstention_summary,
    score_abstention,
    score_distractor_rejection,
    score_multi_slot_answer,
    score_sub_claims,
)


def test_abstention_true_positive():
    assert score_abstention(True, True) == {
        "abstention_tp": 1.0,
        "abstention_fp": 0.0,
        "abstention_fn": 0.0,
    }


def test_abstention_false_positive_when_answering_was_expected():
    assert score_abstention(False, True)["abstention_fp"] == 1.0


def test_abstention_false_negative_when_abstaining_was_expected():
    assert score_abstention(True, False)["abstention_fn"] == 1.0


def test_abstention_summary_computes_corpus_level_rates():
    rows = [
        score_abstention(True, True),
        score_abstention(True, False),
        score_abstention(False, True),
    ]
    summary = abstention_summary(rows)
    assert summary["abstention_precision"] == 0.5
    assert summary["abstention_recall"] == 0.5
    assert summary["abstention_f1"] == 0.5


def test_abstention_summary_is_undefined_without_any_abstention():
    summary = abstention_summary([score_abstention(False, False)])
    assert summary["abstention_precision"] is None
    assert summary["abstention_recall"] is None
    assert summary["abstention_f1"] is None


def test_distractor_rejection_rewards_avoiding_the_outdated_answer():
    assert score_distractor_rejection(("Saccharomyces bulderi",), "Maudiozyma bulderi") == {
        "distractor_rejection": 1.0
    }


def test_distractor_rejection_penalizes_reproducing_it():
    assert score_distractor_rejection(("Saccharomyces bulderi",), "saccharomyces  BULDERI") == {
        "distractor_rejection": 0.0
    }


def test_distractor_rejection_of_no_answer_is_a_rejection():
    assert score_distractor_rejection(("x",), None) == {"distractor_rejection": 1.0}


def test_multi_slot_answer_counts_matched_slots():
    scores = score_multi_slot_answer(
        (("January 2 2022", "Jan 2, 2022"), ("Ada Lovelace",)),
        "On Jan 2, 2022 Ada Lovelace opened it",
    )
    assert scores == {"slot_coverage": 1.0, "all_slots_matched": 1.0}


def test_multi_slot_answer_reports_partial_coverage():
    scores = score_multi_slot_answer(
        (("January 2 2022",), ("Ada Lovelace",)), "Ada Lovelace opened it"
    )
    assert scores == {"slot_coverage": 0.5, "all_slots_matched": 0.0}


def test_sub_claim_accuracy_and_macro_f1():
    scores = score_sub_claims(("true", "false", "true"), ("true", "true", "true"))
    assert scores["sub_claim_accuracy"] == pytest.approx(2 / 3)
    assert scores["sub_claim_macro_f1"] == pytest.approx(0.4)


def test_sub_claims_of_differing_length_are_undefined():
    scores = score_sub_claims(("true", "false"), ("true",))
    assert scores["sub_claim_accuracy"] is None
    assert scores["sub_claim_macro_f1"] is None
```

Ensure `import pytest` is present at the top of `test_metrics.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'abstention_summary'`

- [ ] **Step 3: Write minimal implementation**

Append to `metrics.py`:

```python
def score_abstention(expects_abstention: bool, abstained: bool) -> dict[str, float]:
    """Emit per-case abstention counters.

    Precision and recall cannot be averaged per case, so they are summarized
    once over the run by ``abstention_summary``.
    """
    return {
        "abstention_tp": float(expects_abstention and abstained),
        "abstention_fp": float(not expects_abstention and abstained),
        "abstention_fn": float(expects_abstention and not abstained),
    }


def abstention_summary(rows: list[dict]) -> dict[str, float | None]:
    """Return corpus-level abstention precision, recall, and F1."""
    tp = sum(row.get("abstention_tp", 0.0) for row in rows)
    fp = sum(row.get("abstention_fp", 0.0) for row in rows)
    fn = sum(row.get("abstention_fn", 0.0) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "abstention_f1": f1,
        "abstention_precision": precision,
        "abstention_recall": recall,
    }


def score_distractor_rejection(
    distractor_answers: tuple[str, ...], answer: str | None
) -> dict[str, float]:
    """Score whether a prediction avoided every known distractor answer.

    Reproducing an outdated or counterfactual answer is a distinct failure from
    being merely wrong, so it is measured separately from exact match.
    """
    normalized = _normalize_text(answer or "")
    if not normalized:
        return {"distractor_rejection": 1.0}
    distractors = {_normalize_text(value) for value in distractor_answers}
    return {"distractor_rejection": float(normalized not in distractors)}


def score_multi_slot_answer(
    answer_slots: tuple[tuple[str, ...], ...], answer: str | None
) -> dict[str, float]:
    """Score how many required answer slots appear in a single response."""
    normalized = _normalize_text(answer or "")
    if not answer_slots:
        return {"all_slots_matched": 0.0, "slot_coverage": 0.0}
    matched = sum(
        any(_normalize_text(alias) in normalized for alias in slot) for slot in answer_slots
    )
    coverage = matched / len(answer_slots)
    return {"all_slots_matched": float(matched == len(answer_slots)), "slot_coverage": coverage}


def score_sub_claims(
    expected: tuple[str, ...], predicted: tuple[str, ...]
) -> dict[str, float | None]:
    """Score per-sub-claim labels, undefined when the decomposition disagrees."""
    if not expected or len(expected) != len(predicted):
        return {"sub_claim_accuracy": None, "sub_claim_macro_f1": None}
    accuracy = sum(a == b for a, b in zip(expected, predicted)) / len(expected)
    per_label_f1 = []
    for label in sorted(set(expected) | set(predicted)):
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        denominator = 2 * tp + fp + fn
        per_label_f1.append(2 * tp / denominator if denominator else 0.0)
    macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    return {"sub_claim_accuracy": accuracy, "sub_claim_macro_f1": macro_f1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_metrics.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/metrics.py benchmarks/tests/test_metrics.py
git commit -m "feat(benchmarks): measure abstention, distractors, slots, and sub-claims"
```

---

### Task 14: Runner scores per declared capability and records derived provenance

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/runner.py`
- Test: `benchmarks/tests/test_runner.py`

**Interfaces:**
- Consumes: `registry.REQUIRED_SUITES`, `registry.OPTIONAL_SUITES`, `profiles.get_profile`, `readers.file_digest`, `adapters.base.LoadResult`, the Task 13 metrics.
- Produces: `load_config(path) -> dict`, `validate_config(config) -> list[str]`, `load_predictions(path) -> dict[str, Prediction]` (unchanged), `run_suite(adapter, config, predictions, run_dir, allow_skips=False) -> dict`.

Config shape per dataset: `path` (required), `version` (required), `split` (required), `expected_digest` (optional), plus free-form `run_parameters` recorded verbatim in the manifest (this is where RGB's declared `noise_rate` and `passage_num` live; they are recorded, never used to build anything).

Manifest gains `content_digest`, `record_count`, `evidence_id_origin`, `profile`, and `capabilities_scored`; `dataset.version` stays but is now accompanied by the digest.

- [ ] **Step 1: Write the failing test**

Replace `benchmarks/tests/test_runner.py` with:

```python
import json

import pytest
import yaml

from llm_wiki_bench.adapters.vitaminc import VitaminCAdapter
from llm_wiki_bench.registry import get_adapter
from llm_wiki_bench.runner import load_config, run_suite, validate_config
from llm_wiki_bench.schema import Prediction

FIXTURES = "benchmarks/fixtures"


def _config(tmp_path, **overrides):
    dataset = {
        "path": f"{FIXTURES}/vitaminc.jsonl",
        "version": "vitaminc-fixture",
        "split": "test",
    }
    dataset.update(overrides)
    return {
        "output_root": str(tmp_path / "results"),
        "top_k": 8,
        "datasets": {"vitaminc": dataset},
    }


def _predictions():
    return {
        "fixture000_1": Prediction(case_id="fixture000_1", label="entailment"),
        "fixture000_2": Prediction(case_id="fixture000_2", label="contradiction"),
        "fixture001_1": Prediction(case_id="fixture001_1", label="neutral"),
    }


def test_validate_config_requires_a_split(tmp_path):
    config = _config(tmp_path)
    del config["datasets"]["vitaminc"]["split"]
    assert "datasets.vitaminc.split is required" in validate_config(config)


def test_validate_config_rejects_a_missing_data_file(tmp_path):
    config = _config(tmp_path, path=str(tmp_path / "absent.jsonl"))
    errors = validate_config(config)
    assert any("does not exist" in error for error in errors)


def test_manifest_records_the_digest_and_record_count(tmp_path):
    run_dir = tmp_path / "run"
    manifest = run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)
    assert manifest["dataset"]["content_digest"].startswith("sha256:")
    assert manifest["dataset"]["record_count"] == 3
    assert manifest["evidence_id_origin"] == "upstream"
    assert manifest["profile"] == "grounded_verification"
    assert manifest["capabilities_scored"] == ["label"]


def test_a_pinned_digest_mismatch_fails_the_run(tmp_path):
    config = _config(tmp_path, expected_digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="content digest mismatch"):
        run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")


def test_a_matching_pinned_digest_passes(tmp_path):
    first = run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), tmp_path / "a")
    config = _config(tmp_path, expected_digest=first["dataset"]["content_digest"])
    manifest = run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "b")
    assert manifest["dataset"]["content_digest"] == first["dataset"]["content_digest"]


def test_declared_run_parameters_are_recorded_verbatim(tmp_path):
    config = _config(tmp_path, run_parameters={"noise_rate": 0.6, "passage_num": 5})
    manifest = run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")
    assert manifest["run_parameters"] == {"noise_rate": 0.6, "passage_num": 5}


def test_only_declared_capabilities_are_scored(tmp_path):
    """grounded_verification declares `label`; retrieval must not appear."""
    run_dir = tmp_path / "run"
    run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "label_accuracy" in metrics["aggregate"]
    assert not [key for key in metrics["aggregate"] if key.startswith("recall@")]
    assert "mrr" not in metrics["aggregate"]


def test_abstention_summary_appears_only_for_profiles_declaring_it(tmp_path):
    run_dir = tmp_path / "run"
    run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "abstention_precision" not in metrics


def test_a_missing_prediction_still_fails_without_allow_skips(tmp_path):
    predictions = _predictions()
    del predictions["fixture001_1"]
    with pytest.raises(ValueError, match="missing prediction for case fixture001_1"):
        run_suite(get_adapter("vitaminc"), _config(tmp_path), predictions, tmp_path / "run")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_runner.py -q`
Expected: FAIL — `validate_config` does not know `split`, and the manifest has no digest.

- [ ] **Step 3: Write minimal implementation**

Replace `validate_config`, `_dataset_config`, `_score_case`, and the manifest construction in `run_suite`:

```python
from pathlib import Path

from .metrics import (
    abstention_summary,
    aggregate,
    label_confusion_matrix,
    score_abstention,
    score_answer,
    score_citations,
    score_distractor_rejection,
    score_label,
    score_multi_slot_answer,
    score_retrieval,
    score_sub_claims,
)
from .profiles import get_profile
from .registry import OPTIONAL_SUITES, REQUIRED_SUITES


def validate_config(config: dict) -> list[str]:
    """Return every structural error, and confirm each source file exists."""
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config must be a mapping"]
    if not _nonblank(config.get("output_root")):
        errors.append("output_root is required")
    top_k = config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        errors.append("top_k must be a positive integer")
    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        return errors + ["datasets must be a mapping"]
    for suite in REQUIRED_SUITES:
        if not isinstance(datasets.get(suite), dict):
            errors.append(f"datasets.{suite} is required")
    for suite in (*REQUIRED_SUITES, *OPTIONAL_SUITES):
        details = datasets.get(suite)
        if details is None:
            continue
        if not isinstance(details, dict):
            errors.append(f"datasets.{suite} must be a mapping")
            continue
        for key in ("path", "version", "split"):
            if not _nonblank(details.get(key)):
                errors.append(f"datasets.{suite}.{key} is required")
        path = details.get("path")
        if _nonblank(path) and not Path(path).is_file():
            errors.append(f"datasets.{suite}.path does not exist: {path}")
        digest = details.get("expected_digest")
        if digest is not None and not _nonblank(digest):
            errors.append(f"datasets.{suite}.expected_digest must be a non-blank string")
        parameters = details.get("run_parameters")
        if parameters is not None and not isinstance(parameters, dict):
            errors.append(f"datasets.{suite}.run_parameters must be a mapping")
    return errors
```

In `run_suite`, replace the load and manifest sections:

```python
    dataset = _dataset_config(adapter, config)
    result = adapter.load(Path(dataset["path"]), split=dataset["split"])
    expected_digest = dataset.get("expected_digest")
    if _nonblank(expected_digest) and expected_digest != result.content_digest:
        raise ValueError(
            f"{adapter.name}: content digest mismatch; configuration pins "
            f"{expected_digest} but {dataset['path']} is {result.content_digest}"
        )
    profile = get_profile(adapter.profile)
    cases = result.cases
```

and build the manifest as:

```python
    manifest = {
        "capabilities_scored": sorted(profile.capabilities),
        "case_counts": counts,
        "dataset": {
            "content_digest": result.content_digest,
            "path": dataset["path"],
            "record_count": result.record_count,
            "version": dataset["version"],
        },
        "evidence_id_origin": adapter.evidence_id_origin,
        "profile": profile.name,
        "run_parameters": dict(dataset.get("run_parameters") or {}),
        "split": dataset["split"],
        "suite": adapter.name,
        "top_k": config["top_k"],
    }
```

Replace `_score_case` so it dispatches on declared capabilities:

```python
def _score_case(case: Any, prediction: Prediction, top_k: int, profile: Any) -> dict[str, float | None]:
    """Score exactly the capabilities the profile declares.

    Never infer applicability from whether a field is populated: that is how a
    partial measurement comes to look like a complete one.
    """
    scores: dict[str, float | None] = {}
    capabilities = profile.capabilities
    if "retrieval" in capabilities:
        scores.update(score_retrieval(case.evidence_ids, prediction.ranked_evidence_ids, ks=(1, 3, top_k)))
    if "fine_retrieval" in capabilities:
        fine = score_retrieval(case.fine_evidence_ids, prediction.ranked_evidence_ids, ks=(1, 3, top_k))
        scores.update({f"fine_{key}": value for key, value in fine.items()})
    if "citations" in capabilities:
        scores.update(score_citations(case.evidence_ids, prediction.cited_evidence_ids))
    if "answer" in capabilities:
        scores.update(score_answer(tuple(case.labels["answers"]), prediction.answer))
    if "multi_slot_answer" in capabilities:
        slots = tuple(tuple(slot) for slot in case.labels["answer_slots"])
        scores.update(score_multi_slot_answer(slots, prediction.answer))
    if "abstention" in capabilities:
        scores.update(score_abstention(case.expects_abstention, prediction.abstained))
    if "label" in capabilities:
        scores.update(score_label(case.labels["label"], prediction.label))
    if "distractor_rejection" in capabilities:
        scores.update(
            score_distractor_rejection(tuple(case.labels["distractor_answers"]), prediction.answer)
        )
    if "sub_claim_labels" in capabilities:
        scores.update(
            score_sub_claims(tuple(case.labels["sub_claim_labels"]), prediction.sub_claim_labels)
        )
    return scores
```

Update the call site to `_score_case(case, prediction, int(config["top_k"]), profile)`, and extend the metrics assembly so corpus-level summaries appear only for declaring profiles:

```python
    metrics: dict[str, Any] = {"aggregate": aggregate(metric_rows)}
    if "abstention" in profile.capabilities:
        metrics.update(abstention_summary(metric_rows))
    if labels:
        metrics["label_confusion_matrix"] = label_confusion_matrix(labels)
```

Also update `_dataset_config` to require `split` alongside `path` and `version`, and delete the module-level `REQUIRED_SUITES` constant now that it comes from the registry.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmarks && pytest tests/test_runner.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/runner.py benchmarks/tests/test_runner.py
git commit -m "feat(benchmarks): score declared capabilities and record derived provenance"
```

---

### Task 15: `validate` loads data and `conformance` reads all of it

**Files:**
- Modify: `benchmarks/src/llm_wiki_bench/__main__.py`, `benchmarks/src/llm_wiki_bench/runner.py`
- Test: `benchmarks/tests/test_cli.py`, `benchmarks/tests/test_conformance.py`

**Interfaces:**
- Consumes: `registry.get_adapter`, `registry.enabled_adapters`, `runner.load_config`, `runner.validate_config`.
- Produces: `runner.check_conformance(adapter, dataset: dict, limit: int | None) -> dict` returning `{"suite", "record_count", "content_digest", "failures": tuple[str, ...], "checked"}`; CLI `validate --config [--sample N]` and `conformance --config [--suite NAME]`.

`validate` normalizes the first `--sample` records (default 5) of every configured suite. `conformance` passes `limit=None`, so it normalizes every record and reports each failure with its record number.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/tests/test_conformance.py
import json

import pytest
import yaml

from llm_wiki_bench.__main__ import main
from llm_wiki_bench.registry import get_adapter
from llm_wiki_bench.runner import check_conformance


def test_conformance_reports_digest_and_no_failures_for_a_good_source():
    dataset = {
        "path": "benchmarks/fixtures/vitaminc.jsonl",
        "version": "fixture",
        "split": "test",
    }
    report = check_conformance(get_adapter("vitaminc"), dataset, limit=None)
    assert report["failures"] == ()
    assert report["record_count"] == 3
    assert report["checked"] == 3
    assert report["content_digest"].startswith("sha256:")


def test_conformance_reports_the_failing_record_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"unique_id": "u1", "claim": "c", "evidence": "e", "label": "SUPPORTS"}) + "\n"
        + json.dumps({"unique_id": "u2", "claim": "c", "evidence": "e", "label": "NOT_ENOUGH_INFO"}) + "\n",
        encoding="utf-8",
    )
    dataset = {"path": str(path), "version": "fixture", "split": "test"}
    report = check_conformance(get_adapter("vitaminc"), dataset, limit=None)
    assert len(report["failures"]) == 1
    assert "record 2" in report["failures"][0]


def test_validate_rejects_a_nonexistent_data_path(tmp_path, capsys):
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            name: {"path": str(tmp_path / "absent"), "version": "v", "split": "s"}
            for name in (
                "longmemeval",
                "hoh",
                "vitaminc",
                "rgb_base",
                "rgb_integration",
                "rgb_counterfactual",
            )
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["validate", "--config", str(path)])
    assert "does not exist" in capsys.readouterr().err


def test_conformance_subcommand_accepts_a_single_suite(tmp_path, capsys):
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            "vitaminc": {
                "path": "benchmarks/fixtures/vitaminc.jsonl",
                "version": "fixture",
                "split": "test",
            }
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["conformance", "--config", str(path), "--suite", "vitaminc"]) == 0
    assert "vitaminc" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmarks && pytest tests/test_conformance.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_conformance'`

- [ ] **Step 3: Write minimal implementation**

Add to `runner.py`:

```python
def check_conformance(adapter: BenchmarkAdapter, dataset: dict, limit: int | None) -> dict:
    """Normalize a source and report every record that fails.

    ``limit=None`` reads the whole file. A digest is always returned, so a
    conformance report names the exact bytes it checked.
    """
    source = Path(dataset["path"])
    if not source.is_file():
        raise ValueError(f"{adapter.name}: source does not exist: {source}")
    reader = READERS[adapter.container]
    split = dataset.get("split") or source.stem
    failures: list[str] = []
    record_count = 0
    checked = 0
    for record_number, record in reader(source):
        record_count += 1
        if limit is not None and checked >= limit:
            continue
        checked += 1
        try:
            values = adapter.normalize(record, source, record_number, split)
            metadata = dict(values.pop("metadata", {}))
            BenchmarkCase(
                dataset=adapter.name,
                split=split,
                profile=adapter.profile,
                metadata=metadata,
                **values,
            )
        except (TypeError, ValueError) as error:
            failures.append(str(error))
    return {
        "checked": checked,
        "content_digest": file_digest(source),
        "failures": tuple(failures),
        "record_count": record_count,
        "suite": adapter.name,
    }
```

with the imports `from .adapters.base import BenchmarkAdapter`, `from .readers import READERS, file_digest`, and `from .schema import BenchmarkCase, Prediction`.

In `__main__.py`, add `--sample` to `validate`, add the `conformance` parser, and route both:

```python
    validate.add_argument("--sample", type=int, default=5)
    conformance = commands.add_parser("conformance")
    conformance.add_argument("--config", type=Path, required=True)
    conformance.add_argument("--suite", default=None)
```

```python
        if args.command == "validate":
            config = load_config(args.config)
            errors = validate_config(config)
            if errors:
                raise ValueError("; ".join(errors))
            for adapter in enabled_adapters(config):
                report = check_conformance(adapter, config["datasets"][adapter.name], args.sample)
                if report["failures"]:
                    raise ValueError(f"{adapter.name}: " + "; ".join(report["failures"]))
                print(f"{adapter.name}: {report['checked']}/{report['record_count']} sampled ok")
            print("valid")
            return 0
        if args.command == "conformance":
            config = load_config(args.config)
            names = [args.suite] if args.suite else [a.name for a in enabled_adapters(config)]
            failed = False
            for name in names:
                adapter = get_adapter(name)
                report = check_conformance(adapter, config["datasets"][name], None)
                status = "ok" if not report["failures"] else f"{len(report['failures'])} failures"
                print(f"{name}: {report['record_count']} records, {report['content_digest']}, {status}")
                for failure in report["failures"]:
                    print(f"  {failure}")
                failed = failed or bool(report["failures"])
            return 1 if failed else 0
```

Import `check_conformance`, `enabled_adapters`, and `get_adapter` in `__main__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd benchmarks && pytest tests/test_conformance.py tests/test_cli.py -q`
Expected: PASS. Update any pre-existing `test_cli.py` case that asserted `validate` prints only `valid`, since it now also prints a per-suite sampled line.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/src/llm_wiki_bench/runner.py benchmarks/src/llm_wiki_bench/__main__.py benchmarks/tests/test_conformance.py benchmarks/tests/test_cli.py
git commit -m "feat(benchmarks): make validate load data and add conformance"
```

---

### Task 16: Configuration example, documentation, and full-suite verification

**Files:**
- Modify: `benchmarks/configs/suite.example.yaml`, `benchmarks/README.md`, `.gitignore`
- Test: full suite from both roots

**Interfaces:**
- Consumes: everything above. Produces no new code interface.

- [ ] **Step 1: Rewrite the example configuration**

```bash
cat > benchmarks/configs/suite.example.yaml <<'YAML'
output_root: benchmarks/results
top_k: 8

datasets:
  longmemeval:
    path: benchmarks/data/longmemeval_oracle.json
    version: longmemeval-cleaned-oracle
    split: oracle

  hoh:
    path: benchmarks/data/hoh_qas_240601_241201.parquet
    version: hoh-qas-240601-241201
    split: 240601_241201

  vitaminc:
    path: benchmarks/data/vitaminc_test.jsonl
    version: vitaminc-test
    split: test

  rgb_base:
    path: benchmarks/data/en_refine.json
    version: rgb-en-refine
    split: en_refine
    run_parameters:
      noise_rate: 0.6
      passage_num: 5

  rgb_integration:
    path: benchmarks/data/en_int.json
    version: rgb-en-int
    split: en_int

  rgb_counterfactual:
    path: benchmarks/data/en_fact.json
    version: rgb-en-fact
    split: en_fact

  factlens:
    path: benchmarks/data/fact_lens_benchmark.csv
    version: factlens-benchmark
    split: benchmark
YAML
```

- [ ] **Step 2: Rewrite the README dataset section**

Replace the dataset/setup portion of `benchmarks/README.md` with a table carrying, per suite: download location, license, container, split concept, profile, and whether evidence identifiers are upstream or synthesized. Use the verified values:

| Suite | Source | License | Container | Evidence ids |
| --- | --- | --- | --- | --- |
| `longmemeval` | HF `xiaowu0162/longmemeval-cleaned` | MIT | JSON array | upstream |
| `hoh` | HF `russwest404/HoH-QAs` | Apache-2.0 | Parquet (needs `benchmarks[hoh]`) | synthesized |
| `vitaminc` | HF `tals/vitaminc` (`test.jsonl`) | CC-BY-SA-3.0 | JSONL | upstream |
| `rgb_base` | GitHub `chen700564/RGB` (`en_refine.json`) | none declared | JSONL | synthesized |
| `rgb_integration` | GitHub `chen700564/RGB` (`en_int.json`) | none declared | JSONL | synthesized |
| `rgb_counterfactual` | GitHub `chen700564/RGB` (`en_fact.json`) | none declared | JSONL | synthesized |
| `factlens` | GitHub `megagonlabs/factlens` | BSD-3-Clause | CSV | upstream |

State explicitly, in prose: RGB declares no license, so acquisition is the user's decision and no RGB data is redistributed here; synthesized evidence identifiers mean a retrieval score for those suites is computed over adapter-assigned ids; and RGB's `noise_rate`/`passage_num` are recorded from configuration but not used to assemble anything. Document the commands:

```bash
python -m llm_wiki_bench validate --config benchmarks/configs/suite.local.yaml
python -m llm_wiki_bench conformance --config benchmarks/configs/suite.local.yaml --suite rgb_base
python -m llm_wiki_bench run --config benchmarks/configs/suite.local.yaml --suite rgb_base --predictions preds.jsonl
python -m llm_wiki_bench report --run-dir benchmarks/results/<timestamp>-rgb_base
```

- [ ] **Step 3: Confirm data and results stay ignored**

Run: `git check-ignore -v benchmarks/data/en_refine.json benchmarks/results/x/manifest.json`
Expected: both report a matching `.gitignore` rule. If either is untracked-but-not-ignored, add the rule.

- [ ] **Step 4: Run the whole suite from both roots**

Run: `cd benchmarks && pytest -q`
Expected: PASS, no skips (the `dev` extra installs `pyarrow`).

Run: `cd .. && pytest -q`
Expected: PASS with the main suite included, and no collection errors.

Run: `python -m llm_wiki_bench validate --config benchmarks/configs/suite.example.yaml`
Expected: FAIL naming each `benchmarks/data/...` path as nonexistent — this is the fix for the original defect working; the example config points at data the repository does not carry.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/configs/suite.example.yaml benchmarks/README.md .gitignore
git commit -m "docs(benchmarks): document released sources, licenses, and commands"
```

---

## Verification before opening the PR

- [ ] `cd benchmarks && pytest -q` — all pass, zero skips.
- [ ] `cd <repo root> && pytest -q` — main suite plus benchmark tests, zero collection errors.
- [ ] `grep -rn "source_version" benchmarks/src benchmarks/fixtures` — no matches; the per-record contract is gone.
- [ ] `grep -rn "NOT_ENOUGH_INFO" benchmarks/src` — no matches.
- [ ] `python -X importtime -m pytest benchmarks/tests -q 2> imports.log && grep -cE 'torch|sentence_transformers' imports.log` — reports `0`.
- [ ] Every adapter has been run through `conformance` against its real downloaded source, or is reported in the PR body as **unverified against real data**. Do not describe an adapter as working on the strength of fixtures alone — that is precisely the claim that failed review the first time.

## Self-Review Notes

Spec coverage checked section by section: profiles (Task 1), readers (Task 2), provenance out of records (Tasks 3–4), digest and manifest (Task 14), synthesized-id labeling (Tasks 7–10, 14), LongMemEval abstention and two-level evidence (Tasks 5, 13, 14), HoH id synthesis and temporal profile (Task 10), RGB three suites and pool emission (Tasks 7–9), VitaminC label spelling (Task 6), FactLens repr-list parsing (Task 11), `validate`/`conformance` split (Task 15), fixtures at released shape (Tasks 5–11), documentation and licenses (Task 16).
