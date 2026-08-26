"""Offline benchmark configuration, recorded-prediction execution, and artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .adapters import BenchmarkAdapter, build_case
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
from .readers import READERS, file_digest
from .registry import OPTIONAL_SUITES, REQUIRED_SUITES
from .reports import write_report
from .schema import Prediction


def load_config(path: Path) -> dict:
    """Read a suite configuration without contacting models or dataset services."""
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"unable to load config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


_TOP_K_ERROR = "top_k must be a positive integer"


def _invalid_top_k(config: Mapping[str, Any]) -> bool:
    top_k = config.get("top_k")
    return isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1


def dataset_entry_errors(suite: str, details: Any) -> list[str]:
    """Return every structural error in one ``datasets.<suite>`` entry.

    Shared by ``validate_config`` (which checks every configured suite) and
    the ``conformance`` CLI dispatch (which must validate the handful of
    suites it is about to read before reading any of them, even when
    ``--suite`` narrows scope to less than the whole config).
    """
    if not isinstance(details, dict):
        return [f"datasets.{suite} must be a mapping"]
    errors: list[str] = []
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


def validate_config(config: dict) -> list[str]:
    """Return every structural error, and confirm each source file exists."""
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config must be a mapping"]
    if not _nonblank(config.get("output_root")):
        errors.append("output_root is required")
    if _invalid_top_k(config):
        errors.append(_TOP_K_ERROR)
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
        errors.extend(dataset_entry_errors(suite, details))
    return errors


def load_predictions(path: Path) -> dict[str, Prediction]:
    """Load one JSON object per line, rejecting duplicate case identifiers."""
    predictions: dict[str, Prediction] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to load predictions {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: prediction {line_number}: invalid JSON") from error
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: prediction {line_number}: expected a JSON object")
        try:
            prediction = Prediction(**raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: prediction {line_number}: {error}") from error
        if prediction.case_id in predictions:
            raise ValueError(f"{path}: duplicate prediction case_id {prediction.case_id!r}")
        predictions[prediction.case_id] = prediction
    return predictions


_MAX_REPORTED_FAILURES = 20


def check_conformance(
    adapter: BenchmarkAdapter,
    dataset: dict,
    limit: int | None,
    max_reported_failures: int = _MAX_REPORTED_FAILURES,
) -> dict:
    """Normalize a source and report every record that fails.

    ``limit=None`` reads the whole file. A digest is always returned, so a
    conformance report names the exact bytes it checked.

    A source-wide problem (a bad split, a systemic upstream field) fails
    every record with the same message, so ``failures`` reports only the
    first ``max_reported_failures`` (default 20 -- enough to see whether the
    failures vary by record or repeat the same root cause, small enough that
    the digest line and totals stay on screen even when a whole 100k+ record
    release fails). ``failure_count`` is never capped, so the caller always
    knows the true total.
    """
    source = Path(dataset["path"])
    if not source.is_file():
        raise ValueError(f"{adapter.name}: source does not exist: {source}")
    reader = READERS[adapter.container]
    split = dataset.get("split") or source.stem
    failures: list[str] = []
    failure_count = 0
    record_count = 0
    checked = 0
    for record_number, record in reader(source):
        record_count += 1
        if limit is not None and checked >= limit:
            continue
        checked += 1
        try:
            build_case(adapter, record, source, record_number, split)
        except (TypeError, ValueError) as error:
            failure_count += 1
            if len(failures) < max_reported_failures:
                failures.append(f"record {record_number}: {error}")
    return {
        "checked": checked,
        "content_digest": file_digest(source),
        "failure_count": failure_count,
        "failures": tuple(failures),
        "record_count": record_count,
        "suite": adapter.name,
    }


def run_suite(
    adapter: BenchmarkAdapter,
    config: dict,
    predictions: dict[str, Prediction],
    run_dir: Path,
    allow_skips: bool = False,
) -> dict:
    """Score recorded predictions for one normalized suite and persist artifacts.

    This validates only what ``run_suite`` itself depends on: the one dataset
    entry this adapter needs, and ``top_k``. It does not enforce the whole
    config's required-suite roster; a caller that must reject an incomplete
    multi-suite config up front should call ``validate_config`` itself, as the
    CLI does before dispatching to ``run_suite``.
    """
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError("config must be a mapping with a datasets mapping")
    if _invalid_top_k(config):
        raise ValueError(_TOP_K_ERROR)
    dataset = _dataset_config(adapter, config)
    result = adapter.load(Path(dataset["path"]), split=dataset["split"])
    expected_digest = dataset.get("expected_digest")
    if _nonblank(expected_digest) and expected_digest != result.content_digest:
        raise ValueError(
            f"{adapter.name}: content digest mismatch; configuration pins "
            f"{expected_digest} but {dataset['path']} is {result.content_digest}"
        )
    cases = result.cases
    per_case: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float | None]] = []
    labels: list[tuple[str | None, str | None]] = []
    observed_profiles: Counter[str] = Counter()

    for case in cases:
        prediction = predictions.get(case.id)
        if prediction is None:
            if not allow_skips:
                raise ValueError(f"missing prediction for case {case.id}")
            per_case.append({"case_id": case.id, "dataset": case.dataset, "reason": "missing_prediction", "status": "skipped"})
            continue
        profile = get_profile(case.profile)
        try:
            scores = _score_case(case, prediction, int(config["top_k"]), profile)
        except (TypeError, ValueError) as error:
            per_case.append({"case_id": case.id, "dataset": case.dataset, "reason": str(error), "status": "error"})
            continue
        row = {
            "case_id": case.id,
            "dataset": case.dataset,
            "prediction": _prediction_dict(prediction),
            "scores": scores,
            "status": "ok",
        }
        per_case.append(row)
        metric_rows.append(scores)
        observed_profiles[profile.name] += 1
        if "label" in profile.capabilities:
            labels.append((case.labels["label"], prediction.label))

    counts = {
        "errors": sum(row["status"] == "error" for row in per_case),
        "evaluated": sum(row["status"] == "ok" for row in per_case),
        "skipped": sum(row["status"] == "skipped" for row in per_case),
        "total": len(cases),
    }
    observed = [get_profile(name) for name in observed_profiles]
    capabilities_scored = sorted({capability for profile in observed for capability in profile.capabilities})
    manifest = {
        "capabilities_scored": capabilities_scored,
        "case_counts": counts,
        "dataset": {
            "content_digest": result.content_digest,
            "path": dataset["path"],
            "record_count": result.record_count,
            "version": dataset["version"],
        },
        "evidence_id_origin": adapter.evidence_id_origin,
        "profiles": dict(sorted(observed_profiles.items())),
        "run_parameters": dict(dataset.get("run_parameters") or {}),
        "split": dataset["split"],
        "suite": adapter.name,
        "top_k": config["top_k"],
    }
    metrics: dict[str, Any] = {"aggregate": aggregate(metric_rows)}
    if any("abstention" in profile.capabilities for profile in observed):
        metrics.update(abstention_summary(metric_rows))
    if labels:
        metrics["label_confusion_matrix"] = label_confusion_matrix(labels)
    _write_report(Path(run_dir), manifest, per_case, metrics)
    _write_jsonl(Path(run_dir) / "skips.jsonl", [row for row in per_case if row["status"] != "ok"])
    return manifest


def timestamped_run_dir(output_root: Path, suite: str) -> Path:
    """Reserve a unique UTC timestamped output directory for a suite execution."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        name = f"{stamp}-{suite}" if suffix == 1 else f"{stamp}-{suite}-{suffix - 1}"
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            pass
        suffix += 1


def _dataset_config(adapter: BenchmarkAdapter, config: Mapping[str, Any]) -> dict[str, Any]:
    details = config["datasets"].get(adapter.name)
    if not isinstance(details, dict) or not _nonblank(details.get("path")):
        state = "not_configured" if not adapter.required else "missing dataset path"
        raise ValueError(f"{adapter.name}: {state}")
    if not _nonblank(details.get("version")):
        raise ValueError(f"{adapter.name}: missing dataset version")
    if not _nonblank(details.get("split")):
        raise ValueError(f"{adapter.name}: missing dataset split")
    if not Path(details["path"]).is_file():
        raise ValueError(f"{adapter.name}: dataset path does not exist: {details['path']}")
    return details


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


def _prediction_dict(prediction: Prediction) -> dict[str, Any]:
    return {
        "abstained": prediction.abstained,
        "answer": prediction.answer,
        "cited_evidence_ids": prediction.cited_evidence_ids,
        "label": prediction.label,
        "latency_ms": prediction.latency_ms,
        "ranked_evidence_ids": prediction.ranked_evidence_ids,
    }


def _write_report(run_dir: Path, manifest: dict, per_case: list[dict], metrics: dict) -> None:
    """Write artifacts, naming what was unserializable rather than failing inside json.dumps.

    Every one of the three JSON payloads can carry a value nobody coerced:
    ``manifest`` includes config-supplied ``run_parameters`` verbatim (a
    ``NaN`` such as ``run_parameters: {noise_rate: .nan}`` survives
    ``yaml.safe_load`` as a Python float), adapters copy raw upstream values
    into ``metadata['source_fields']`` (and transitively into ``per_case``
    rows) with no type coercion, and ``metrics`` is runner-computed but
    checked on the same footing for defense in depth. ``reports._json`` uses
    ``allow_nan=False`` and would otherwise crash with a bare, unattributed
    error from deep inside the standard library. Check each payload up front
    so a bad value fails loudly and names its source.
    """
    _require_serializable(manifest, "manifest (check configuration values such as run_parameters)")
    for row in per_case:
        _require_serializable(row, f"case {row.get('case_id')!r} result")
    _require_serializable(metrics, "metrics")
    write_report(run_dir, manifest, per_case, metrics)


def _require_serializable(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not JSON-serializable: {error}") from error


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
