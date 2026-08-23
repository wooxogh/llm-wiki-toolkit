"""Offline benchmark configuration, recorded-prediction execution, and artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .adapters import BenchmarkAdapter
from .metrics import aggregate, label_confusion_matrix, score_answer, score_citations, score_label, score_retrieval
from .reports import write_report
from .schema import Prediction


REQUIRED_SUITES = ("longmemeval", "hoh", "vitaminc", "rgb")


def load_config(path: Path) -> dict:
    """Read a suite configuration without contacting models or dataset services."""
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"unable to load config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


def validate_config(config: dict) -> list[str]:
    """Return all structural configuration errors; this never loads a dataset."""
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
        details = datasets.get(suite)
        if not isinstance(details, dict):
            errors.append(f"datasets.{suite} is required")
            continue
        if not _nonblank(details.get("path")):
            errors.append(f"datasets.{suite}.path is required")
        if not _nonblank(details.get("version")):
            errors.append(f"datasets.{suite}.version is required")
    for suite, details in datasets.items():
        if suite not in (*REQUIRED_SUITES, "factlens") or not isinstance(details, dict):
            continue
        if "path" in details and not _nonblank(details["path"]):
            errors.append(f"datasets.{suite}.path must be a non-blank string")
        if "version" in details and not _nonblank(details["version"]):
            errors.append(f"datasets.{suite}.version must be a non-blank string")
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


def run_suite(
    adapter: BenchmarkAdapter,
    config: dict,
    predictions: dict[str, Prediction],
    run_dir: Path,
    allow_skips: bool = False,
) -> dict:
    """Score recorded predictions for one normalized suite and persist artifacts."""
    errors = validate_config(config)
    if errors:
        raise ValueError("invalid config: " + "; ".join(errors))
    dataset = _dataset_config(adapter, config)
    split = dataset.get("split")
    cases = adapter.load(Path(dataset["path"]), split=split)
    per_case: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float | None]] = []
    labels: list[tuple[str | None, str | None]] = []

    for case in cases:
        prediction = predictions.get(case.id)
        if prediction is None:
            if not allow_skips:
                raise ValueError(f"missing prediction for case {case.id}")
            per_case.append({"case_id": case.id, "dataset": case.dataset, "reason": "missing_prediction", "status": "skipped"})
            continue
        try:
            scores = _score_case(case, prediction, int(config["top_k"]))
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
        if "label" in case.labels:
            labels.append((case.labels["label"], prediction.label))

    counts = {
        "errors": sum(row["status"] == "error" for row in per_case),
        "evaluated": sum(row["status"] == "ok" for row in per_case),
        "skipped": sum(row["status"] == "skipped" for row in per_case),
        "total": len(cases),
    }
    manifest = {
        "case_counts": counts,
        "dataset": {"path": dataset["path"], "version": dataset["version"]},
        "split": split,
        "suite": adapter.name,
        "top_k": config["top_k"],
    }
    metrics: dict[str, Any] = {"aggregate": aggregate(metric_rows)}
    if labels:
        metrics["label_confusion_matrix"] = label_confusion_matrix(labels)
    write_report(Path(run_dir), manifest, per_case, metrics)
    _write_jsonl(Path(run_dir) / "skips.jsonl", [row for row in per_case if row["status"] != "ok"])
    return manifest


def timestamped_run_dir(output_root: Path, suite: str) -> Path:
    """Reserve a unique UTC timestamped output directory for a suite execution."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = Path(output_root) / f"{stamp}-{suite}"
    suffix = 1
    while candidate.exists():
        candidate = Path(output_root) / f"{stamp}-{suite}-{suffix}"
        suffix += 1
    return candidate


def _dataset_config(adapter: BenchmarkAdapter, config: Mapping[str, Any]) -> dict[str, Any]:
    details = config["datasets"].get(adapter.name)
    if not isinstance(details, dict) or not _nonblank(details.get("path")):
        state = "not_configured" if not adapter.required else "missing dataset path"
        raise ValueError(f"{adapter.name}: {state}")
    if not _nonblank(details.get("version")):
        raise ValueError(f"{adapter.name}: missing dataset version")
    return details


def _score_case(case: Any, prediction: Prediction, top_k: int) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    if case.evidence_ids:
        scores.update(score_retrieval(case.evidence_ids, prediction.ranked_evidence_ids, ks=(1, 3, top_k)))
        scores.update(score_citations(case.evidence_ids, prediction.cited_evidence_ids))
    answers = case.labels.get("answers")
    if answers is not None:
        scores.update(score_answer(tuple(answers), prediction.answer))
    scores.update(score_label(case.labels.get("label"), prediction.label))
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
