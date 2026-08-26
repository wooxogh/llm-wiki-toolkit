"""Deterministic report serialization for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(run_dir: Path, manifest: dict, per_case: list[dict], metrics: dict) -> None:
    """Write the benchmark artifact set using stable UTF-8 JSON and Markdown."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", manifest)
    _write_jsonl(run_dir / "per_case.jsonl", per_case)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_markdown(manifest, metrics), encoding="utf-8", newline="\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_json(value) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    contents = "".join(_json(row) + "\n" for row in rows)
    path.write_text(contents, encoding="utf-8", newline="\n")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _markdown(manifest: dict, metrics: dict) -> str:
    """Render a manifest header above the metrics.

    A `metrics.json` with `--allow-skips` where most cases had no prediction
    would otherwise look identical to a complete run: only the manifest
    carries `case_counts`, the digest, the split, and which capabilities were
    actually scored. Rendering them here makes `report.md` -- the
    human-facing artifact -- self-describing rather than requiring a reader
    to cross-reference `manifest.json` separately.
    """
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    header = [
        ("suite", manifest.get("suite")),
        ("split", manifest.get("split")),
        ("version", dataset.get("version")),
        ("content_digest", dataset.get("content_digest")),
        ("case_counts", manifest.get("case_counts")),
        ("capabilities_scored", manifest.get("capabilities_scored")),
    ]
    lines = ["# Benchmark report", "", "## Run", ""]
    lines.extend(f"- {key}: {_format(value)}" for key, value in header)
    lines += ["", "## Metrics", ""]
    lines.extend(f"- {key}: {_format(value)}" for key, value in sorted(metrics.items()))
    return "\n".join(lines) + "\n"


def _format(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    return str(value)
