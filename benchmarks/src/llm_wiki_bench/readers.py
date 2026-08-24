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
