"""Filesystem helpers for `.llm_wiki_v2` artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from llm_wiki.paths import content_root
from llm_wiki.v2.schemas import (ARTIFACT_SCHEMA_VERSION, CHUNK_SCHEMA_VERSION,
                                  CONCEPT_EXTRACTION_JSON_SCHEMA, CONCEPT_PROMPT_VERSION,
                                  CONCEPT_SCHEMA_VERSION, PLACEMENT_JSON_SCHEMA,
                                  PLACEMENT_PROMPT_VERSION, RELATION_JSON_SCHEMA,
                                  RELATION_PROMPT_VERSION, TEMPORAL_PROMPT_VERSION)

T = TypeVar("T")


def artifact_root(vault: Path | None = None) -> Path:
    return content_root(vault) / ".llm_wiki_v2"


def artifact_path(name: str, vault: Path | None = None) -> Path:
    return artifact_root(vault) / name


def ensure_layout(vault: Path | None = None) -> Path:
    root = artifact_root(vault)
    for rel in ("concept_embeddings", "net"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    path = root / "schemas.json"
    if not path.exists():
        write_schema_manifest(vault)
    return root


def current_schema_manifest() -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "concept_extraction": CONCEPT_EXTRACTION_JSON_SCHEMA,
        "placement": PLACEMENT_JSON_SCHEMA,
        "relation": RELATION_JSON_SCHEMA,
        "concept_schema_version": CONCEPT_SCHEMA_VERSION,
        "prompt_versions": {
            "concept_extraction": CONCEPT_PROMPT_VERSION,
            "placement": PLACEMENT_PROMPT_VERSION,
            "relation": RELATION_PROMPT_VERSION,
            "temporal": TEMPORAL_PROMPT_VERSION,
        },
    }


def read_schema_manifest(vault: Path | None = None) -> dict | None:
    path = artifact_root(vault) / "schemas.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def schema_staleness(vault: Path | None = None) -> str | None:
    saved = read_schema_manifest(vault)
    if saved is None:
        return "v2 schema manifest missing or invalid"
    current = current_schema_manifest()
    if saved != current:
        return "v2 artifact/prompt schema stale; run a full wiki-concepts build and wiki-net build"
    return None


def write_schema_manifest(vault: Path | None = None) -> Path:
    root = artifact_root(vault)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "schemas.json"
    path.write_text(json.dumps(current_schema_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_jsonl(path: Path, factory) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(factory(json.loads(line)))
    return rows


def write_jsonl(path: Path, rows: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    for row in rows:
        data = row.to_dict() if hasattr(row, "to_dict") else row
        text += json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = row.to_dict() if hasattr(row, "to_dict") else row
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
