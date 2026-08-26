"""Small, permissive configuration loader for a V3 Markdown vault."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


CONFIG_FILENAME = "wiki.toml"


@dataclass(frozen=True)
class Config:
    config_dir: Path
    root: Path
    artifact_dir: Path
    correction_dir: Path
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_device: str = "auto"
    embedding_batch_size: int = 32
    chunk_boundary_keep_threshold: float = 0.66
    chunk_candidate_budget: float = 0.50
    knn_k: int = 3
    candidate_pool: int = 50
    rrf_k: int = 60
    text_weight: float = 1.0
    dense_weight: float = 2.0
    tree_weight: float = 0.8
    knn_weight: float = 0.8
    auto_answer_cosine: float = 0.55
    auto_none_cosine: float = 0.30
    auto_margin: float = 0.04
    review_similarity: float = 0.72
    review_query_limit: int = 12


def find_config_dir(start: Path | None = None) -> Path:
    explicit = os.environ.get("WIKI_VAULT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    return current


def _positive_int(value: object, default: int, name: str) -> int:
    resolved = default if value is None else value
    if not isinstance(resolved, int) or resolved <= 0:
        raise ValueError(f"[v3] {name} must be a positive integer")
    return resolved


def _number(value: object, default: float, name: str) -> float:
    resolved = default if value is None else value
    if not isinstance(resolved, (int, float)):
        raise ValueError(f"[v3] {name} must be a number")
    return float(resolved)


def _fraction(value: object, default: float, name: str, *, allow_zero: bool) -> float:
    resolved = _number(value, default, name)
    lower_ok = resolved >= 0.0 if allow_zero else resolved > 0.0
    if not lower_ok or resolved > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"[v3] {name} must be in {interval}")
    return resolved


def load(start: Path | None = None) -> Config:
    config_dir = find_config_dir(start)
    path = config_dir / CONFIG_FILENAME
    raw = tomllib.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    vault = raw.get("vault", {})
    v3 = raw.get("v3", {})
    if not isinstance(vault, dict) or not isinstance(v3, dict):
        raise ValueError("[vault] and [v3] must be TOML tables")

    root_value = vault.get("root", ".")
    if not isinstance(root_value, str):
        raise ValueError("[vault] root must be a string")
    root = (config_dir / root_value).resolve()
    artifact_name = v3.get("artifact_dir", ".llm_wiki_v3")
    correction_name = v3.get("correction_dir", "_wiki_corrections")
    if not isinstance(artifact_name, str) or not isinstance(correction_name, str):
        raise ValueError("[v3] artifact_dir and correction_dir must be strings")

    return Config(
        config_dir=config_dir,
        root=root,
        artifact_dir=(root / artifact_name).resolve(),
        correction_dir=(root / correction_name).resolve(),
        model_id=str(v3.get("model_id", "Qwen/Qwen3-Embedding-0.6B")),
        embed_device=str(v3.get("embed_device", "auto")),
        embedding_batch_size=_positive_int(v3.get("embedding_batch_size"), 32, "embedding_batch_size"),
        chunk_boundary_keep_threshold=_fraction(
            v3.get("chunk_boundary_keep_threshold"),
            0.66,
            "chunk_boundary_keep_threshold",
            allow_zero=True,
        ),
        chunk_candidate_budget=_fraction(
            v3.get("chunk_candidate_budget"),
            0.50,
            "chunk_candidate_budget",
            allow_zero=False,
        ),
        knn_k=_positive_int(v3.get("knn_k"), 3, "knn_k"),
        candidate_pool=_positive_int(v3.get("candidate_pool"), 50, "candidate_pool"),
        rrf_k=_positive_int(v3.get("rrf_k"), 60, "rrf_k"),
        text_weight=_number(v3.get("text_weight"), 1.0, "text_weight"),
        dense_weight=_number(v3.get("dense_weight"), 2.0, "dense_weight"),
        tree_weight=_number(v3.get("tree_weight"), 0.8, "tree_weight"),
        knn_weight=_number(v3.get("knn_weight"), 0.8, "knn_weight"),
        auto_answer_cosine=_number(v3.get("auto_answer_cosine"), 0.55, "auto_answer_cosine"),
        auto_none_cosine=_number(v3.get("auto_none_cosine"), 0.30, "auto_none_cosine"),
        auto_margin=_number(v3.get("auto_margin"), 0.04, "auto_margin"),
        review_similarity=_number(v3.get("review_similarity"), 0.72, "review_similarity"),
        review_query_limit=_positive_int(v3.get("review_query_limit"), 12, "review_query_limit"),
    )
