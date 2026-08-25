"""Build Chunker-V3, dense, sparse, Tree, and k-NN artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import yaml

from .config import Config, load
from .hygiene import apply_events, read_events
from .io import configure_stdio, read_json, read_jsonl, write_json, write_jsonl
from .pathing import relative_to_root
from .text import embedding_text, fielded_text, tokenize


SCHEMA_VERSION = 1
IGNORED_DIRECTORIES = {".git", ".llm_wiki_v3", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


class Embedder(Protocol):
    model_id: str
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _relative(path: Path, root: Path) -> str:
    return relative_to_root(path, root).as_posix()


def collect_markdown(config: Config) -> list[Path]:
    if not config.root.is_dir():
        raise FileNotFoundError(f"vault does not exist: {config.root}")
    paths = []
    for path in config.root.rglob("*.md"):
        relative_parts = relative_to_root(path, config.root).parts
        if any(part in IGNORED_DIRECTORIES for part in relative_parts):
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: _relative(path, config.root))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    value = yaml.safe_load(text[3:end]) or {}
    return value if isinstance(value, dict) else {}


def _chunk_document(path: Path, config: Config, chunker) -> list[dict[str, Any]]:
    relative = _relative(path, config.root)
    document_id = relative[:-3] if relative.lower().endswith(".md") else relative
    frontmatter = _frontmatter(path)
    result = chunker.chunk_file(path, document_id=document_id)
    rows = []
    for chunk in result.chunks:
        row = chunk.to_dict()
        row["source_path"] = relative
        row["document_kind"] = frontmatter.get("llm_wiki_v3_kind", "source")
        row["correction_event_ids"] = [frontmatter["correction_event_id"]] if frontmatter.get("correction_event_id") else []
        row["corrects"] = [str(value) for value in frontmatter.get("corrects_chunk_ids") or []]
        rows.append(row)
    return rows


def _tree(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[tuple[str, ...], dict[str, Any]] = {}
    for chunk in chunks:
        source = str(chunk["source_path"])
        directory = tuple(Path(source).parent.parts) if Path(source).parent != Path(".") else ()
        paths = [
            ("root",),
            ("root", "directory", *directory),
            ("root", "directory", *directory, "document", str(chunk["document_id"])),
        ]
        heading_prefix = paths[-1]
        for heading in chunk.get("heading_path") or []:
            heading_prefix = (*heading_prefix, "heading", str(heading))
            paths.append(heading_prefix)
        for key in paths:
            node = nodes.setdefault(
                key,
                {
                    "node_id": "tree:" + hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:16],
                    "kind": "root" if key == ("root",) else ("heading" if "heading" in key else ("document" if "document" in key else "directory")),
                    "label": key[-1],
                    "path": list(key),
                    "chunk_ids": [],
                },
            )
            node["chunk_ids"].append(chunk["id"])
    return {"schema_version": SCHEMA_VERSION, "nodes": sorted(nodes.values(), key=lambda row: row["path"])}


def _knn(vectors: np.ndarray, chunks: list[dict[str, Any]], k: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = len(vectors)
    if count == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.float32), {"groups": []}
    if count == 1:
        return np.empty((1, 0), dtype=np.int64), np.empty((1, 0), dtype=np.float32), {
            "groups": [{"group_id": "group-1", "chunk_ids": [chunks[0]["id"]], "representative_chunk_ids": [chunks[0]["id"]]}]
        }
    usable_k = min(max(1, k), count - 1)
    similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)
    np.fill_diagonal(similarities, -np.inf)
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :usable_k]
    scores = np.take_along_axis(similarities, indices, axis=1)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, neighbors in enumerate(indices):
        for right in neighbors:
            adjacency[left].add(int(right))
            adjacency[int(right)].add(left)
    remaining = set(range(count))
    groups = []
    while remaining:
        first = min(remaining)
        queue = deque([first])
        component = []
        remaining.remove(first)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        centroid = vectors[component].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        representatives = sorted(component, key=lambda index: (-float(vectors[index] @ centroid), chunks[index]["id"]))[:5]
        groups.append(
            {
                "group_id": f"group-{len(groups) + 1}",
                "chunk_ids": [chunks[index]["id"] for index in component],
                "representative_chunk_ids": [chunks[index]["id"] for index in representatives],
            }
        )
    return indices, scores, {"schema_version": SCHEMA_VERSION, "k": usable_k, "groups": groups}


def _default_runtime(config: Config):
    from .chunking import ChunkerConfig, HybridMarkdownChunker
    from .embedder import QwenEmbedder

    embedder = QwenEmbedder(config.model_id, device=config.embed_device, batch_size=config.embedding_batch_size)
    chunker_config = ChunkerConfig(
        model_id=config.model_id,
        device=config.embed_device,
        embedding_batch_size=config.embedding_batch_size,
        boundary_keep_threshold=config.chunk_boundary_keep_threshold,
        candidate_budget=config.chunk_candidate_budget,
        cache_directory=config.artifact_dir / "sentence_embedding_cache",
    )
    return embedder, HybridMarkdownChunker(chunker_config, embedder=embedder._delegate)


def build(config: Config, *, full: bool = False, embedder: Embedder | None = None, chunker=None) -> dict[str, Any]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    if embedder is None or chunker is None:
        default_embedder, default_chunker = _default_runtime(config)
        embedder = embedder or default_embedder
        chunker = chunker or default_chunker

    paths = collect_markdown(config)
    hashes = {_relative(path, config.root): _hash(path) for path in paths}
    manifest_path = config.artifact_dir / "manifest.json"
    prior_manifest = read_json(manifest_path, {}) or {}
    prior_chunks = read_jsonl(config.artifact_dir / "chunks.jsonl")
    vectors_path = config.artifact_dir / "vectors.npy"
    compatible = (
        not full
        and vectors_path.is_file()
        and prior_manifest.get("schema_version") == SCHEMA_VERSION
        and prior_manifest.get("model_id") == config.model_id
        and len(prior_chunks) == int(prior_manifest.get("chunk_count", -1))
    )
    prior_vectors = np.load(vectors_path, allow_pickle=False) if compatible else np.empty((0, 0), dtype=np.float32)
    if compatible and len(prior_vectors) != len(prior_chunks):
        compatible = False

    prior_hashes = prior_manifest.get("documents", {}) if compatible else {}
    changed = {relative for relative, digest in hashes.items() if prior_hashes.get(relative) != digest}
    deleted = set(prior_hashes) - set(hashes)
    retained_rows = []
    retained_vectors = []
    if compatible:
        for index, row in enumerate(prior_chunks):
            if row.get("source_path") not in changed and row.get("source_path") not in deleted:
                retained_rows.append(row)
                retained_vectors.append(prior_vectors[index])

    by_relative = {_relative(path, config.root): path for path in paths}
    new_rows = []
    for relative in sorted(changed if compatible else hashes):
        new_rows.extend(_chunk_document(by_relative[relative], config, chunker))
    new_vectors = np.asarray(embedder.encode([embedding_text(row) for row in new_rows]), dtype=np.float32) if new_rows else None

    base_rows = retained_rows + new_rows
    vector_parts = []
    if retained_vectors:
        vector_parts.append(np.asarray(retained_vectors, dtype=np.float32))
    if new_vectors is not None and len(new_vectors):
        vector_parts.append(new_vectors)
    if vector_parts:
        vectors = np.vstack(vector_parts).astype(np.float32, copy=False)
    else:
        vectors = np.empty((0, 0), dtype=np.float32)

    order = sorted(range(len(base_rows)), key=lambda index: (base_rows[index]["source_path"], base_rows[index]["ordinal"], base_rows[index]["id"]))
    base_rows = [base_rows[index] for index in order]
    vectors = vectors[order] if len(order) else vectors
    rows = apply_events(base_rows, read_events(config))
    write_jsonl(config.artifact_dir / "chunks.jsonl", rows)
    np.save(vectors_path, vectors, allow_pickle=False)
    write_json(config.artifact_dir / "tree.json", _tree(rows))
    knn_indices, knn_scores, groups = _knn(vectors, rows, config.knn_k)
    np.savez(config.artifact_dir / "knn_graph.npz", indices=knn_indices, scores=knn_scores)
    write_json(config.artifact_dir / "knn_groups.json", groups)
    write_json(
        config.artifact_dir / "sparse_index" / "documents.json",
        [{"chunk_id": row["id"], "tokens": tokenize(fielded_text(row))} for row in rows],
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_id": config.model_id,
        "documents": hashes,
        "document_count": len(hashes),
        "chunk_count": len(rows),
        "changed_document_count": len(changed if compatible else hashes),
        "deleted_document_count": len(deleted),
        "reused_chunk_count": len(retained_rows),
        "embedded_chunk_count": len(new_rows),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Build all LLM-Wiki V3 retrieval artifacts")
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        config = load(args.vault)
        result = build(config, full=args.full)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"wiki-embed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(
            f"OK: {result['document_count']} documents / {result['chunk_count']} chunks; "
            f"embedded={result['embedded_chunk_count']} reused={result['reused_chunk_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
